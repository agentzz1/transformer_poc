--------------------------------------------------------------------------------
-- gemm_mm.vhd — Memory-Mapped General Matrix Multiply (GEMM)
--
-- Computes:  C = A * B + C_bias
--
-- Dimensions: A(M x K), B(K x N), C(M x N)
--
-- Interface:
--   A, B, C ports use memory-mapped read handshaking:
--     Cycle N:   gemm_mm drives (_addr, _re)
--     Cycle N+1: external memory returns (_data); user asserts _valid
--
--   Output is a streaming channel-sequential row-major stream:
--     o_valid asserted per element, o_last on final element, o_channel = flat index
--
--   Start/done handshaking: pulse start, wait for done.
--
-- Compatible with the accel library streaming conventions.
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

use work.clog2_pkg.all;

entity gemm_mm is
    generic (
        DATA_WIDTH : integer;
        M          : integer;
        K          : integer;
        N          : integer;
        max_size_x : integer := 512
    );
    port (
        clk   : in  std_logic;
        rstn  : in  std_logic;

        start : in  std_logic;
        done  : out std_logic;

        -- A matrix (M x K) row-major
        a_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        a_addr  : out std_logic_vector(clog2(M * K) - 1 downto 0);
        a_re    : out std_logic;
        a_valid : in  std_logic;

        -- B matrix (K x N) row-major
        b_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        b_addr  : out std_logic_vector(clog2(K * N) - 1 downto 0);
        b_re    : out std_logic;
        b_valid : in  std_logic;

        -- C matrix / bias (M x N) row-major
        c_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        c_addr  : out std_logic_vector(clog2(M * N) - 1 downto 0);
        c_re    : out std_logic;
        c_valid : in  std_logic;

        -- Output stream (M x N elements, row-major)
        o_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        o_valid   : out std_logic;
        o_last    : out std_logic;
        o_channel : out integer
    );
end entity gemm_mm;

architecture rtl of gemm_mm is

    ---------------------------------------------------------------------------
    -- Address widths
    ---------------------------------------------------------------------------
    constant AW_A : positive := clog2(M * K);
    constant AW_B : positive := clog2(K * N);
    constant AW_C : positive := clog2(M * N);
    constant AW_O : positive := clog2(M * N);

    ---------------------------------------------------------------------------
    -- FSM
    ---------------------------------------------------------------------------
    -- ST_INIT walks once over c_buf to zero it (one entry/cycle) so the array
    -- doesn't need a synchronous wide reset; that lets Vivado infer BRAM
    -- instead of a giant flip-flop array.
    type state_t is (ST_IDLE, ST_INIT, ST_LOAD_C, ST_MAC, ST_OUTPUT, ST_DONE);
    signal state : state_t;

    ---------------------------------------------------------------------------
    -- C accumulator buffer (M x N)
    --
    -- Width sized for int8/int16 inputs: max |sum| = K * 2^(2*(DW-1))
    --   DW=8,  K<=256:  needs 22 bits + sign  -> 24 bits comfortably
    --   DW=16, K<=256:  needs 38 bits + sign  -> 40 bits comfortably
    -- We pick max(24, 2*DW + clog2(K) + 4) so the buffer fits in BRAM rather
    -- than burning thousands of flip-flops with a 64-bit width.
    ---------------------------------------------------------------------------
    function acc_width_calc return positive is
        variable w : integer;
    begin
        w := 2 * DATA_WIDTH + clog2(K + 1) + 4;
        if w < 24 then w := 24; end if;
        return w;
    end function;
    constant ACC_WIDTH : positive := acc_width_calc;
    subtype accum_t is signed(ACC_WIDTH - 1 downto 0);
    type c_buf_t is array (0 to M * N - 1) of accum_t;
    signal c_buf : c_buf_t;

    -- INIT-phase counter (zeros c_buf one entry per cycle)
    signal init_cnt : integer range 0 to M * N := 0;

    ---------------------------------------------------------------------------
    -- MAC loop counters
    ---------------------------------------------------------------------------
    signal mac_m : integer := 0;   -- output row
    signal mac_n : integer := 0;   -- output col
    signal mac_k : integer := 0;   -- inner dimension

    -- Single accumulator register for the inner (k) loop.  Replaces a per-cycle
    -- read-modify-write on c_buf (which Vivado mapped to async distributed RAM
    -- and which then produced wrong results on real hardware).  c_buf is now
    -- written only once per (m,n) at k = K-1.
    signal mac_acc : accum_t := (others => '0');

    ---------------------------------------------------------------------------
    -- Pipelined read registers (addr/re in cycle N -> data in cycle N+1)
    ---------------------------------------------------------------------------
    signal a_data_r : signed(DATA_WIDTH - 1 downto 0);
    signal b_data_r : signed(DATA_WIDTH - 1 downto 0);
    signal c_data_r : signed(DATA_WIDTH - 1 downto 0);

    signal a_valid_r : std_logic;
    signal b_valid_r : std_logic;
    signal c_valid_r : std_logic;

    ---------------------------------------------------------------------------
    -- Registered outputs
    ---------------------------------------------------------------------------
    signal o_data_r    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal o_valid_r   : std_logic;
    signal o_last_r    : std_logic;
    signal o_channel_r : integer := 0;

    ---------------------------------------------------------------------------
    -- DSP-mapping hint for the combinational A*B multiply in p_fsm.
    -- We don't pipeline-register it here (would shift the FSM by 1 cycle);
    -- the attribute on the architecture lets Vivado decide DSP vs LUT.
    ---------------------------------------------------------------------------
    attribute use_dsp : string;
    attribute use_dsp of rtl : architecture is "yes";

    ---------------------------------------------------------------------------
    -- Flat index helpers
    ---------------------------------------------------------------------------
    signal a_flat : integer := 0;
    signal b_flat : integer := 0;
    signal c_flat : integer := 0;
    signal o_flat : integer := 0;

    function sat16 (
        value : signed
    ) return signed is
        variable max_v : signed(value'length - 1 downto 0);
        variable min_v : signed(value'length - 1 downto 0);
    begin
        max_v := to_signed(2 ** (DATA_WIDTH - 1) - 1, value'length);
        min_v := to_signed(-(2 ** (DATA_WIDTH - 1)), value'length);

        if value > max_v then
            return to_signed(2 ** (DATA_WIDTH - 1) - 1, DATA_WIDTH);
        elsif value < min_v then
            return to_signed(-(2 ** (DATA_WIDTH - 1)), DATA_WIDTH);
        end if;

        return resize(value, DATA_WIDTH);
    end function;

begin

    ---------------------------------------------------------------------------
    -- Flattened address computation
    ---------------------------------------------------------------------------
    a_flat <= mac_m * K + mac_k;
    b_flat <= mac_k * N + mac_n;
    c_flat <= mac_m * N + mac_n;
    o_flat <= mac_m * N + mac_n;

    ---------------------------------------------------------------------------
    -- Memory interface: combinational addr/re
    ---------------------------------------------------------------------------
    a_addr <= std_logic_vector(to_unsigned(a_flat, AW_A));
    b_addr <= std_logic_vector(to_unsigned(b_flat, AW_B));
    c_addr <= std_logic_vector(to_unsigned(c_flat, AW_C));
    a_re   <= '1' when state = ST_MAC else '0';
    b_re   <= '1' when state = ST_MAC else '0';
    c_re   <= '1' when state = ST_LOAD_C else '0';

    ---------------------------------------------------------------------------
    -- Registered read data (cycle N addr/re -> cycle N+1 data/valid)
    ---------------------------------------------------------------------------
    p_read_regs : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                a_data_r  <= (others => '0');
                b_data_r  <= (others => '0');
                c_data_r  <= (others => '0');
                a_valid_r <= '0';
                b_valid_r <= '0';
                c_valid_r <= '0';
            else
                a_data_r  <= signed(a_data);
                b_data_r  <= signed(b_data);
                c_data_r  <= signed(c_data);
                a_valid_r <= a_valid;
                b_valid_r <= b_valid;
                c_valid_r <= c_valid;
            end if;
        end if;
    end process p_read_regs;

    ---------------------------------------------------------------------------
    -- Main FSM: MAC accumulation then streaming output
    ---------------------------------------------------------------------------
    p_fsm : process (clk) is
        variable v_product : signed(2 * DATA_WIDTH - 1 downto 0);
        variable v_acc     : accum_t;
        variable v_idx     : integer;
        variable v_result  : signed(DATA_WIDTH - 1 downto 0);
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                -- Note: c_buf is NOT reset here -- a wide reset prevents BRAM
                -- inference. c_buf is zeroed sequentially in ST_INIT instead.
                state     <= ST_IDLE;
                mac_m     <= 0;
                mac_n     <= 0;
                mac_k     <= 0;
                mac_acc   <= (others => '0');
                init_cnt  <= 0;
                o_data_r  <= (others => '0');
                o_valid_r <= '0';
                o_last_r  <= '0';
                o_channel_r <= 0;
                done      <= '0';
            else
                -- One-cycle defaults
                o_valid_r <= '0';
                o_last_r  <= '0';
                done      <= '0';

                case state is

                    -----------------------------------------------------------
                    -- IDLE: wait for start pulse
                    -----------------------------------------------------------
                    when ST_IDLE =>
                        mac_m    <= 0;
                        mac_n    <= 0;
                        mac_k    <= 0;
                        init_cnt <= 0;
                        if start = '1' then
                            state <= ST_INIT;
                        end if;

                    -----------------------------------------------------------
                    -- INIT: zero c_buf one entry per cycle (M*N cycles)
                    -- Sequential single-port write -> BRAM-inferable.
                    -----------------------------------------------------------
                    when ST_INIT =>
                        c_buf(init_cnt) <= (others => '0');
                        if init_cnt = M * N - 1 then
                            init_cnt <= 0;
                            state    <= ST_LOAD_C;
                        else
                            init_cnt <= init_cnt + 1;
                        end if;

                    -----------------------------------------------------------
                    -- LOAD_C: read bias/initial C values (M x N cycles)
                    --
                    -- We issue c_re for each C element and capture it into
                    -- c_buf as the initial accumulator value.
                    -----------------------------------------------------------
                    when ST_LOAD_C =>
                        if c_valid = '1' then
                            c_buf(c_flat) <= shift_left(resize(signed(c_data), ACC_WIDTH), DATA_WIDTH - 1);
                            if mac_n = N - 1 then
                                mac_n <= 0;
                                if mac_m = M - 1 then
                                    mac_m <= 0;
                                    mac_n <= 0;
                                    mac_k <= 0;
                                    state <= ST_MAC;
                                else
                                    mac_m <= mac_m + 1;
                                end if;
                            else
                                mac_n <= mac_n + 1;
                            end if;
                        end if;

                    -----------------------------------------------------------
                    -- MAC: accumulate A(m,k) * B(k,n) across k for each (m,n)
                    --
                    -- Cycle N: issue a_re, b_re for (m,k) and (k,n)
                    -- Cycle N+1: registered data arrives; multiply and accumulate
                    --
                    -- We run K passes per (m,n), then advance to next element.
                    -- The total is M*N*K MAC operations.
                    -----------------------------------------------------------
                    when ST_MAC =>
                        if a_valid = '1' and b_valid = '1' then
                            v_product := signed(a_data) * signed(b_data);
                            v_idx     := mac_m * N + mac_n;
                            -- Accumulate in a single register across the k-loop.
                            -- At k=0 start from the bias held in c_buf (loaded in
                            -- ST_LOAD_C); afterwards add into mac_acc.  Write c_buf
                            -- back ONCE at k=K-1.  No per-cycle RMW on c_buf.
                            if mac_k = 0 then
                                v_acc := c_buf(v_idx) + resize(v_product, ACC_WIDTH);
                            else
                                v_acc := mac_acc + resize(v_product, ACC_WIDTH);
                            end if;
                            mac_acc <= v_acc;

                            if mac_k = K - 1 then
                                c_buf(v_idx) <= v_acc;
                                mac_k <= 0;
                                if mac_n = N - 1 then
                                    mac_n <= 0;
                                    if mac_m = M - 1 then
                                        mac_m <= 0;
                                        state <= ST_OUTPUT;
                                    else
                                        mac_m <= mac_m + 1;
                                    end if;
                                else
                                    mac_n <= mac_n + 1;
                                end if;
                            else
                                mac_k <= mac_k + 1;
                            end if;
                        end if;

                    -----------------------------------------------------------
                    -- OUTPUT: stream accumulated results
                    --
                    -- One element per cycle, row-major, o_valid asserted.
                    -- o_last on the final element.
                    -----------------------------------------------------------
                    when ST_OUTPUT =>
                        v_idx    := mac_m * N + mac_n;
                        v_result := sat16(shift_right(c_buf(v_idx), DATA_WIDTH - 1));

                        o_data_r    <= std_logic_vector(v_result);
                        o_valid_r   <= '1';
                        o_channel_r <= v_idx;

                        if mac_n = N - 1 and mac_m = M - 1 then
                            o_last_r <= '1';
                            state    <= ST_DONE;
                        else
                            if mac_n = N - 1 then
                                mac_n <= 0;
                                mac_m <= mac_m + 1;
                            else
                                mac_n <= mac_n + 1;
                            end if;
                        end if;

                    -----------------------------------------------------------
                    -- DONE: assert done for one cycle, return to IDLE
                    -----------------------------------------------------------
                    when ST_DONE =>
                        done  <= '1';
                        state <= ST_IDLE;

                    when others =>
                        state <= ST_IDLE;

                end case;
            end if;
        end if;
    end process p_fsm;

    ---------------------------------------------------------------------------
    -- Output assignments
    ---------------------------------------------------------------------------
    o_data    <= o_data_r;
    o_valid   <= o_valid_r;
    o_last    <= o_last_r;
    o_channel <= o_channel_r;

end architecture rtl;
