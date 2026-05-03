--------------------------------------------------------------------------------
-- gemm_os_adapter.vhd — Sequential MAC-based GEMM with memory-mapped interface
--
-- Provides the interface expected by mha_controller.vhd and ffn.vhd:
--   start/done handshake
--   Memory-mapped A/B/C ports (addr, re, data, valid)
--   Streaming output (o_data, o_valid, o_last, o_channel)
--
-- Internally performs a simple sequential multiply-accumulate over K cycles
-- per output element.  Not a systolic array — intended for PoC / simulation.
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

use work.clog2_pkg.all;

entity gemm_os_adapter is
    generic (
        DATA_WIDTH : positive := 16;
        M          : positive := 64;
        K          : positive := 512;
        N          : positive := 64
    );
    port (
        clk   : in  std_logic;
        rstn  : in  std_logic;

        start : in  std_logic;
        done  : out std_logic;

        -- A matrix read (M x K, row-major)
        a_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        a_addr  : out std_logic_vector(clog2(M * K) - 1 downto 0);
        a_re    : out std_logic;
        a_valid : in  std_logic;

        -- B matrix read (K x N, row-major)
        b_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        b_addr  : out std_logic_vector(clog2(K * N) - 1 downto 0);
        b_re    : out std_logic;
        b_valid : in  std_logic;

        -- C matrix read / bias (M x N, row-major)
        c_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        c_addr  : out std_logic_vector(clog2(M * N) - 1 downto 0);
        c_re    : out std_logic;
        c_valid : in  std_logic;

        -- Output stream
        o_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        o_valid   : out std_logic;
        o_last    : out std_logic;
        o_channel : out integer range 0 to 2**30 - 1
    );
end entity gemm_os_adapter;

architecture rtl of gemm_os_adapter is

    constant AW_A : positive := clog2(M * K);
    constant AW_B : positive := clog2(K * N);
    constant AW_C : positive := clog2(M * N);
    constant AW_O : positive := clog2(M * N);

    constant ACCUM_WIDTH : positive := 2 * DATA_WIDTH + clog2(K);

    type state_t is (ST_IDLE, ST_READ_C, ST_MAC, ST_OUTPUT, ST_DONE);
    signal state : state_t;

    signal m_cnt : integer range 0 to M - 1;
    signal n_cnt : integer range 0 to N - 1;
    signal k_cnt : integer range 0 to K - 1;

    signal accum : signed(ACCUM_WIDTH - 1 downto 0);

    signal a_reg : signed(DATA_WIDTH - 1 downto 0);
    signal b_reg : signed(DATA_WIDTH - 1 downto 0);
    signal c_reg : signed(DATA_WIDTH - 1 downto 0);

    signal a_reg_valid : std_logic;
    signal b_reg_valid : std_logic;
    signal c_reg_valid : std_logic;

    signal mac_done     : std_logic;
    signal output_cnt   : integer range 0 to M * N - 1;
    signal start_d1     : std_logic;
    signal start_pulse  : std_logic;

begin

    ---------------------------------------------------------------------------
    -- Start-pulse detection (registered)
    ---------------------------------------------------------------------------
    p_start : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                start_d1 <= '0';
            else
                start_d1 <= start;
            end if;
        end if;
    end process p_start;

    start_pulse <= '1' when start = '1' and start_d1 = '0' else '0';

    ---------------------------------------------------------------------------
    -- Main FSM
    ---------------------------------------------------------------------------
    p_fsm : process (clk) is
        variable v_prod : signed(2 * DATA_WIDTH - 1 downto 0);
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                state       <= ST_IDLE;
                m_cnt       <= 0;
                n_cnt       <= 0;
                k_cnt       <= 0;
                accum       <= (others => '0');
                a_reg       <= (others => '0');
                b_reg       <= (others => '0');
                c_reg       <= (others => '0');
                a_reg_valid <= '0';
                b_reg_valid <= '0';
                c_reg_valid <= '0';
                output_cnt  <= 0;
                o_data      <= (others => '0');
                o_valid     <= '0';
                o_last      <= '0';
                o_channel   <= 0;
                done        <= '0';
                a_re        <= '0';
                b_re        <= '0';
                c_re        <= '0';

            else
                -- Defaults
                done    <= '0';
                o_valid <= '0';
                o_last  <= '0';
                a_re    <= '0';
                b_re    <= '0';
                c_re    <= '0';

                case state is

                    when ST_IDLE =>
                        output_cnt <= 0;
                        if start_pulse = '1' then
                            state <= ST_READ_C;
                            m_cnt <= 0;
                            n_cnt <= 0;
                            k_cnt <= 0;
                            accum <= (others => '0');
                        end if;

                    -----------------------------------------------------------------
                    -- Read bias C for element (0,0) first, then move to MAC
                    -----------------------------------------------------------------
                    when ST_READ_C =>
                        c_re   <= '1';
                        c_addr <= std_logic_vector(to_unsigned(0, AW_C));
                        if c_valid = '1' then
                            c_reg       <= signed(c_data);
                            c_reg_valid <= '1';
                            state       <= ST_MAC;
                            k_cnt       <= 0;
                            -- Issue first A/B read addresses
                            a_re   <= '1';
                            a_addr <= std_logic_vector(to_unsigned(0, AW_A));
                            b_re   <= '1';
                            b_addr <= std_logic_vector(to_unsigned(0, AW_B));
                        end if;

                    -----------------------------------------------------------------
                    -- MAC loop: accumulate A(m,k) * B(k,n) over k
                    -----------------------------------------------------------------
                    when ST_MAC =>
                        -- Latch read data from previous cycle
                        if a_valid = '1' then
                            a_reg       <= signed(a_data);
                            a_reg_valid <= '1';
                        end if;
                        if b_valid = '1' then
                            b_reg       <= signed(b_data);
                            b_reg_valid <= '1';
                        end if;

                        -- MAC on registered data (delayed by one cycle from re)
                        if a_reg_valid = '1' and b_reg_valid = '1' then
                            v_prod := a_reg * b_reg;
                            accum  <= accum + resize(v_prod, ACCUM_WIDTH);
                        end if;

                        -- Advance K
                        if k_cnt = K - 1 then
                            -- MAC done for this output element
                            state       <= ST_OUTPUT;
                            a_reg_valid <= '0';
                            b_reg_valid <= '0';
                        else
                            k_cnt <= k_cnt + 1;
                            -- Issue next A/B reads
                            a_re   <= '1';
                            a_addr <= std_logic_vector(to_unsigned(m_cnt * K + k_cnt + 1, AW_A));
                            b_re   <= '1';
                            b_addr <= std_logic_vector(to_unsigned((k_cnt + 1) * N + n_cnt, AW_B));
                        end if;

                    -----------------------------------------------------------------
                    -- Output one element: C + accum, then advance to next
                    -----------------------------------------------------------------
                    when ST_OUTPUT =>
                        o_data  <= std_logic_vector(
                            resize(accum + resize(c_reg, ACCUM_WIDTH), DATA_WIDTH)
                        );
                        o_valid <= '1';
                        o_channel <= output_cnt;

                        if output_cnt = M * N - 1 then
                            o_last <= '1';
                            done   <= '1';
                            state  <= ST_DONE;
                        else
                            output_cnt <= output_cnt + 1;
                            -- Advance to next element
                            if n_cnt = N - 1 then
                                n_cnt <= 0;
                                m_cnt <= m_cnt + 1;
                            else
                                n_cnt <= n_cnt + 1;
                            end if;
                            -- Reset accum, read next C, start next MAC
                            accum <= (others => '0');
                            state <= ST_READ_C;
                        end if;

                    -----------------------------------------------------------------
                    -- DONE: one cycle, then back to IDLE
                    -----------------------------------------------------------------
                    when ST_DONE =>
                        state <= ST_IDLE;

                    when others =>
                        state <= ST_IDLE;

                end case;
            end if;
        end if;
    end process p_fsm;

end architecture rtl;
