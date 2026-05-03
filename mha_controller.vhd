--------------------------------------------------------------------------------
-- mha_controller.vhd -- Multi-Head Self-Attention Controller
--
-- Post-LN Transformer encoder block.  Sequential head-by-head processing
-- with gemm_os instances for Q / K / V / O projections.
--
-- FSM:
--   IDLE -> BUFFER_INPUT -> PROJ_Q -> PROJ_K -> PROJ_V ->
--   COMPUTE_SCORES -> APPLY_SOFTMAX -> ATTEND_V ->
--   [next head, back to PROJ_Q] -> OUTPUT_PROJ -> DONE
--
-- Weight memory layout (row-major):
--   W_Q, W_K, W_V  each MODEL_DIM x MODEL_DIM = 512 x 512.
--   Per-head slice h:  columns [h*HEAD_DIM .. (h+1)*HEAD_DIM-1] across all rows.
--   W_O  MODEL_DIM x MODEL_DIM, no slicing.
--
-- Memory read timing  (matching ffn.vhd convention):
--   Cycle N:   gemm_os drives (_addr, _re) -- these are registered inside gemm_os
--              Combinational address mapping to external memory _addr / _re
--   Cycle N+1: external weight memory or internal buffer provides _data
--              Registered (_data -> _data_r), presented to gemm_os _valid=1
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;
    use ieee.math_real.all;

use work.clog2_pkg.all;
use work.utilities.all;

entity mha_controller is
    generic (
        DATA_WIDTH : positive := 16;
        NUM_HEADS  : positive := 8;
        HEAD_DIM   : positive := 64;
        SEQ_LEN    : positive := 64;
        MODEL_DIM  : positive := 512
    );
    port (
        clk  : in std_logic;
        rstn : in std_logic;

        start : in std_logic;
        done  : out std_logic;

        -- Streaming input  (SEQ_LEN * MODEL_DIM elements)
        i_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        i_valid   : in  std_logic;
        i_last    : in  std_logic;
        i_channel : in  integer;

        -- Streaming output  (SEQ_LEN * MODEL_DIM elements)
        o_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        o_valid   : out std_logic;
        o_last    : out std_logic;
        o_channel : out integer;

        -- W_Q weight memory  (MODEL_DIM x MODEL_DIM)
        w_q_addr : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
        w_q_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        w_q_re   : out std_logic;

        -- W_K weight memory  (MODEL_DIM x MODEL_DIM)
        w_k_addr : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
        w_k_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        w_k_re   : out std_logic;

        -- W_V weight memory  (MODEL_DIM x MODEL_DIM)
        w_v_addr : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
        w_v_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        w_v_re   : out std_logic;

        -- W_O output projection  (MODEL_DIM x MODEL_DIM)
        w_o_addr : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
        w_o_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        w_o_re   : out std_logic
    );
end entity mha_controller;

architecture rtl of mha_controller is

    ---------------------------------------------------------------------------
    -- FSM state type
    ---------------------------------------------------------------------------
    type state_t is (
        ST_IDLE,
        ST_BUFFER_INPUT,
        ST_PROJ_Q,
        ST_PROJ_K,
        ST_PROJ_V,
        ST_COMPUTE_SCORES,
        ST_APPLY_SOFTMAX,
        ST_ATTEND_V,
        ST_OUTPUT_PROJ,
        ST_DONE
    );

    signal state : state_t;

    ---------------------------------------------------------------------------
    -- gemm_os component declaration  (memory-mapped, matching ffn.vhd usage)
    --
    -- gemm_os internally uses ROWS=M, COLS=N, and expects A(MxK), B(KxN).
    -- All ports use _addr / _re / _data / _valid handshake:
    --   Cycle N:   gemm_os drives (_addr, _re)
    --   Cycle N+1: user provides (_data, _valid)
    ---------------------------------------------------------------------------
    component gemm_mm is
        generic (
            DATA_WIDTH : integer;
            M          : integer;
            K          : integer;
            N          : integer
        );
        port (
            clk       : in  std_logic;
            rstn      : in  std_logic;
            start     : in  std_logic;
            done      : out std_logic;

            a_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            a_addr    : out std_logic_vector(clog2(M * K) - 1 downto 0);
            a_re      : out std_logic;
            a_valid   : in  std_logic;

            b_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            b_addr    : out std_logic_vector(clog2(K * N) - 1 downto 0);
            b_re      : out std_logic;
            b_valid   : in  std_logic;

            c_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            c_addr    : out std_logic_vector(clog2(M * N) - 1 downto 0);
            c_re      : out std_logic;
            c_valid   : in  std_logic;

            o_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
            o_valid   : out std_logic;
            o_last    : out std_logic;
            o_channel : out integer
        );
    end component;

    ---------------------------------------------------------------------------
    -- softmax component declaration
    ---------------------------------------------------------------------------
    component softmax is
        generic (
            DATA_WIDTH : positive := 16;
            SEQ_LEN    : positive := 64
        );
        port (
            clk       : in  std_logic;
            rstn      : in  std_logic;
            i_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            i_valid   : in  std_logic;
            i_last    : in  std_logic;
            i_channel : in  integer;
            o_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
            o_valid   : out std_logic;
            o_last    : out std_logic;
            o_channel : out integer
        );
    end component;

    ---------------------------------------------------------------------------
    -- Derived constants
    ---------------------------------------------------------------------------
    constant AW_W_FULL : positive := clog2(MODEL_DIM * MODEL_DIM);

    -- Head-projection gemm_os: M=SEQ_LEN, K=MODEL_DIM, N=HEAD_DIM
    constant AW_A_HD : positive := clog2(SEQ_LEN * MODEL_DIM);
    constant AW_B_HD : positive := clog2(MODEL_DIM * HEAD_DIM);
    constant AW_C_HD : positive := clog2(SEQ_LEN * HEAD_DIM);

    -- Output-projection gemm_os: M=SEQ_LEN, K=MODEL_DIM, N=MODEL_DIM
    constant AW_A_OUT : positive := clog2(SEQ_LEN * MODEL_DIM);
    constant AW_B_OUT : positive := clog2(MODEL_DIM * MODEL_DIM);
    constant AW_C_OUT : positive := clog2(SEQ_LEN * MODEL_DIM);

    -- sqrt(HEAD_DIM) = sqrt(64) = 8 -> right shift by 3
    constant LOG_SQRT_HD : natural := integer(ceil(log2(real(HEAD_DIM)))) / 2;

    ---------------------------------------------------------------------------
    -- Internal buffer types
    ---------------------------------------------------------------------------
    type buf_2d_slm_t is array (0 to SEQ_LEN - 1, 0 to MODEL_DIM - 1)
        of std_logic_vector(DATA_WIDTH - 1 downto 0);

    type buf_2d_shd_t is array (0 to SEQ_LEN - 1, 0 to HEAD_DIM - 1)
        of std_logic_vector(DATA_WIDTH - 1 downto 0);

    type buf_2d_ss_t is array (0 to SEQ_LEN - 1, 0 to SEQ_LEN - 1)
        of std_logic_vector(DATA_WIDTH - 1 downto 0);

    ---------------------------------------------------------------------------
    -- Internal storage
    ---------------------------------------------------------------------------
    signal input_buf : buf_2d_slm_t;
    signal input_cnt : integer := 0;

    signal Q_buf      : buf_2d_shd_t;
    signal K_buf      : buf_2d_shd_t;
    signal V_buf      : buf_2d_shd_t;
    signal S_buf      : buf_2d_ss_t;
    signal concat_buf : buf_2d_slm_t;

    ---------------------------------------------------------------------------
    -- Control counters
    ---------------------------------------------------------------------------
    signal head_idx : integer := 0;

    -- COMPUTE_SCORES triple loop
    signal cs_i : integer := 0;
    signal cs_j : integer := 0;
    signal cs_d : integer := 0;
    signal cs_acc : signed(63 downto 0) := (others => '0');

    -- APPLY_SOFTMAX row and feed/capture state
    signal as_row        : integer := 0;
    signal sm_feed_cnt   : integer := 0;
    signal sm_cap_cnt    : integer := 0;
    signal sm_feeding    : std_logic;
    signal sm_capturing  : std_logic;

    -- ATTEND_V triple loop
    signal av_i : integer := 0;
    signal av_d : integer := 0;
    signal av_j : integer := 0;
    signal av_acc : signed(63 downto 0) := (others => '0');

    ---------------------------------------------------------------------------
    -- Head-projection gemm_os  (Q / K / V, M=SEQ_LEN, K=MODEL_DIM, N=HEAD_DIM)
    ---------------------------------------------------------------------------
    signal hd_gemm_start   : std_logic;
    signal hd_gemm_done    : std_logic;
    signal hd_gemm_a_addr  : std_logic_vector(AW_A_HD - 1 downto 0);
    signal hd_gemm_a_data  : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal hd_gemm_a_re    : std_logic;
    signal hd_gemm_a_valid : std_logic;
    signal hd_gemm_b_addr  : std_logic_vector(AW_B_HD - 1 downto 0);
    signal hd_gemm_b_data  : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal hd_gemm_b_re    : std_logic;
    signal hd_gemm_b_valid : std_logic;
    signal hd_gemm_c_addr  : std_logic_vector(AW_C_HD - 1 downto 0);
    signal hd_gemm_c_data  : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal hd_gemm_c_re    : std_logic;
    signal hd_gemm_c_valid : std_logic;
    signal hd_gemm_o_data  : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal hd_gemm_o_valid : std_logic;
    signal hd_gemm_o_last  : std_logic;
    signal hd_gemm_o_chan  : integer := 0;
    signal hd_cap_cnt      : integer := 0;
    signal hd_gemm_active  : std_logic;

    -- Pipelined read data registers (addr/re in cycle N -> data in cycle N+1)
    signal hd_a_data_r : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal hd_b_data_r : std_logic_vector(DATA_WIDTH - 1 downto 0);

    ---------------------------------------------------------------------------
    -- Output-projection gemm_os  (W_O, M=SEQ_LEN, K=MODEL_DIM, N=MODEL_DIM)
    ---------------------------------------------------------------------------
    signal out_gemm_start   : std_logic;
    signal out_gemm_done    : std_logic;
    signal out_gemm_a_addr  : std_logic_vector(AW_A_OUT - 1 downto 0);
    signal out_gemm_a_data  : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal out_gemm_a_re    : std_logic;
    signal out_gemm_a_valid : std_logic;
    signal out_gemm_b_addr  : std_logic_vector(AW_B_OUT - 1 downto 0);
    signal out_gemm_b_data  : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal out_gemm_b_re    : std_logic;
    signal out_gemm_b_valid : std_logic;
    signal out_gemm_c_addr  : std_logic_vector(AW_C_OUT - 1 downto 0);
    signal out_gemm_c_data  : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal out_gemm_c_re    : std_logic;
    signal out_gemm_c_valid : std_logic;
    signal out_gemm_o_data  : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal out_gemm_o_valid : std_logic;
    signal out_gemm_o_last  : std_logic;
    signal out_gemm_o_chan  : integer := 0;
    signal out_cap_cnt      : integer := 0;
    signal out_gemm_active  : std_logic;

    -- Pipelined read data registers
    signal out_a_data_r : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal out_b_data_r : std_logic_vector(DATA_WIDTH - 1 downto 0);

    ---------------------------------------------------------------------------
    -- softmax interface
    ---------------------------------------------------------------------------
    signal sm_i_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal sm_i_valid   : std_logic;
    signal sm_i_last    : std_logic;
    signal sm_i_channel : integer := 0;
    signal sm_o_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal sm_o_valid   : std_logic;
    signal sm_o_last    : std_logic;
    signal sm_o_channel : integer := 0;

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
    -- Done output
    ---------------------------------------------------------------------------
    done <= '1' when state = ST_DONE else '0';

    ---------------------------------------------------------------------------
    -- Head-projection GEMM data.
    ---------------------------------------------------------------------------
    p_hd_data_comb : process (all) is
        variable v_flat : natural range 0 to SEQ_LEN * MODEL_DIM - 1;
        variable v_row  : integer;
        variable v_col  : integer;
    begin
        hd_gemm_a_data <= (others => '0');
        hd_gemm_b_data <= (others => '0');

        if hd_gemm_a_re = '1' then
            v_flat := to_integer(unsigned(hd_gemm_a_addr));
            v_row  := v_flat / MODEL_DIM;
            v_col  := v_flat mod MODEL_DIM;
            hd_gemm_a_data <= input_buf(v_row, v_col);
        end if;

        if state = ST_PROJ_Q then
            hd_gemm_b_data <= w_q_data;
        elsif state = ST_PROJ_K then
            hd_gemm_b_data <= w_k_data;
        elsif state = ST_PROJ_V then
            hd_gemm_b_data <= w_v_data;
        end if;
    end process p_hd_data_comb;

    ---------------------------------------------------------------------------
    -- Output-projection GEMM data.
    ---------------------------------------------------------------------------
    p_out_data_comb : process (all) is
        variable v_flat : natural range 0 to SEQ_LEN * MODEL_DIM - 1;
        variable v_row  : integer;
        variable v_col  : integer;
    begin
        out_gemm_a_data <= (others => '0');
        out_gemm_b_data <= (others => '0');

        if out_gemm_a_re = '1' then
            v_flat := to_integer(unsigned(out_gemm_a_addr));
            v_row  := v_flat / MODEL_DIM;
            v_col  := v_flat mod MODEL_DIM;
            out_gemm_a_data <= concat_buf(v_row, v_col);
        end if;

        if state = ST_OUTPUT_PROJ then
            out_gemm_b_data <= w_o_data;
        end if;
    end process p_out_data_comb;

    ---------------------------------------------------------------------------
    -- Head-projection A-port: pipelined read from input_buf
    --   addr -> (row, col) -> registered data -> gemm_os
    --
    -- Head-projection B-port: pipelined read from external weight memory.
    --   The weight _addr/_re are driven combinationally from the gemm_os
    --   outputs (see w_q_addr/w_q_re assignments below), so the external
    --   memory returns data the following cycle.  We register it here to
    --   align with the gemm_os handshake expectation.
    ---------------------------------------------------------------------------
    p_hd_read : process (clk) is
        variable v_flat : natural range 0 to SEQ_LEN * MODEL_DIM - 1;
        variable v_row  : integer;
        variable v_col  : integer;
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                hd_a_data_r <= (others => '0');
                hd_b_data_r <= (others => '0');
            else
                -- A-port: from input_buf
                if (state = ST_PROJ_Q or state = ST_PROJ_K or state = ST_PROJ_V)
                    and hd_gemm_a_re = '1' then
                    v_flat := to_integer(unsigned(hd_gemm_a_addr));
                    v_row  := v_flat / MODEL_DIM;
                    v_col  := v_flat mod MODEL_DIM;
                    hd_a_data_r <= input_buf(v_row, v_col);
                else
                    hd_a_data_r <= (others => '0');
                end if;

                -- B-port: from external weight memory (address mapping is combinational)
                if state = ST_PROJ_Q then
                    hd_b_data_r <= w_q_data;
                elsif state = ST_PROJ_K then
                    hd_b_data_r <= w_k_data;
                elsif state = ST_PROJ_V then
                    hd_b_data_r <= w_v_data;
                else
                    hd_b_data_r <= (others => '0');
                end if;
            end if;
        end if;
    end process p_hd_read;

    ---------------------------------------------------------------------------
    -- Output-projection A-port: pipelined read from concat_buf
    -- Output-projection B-port: pipelined read from external W_O memory
    ---------------------------------------------------------------------------
    p_out_read : process (clk) is
        variable v_flat : natural range 0 to SEQ_LEN * MODEL_DIM - 1;
        variable v_row  : integer;
        variable v_col  : integer;
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                out_a_data_r <= (others => '0');
                out_b_data_r <= (others => '0');
            else
                -- A-port: from concat_buf
                if state = ST_OUTPUT_PROJ and out_gemm_a_re = '1' then
                    v_flat := to_integer(unsigned(out_gemm_a_addr));
                    v_row  := v_flat / MODEL_DIM;
                    v_col  := v_flat mod MODEL_DIM;
                    out_a_data_r <= concat_buf(v_row, v_col);
                else
                    out_a_data_r <= (others => '0');
                end if;

                -- B-port: from external W_O memory
                if state = ST_OUTPUT_PROJ then
                    out_b_data_r <= w_o_data;
                else
                    out_b_data_r <= (others => '0');
                end if;
            end if;
        end if;
    end process p_out_read;

    ---------------------------------------------------------------------------
    -- Weight-memory address mapping  (combinational)
    --
    -- For head-projection Q / K / V:
    --   gemm_os B-port is (MODEL_DIM x HEAD_DIM) row-major.
    --   b_addr = k * HEAD_DIM + n   (k in 0..MODEL_DIM-1, n in 0..HEAD_DIM-1)
    --
    --   Physical weight memory is (MODEL_DIM x MODEL_DIM) row-major.
    --   For head H, columns [H*HEAD_DIM .. (H+1)*HEAD_DIM-1]:
    --     w_addr = k * MODEL_DIM + H * HEAD_DIM + n
    --
    -- For output-projection W_O:
    --   gemm_os B-port is (MODEL_DIM x MODEL_DIM) -- direct match.
    --   w_o_addr = b_addr  (passthrough)
    --
    -- These are combinational so the external memory sees the correct
    -- address in the SAME cycle as gemm_os asserts _re.
    ---------------------------------------------------------------------------
    p_weight_map : process (all) is
        variable v_b_flat : natural range 0 to MODEL_DIM * HEAD_DIM - 1;
        variable v_k      : integer;
        variable v_n      : integer;
        variable v_w_phys : natural range 0 to MODEL_DIM * MODEL_DIM - 1;
    begin
        -- Defaults: zero, read disabled
        w_q_addr <= (others => '0');
        w_k_addr <= (others => '0');
        w_v_addr <= (others => '0');
        w_o_addr <= (others => '0');
        w_q_re   <= '0';
        w_k_re   <= '0';
        w_v_re   <= '0';
        w_o_re   <= '0';

        -- Q projection
        if state = ST_PROJ_Q and hd_gemm_b_re = '1' then
            v_b_flat  := to_integer(unsigned(hd_gemm_b_addr));
            v_k       := v_b_flat / HEAD_DIM;
            v_n       := v_b_flat mod HEAD_DIM;
            v_w_phys  := v_k * MODEL_DIM + head_idx * HEAD_DIM + v_n;
            w_q_addr  <= std_logic_vector(to_unsigned(v_w_phys, AW_W_FULL));
            w_q_re    <= '1';
        end if;

        -- K projection
        if state = ST_PROJ_K and hd_gemm_b_re = '1' then
            v_b_flat  := to_integer(unsigned(hd_gemm_b_addr));
            v_k       := v_b_flat / HEAD_DIM;
            v_n       := v_b_flat mod HEAD_DIM;
            v_w_phys  := v_k * MODEL_DIM + head_idx * HEAD_DIM + v_n;
            w_k_addr  <= std_logic_vector(to_unsigned(v_w_phys, AW_W_FULL));
            w_k_re    <= '1';
        end if;

        -- V projection
        if state = ST_PROJ_V and hd_gemm_b_re = '1' then
            v_b_flat  := to_integer(unsigned(hd_gemm_b_addr));
            v_k       := v_b_flat / HEAD_DIM;
            v_n       := v_b_flat mod HEAD_DIM;
            v_w_phys  := v_k * MODEL_DIM + head_idx * HEAD_DIM + v_n;
            w_v_addr  <= std_logic_vector(to_unsigned(v_w_phys, AW_W_FULL));
            w_v_re    <= '1';
        end if;

        -- Output projection  (W_O, direct passthrough)
        if state = ST_OUTPUT_PROJ and out_gemm_b_re = '1' then
            w_o_addr  <= out_gemm_b_addr;
            w_o_re    <= '1';
        end if;
    end process p_weight_map;

    ---------------------------------------------------------------------------
    -- Head-projection gemm_os instance  (Q / K / V, one head at a time)
    ---------------------------------------------------------------------------
    u_hd_gemm : gemm_mm
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            M          => SEQ_LEN,
            K          => MODEL_DIM,
            N          => HEAD_DIM
        )
        port map (
            clk       => clk,
            rstn      => rstn,
            start     => hd_gemm_start,
            done      => hd_gemm_done,
            a_data    => hd_gemm_a_data,
            a_addr    => hd_gemm_a_addr,
            a_re      => hd_gemm_a_re,
            a_valid   => hd_gemm_a_valid,
            b_data    => hd_gemm_b_data,
            b_addr    => hd_gemm_b_addr,
            b_re      => hd_gemm_b_re,
            b_valid   => hd_gemm_b_valid,
            c_data    => hd_gemm_c_data,
            c_addr    => hd_gemm_c_addr,
            c_re      => hd_gemm_c_re,
            c_valid   => hd_gemm_c_valid,
            o_data    => hd_gemm_o_data,
            o_valid   => hd_gemm_o_valid,
            o_last    => hd_gemm_o_last,
            o_channel => hd_gemm_o_chan
        );

    ---------------------------------------------------------------------------
    -- Output-projection gemm_os instance  (W_O)
    ---------------------------------------------------------------------------
    u_out_gemm : gemm_mm
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            M          => SEQ_LEN,
            K          => MODEL_DIM,
            N          => MODEL_DIM
        )
        port map (
            clk       => clk,
            rstn      => rstn,
            start     => out_gemm_start,
            done      => out_gemm_done,
            a_data    => out_gemm_a_data,
            a_addr    => out_gemm_a_addr,
            a_re      => out_gemm_a_re,
            a_valid   => out_gemm_a_valid,
            b_data    => out_gemm_b_data,
            b_addr    => out_gemm_b_addr,
            b_re      => out_gemm_b_re,
            b_valid   => out_gemm_b_valid,
            c_data    => out_gemm_c_data,
            c_addr    => out_gemm_c_addr,
            c_re      => out_gemm_c_re,
            c_valid   => out_gemm_c_valid,
            o_data    => out_gemm_o_data,
            o_valid   => out_gemm_o_valid,
            o_last    => out_gemm_o_last,
            o_channel => out_gemm_o_chan
        );

    ---------------------------------------------------------------------------
    -- softmax instance
    ---------------------------------------------------------------------------
    u_softmax : softmax
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            SEQ_LEN    => SEQ_LEN
        )
        port map (
            clk       => clk,
            rstn      => rstn,
            i_data    => sm_i_data,
            i_valid   => sm_i_valid,
            i_last    => sm_i_last,
            i_channel => sm_i_channel,
            o_data    => sm_o_data,
            o_valid   => sm_o_valid,
            o_last    => sm_o_last,
            o_channel => sm_o_channel
        );

    ---------------------------------------------------------------------------
    -- MAIN FSM  (single-process, all registered state)
    ---------------------------------------------------------------------------
    p_fsm : process (clk) is

        variable v_acc  : signed(63 downto 0);
        variable v_prod : signed(2 * DATA_WIDTH - 1 downto 0);
        variable v_row  : integer;
        variable v_col  : integer;

    begin
        if rising_edge(clk) then
            if rstn = '0' then
                state           <= ST_IDLE;
                head_idx        <= 0;
                input_cnt       <= 0;
                cs_i            <= 0;
                cs_j            <= 0;
                cs_d            <= 0;
                cs_acc          <= (others => '0');
                as_row          <= 0;
                av_i            <= 0;
                av_d            <= 0;
                av_j            <= 0;
                av_acc          <= (others => '0');
                hd_gemm_start   <= '0';
                hd_gemm_a_valid <= '0';
                hd_gemm_b_valid <= '0';
                hd_gemm_c_valid <= '0';
                hd_gemm_c_data  <= (others => '0');
                hd_cap_cnt      <= 0;
                hd_gemm_active   <= '0';
                out_gemm_active  <= '0';
                out_gemm_start   <= '0';
                out_gemm_a_valid <= '0';
                out_gemm_b_valid <= '0';
                out_gemm_c_valid <= '0';
                out_gemm_c_data  <= (others => '0');
                sm_i_valid      <= '0';
                sm_i_last       <= '0';
                sm_i_data       <= (others => '0');
                sm_i_channel    <= 0;
                sm_feeding      <= '0';
                sm_capturing    <= '0';
                sm_feed_cnt     <= 0;
                sm_cap_cnt      <= 0;
                o_data          <= (others => '0');
                o_valid         <= '0';
                o_last          <= '0';
                o_channel       <= 0;

            else
                -----------------------------------------------------------------
                -- One-cycle defaults: pulse signals auto-deassert
                -----------------------------------------------------------------
                hd_gemm_start  <= '0';
                out_gemm_start <= '0';
                sm_i_valid     <= '0';
                sm_i_last      <= '0';
                o_valid        <= '0';
                o_last         <= '0';

                -----------------------------------------------------------------
                -- State machine
                -----------------------------------------------------------------
                case state is

                    ------------------------------------------------------------
                    -- ST_IDLE  -- wait for start pulse
                    ------------------------------------------------------------
                    when ST_IDLE =>
                        input_cnt <= 0;
                        head_idx  <= 0;
                        if start = '1' then
                            report "MHA: Starting ST_BUFFER_INPUT" severity note;
                            if i_valid = '1' then
                                v_row := 0;
                                v_col := 0;
                                input_buf(v_row, v_col) <= i_data;
                                input_cnt <= 1;
                                if SEQ_LEN * MODEL_DIM = 1 then
                                    state <= ST_PROJ_Q;
                                else
                                    state <= ST_BUFFER_INPUT;
                                end if;
                            else
                                state <= ST_BUFFER_INPUT;
                            end if;
                        end if;

                    ------------------------------------------------------------
                    -- ST_BUFFER_INPUT  -- capture SEQ_LEN x MODEL_DIM elements
                    ------------------------------------------------------------
                    when ST_BUFFER_INPUT =>
                        if i_valid = '1' then
                            v_row := input_cnt / MODEL_DIM;
                            v_col := input_cnt mod MODEL_DIM;
                            input_buf(v_row, v_col) <= i_data;
                            if input_cnt = SEQ_LEN * MODEL_DIM - 1 then
                                input_cnt <= 0;
                                state     <= ST_PROJ_Q;
                                report "MHA: Finished buffering input, starting Head 0" severity note;
                            else
                                input_cnt <= input_cnt + 1;
                                if input_cnt >= 224 then
                                    report "MHA: Captured element " & integer'image(input_cnt) severity note;
                                end if;
                                if (input_cnt + 1) mod 32 = 0 then
                                    report "MHA: Buffered elements: " & integer'image(input_cnt + 1) severity note;
                                end if;
                            end if;
                        end if;

                    ------------------------------------------------------------
                    -- ST_PROJ_Q  -- Q = X * W_Q[head]
                    ------------------------------------------------------------
                    when ST_PROJ_Q =>
                        hd_gemm_a_valid <= '1';
                        hd_gemm_b_valid <= '1';
                        hd_gemm_c_valid <= '1';
                        hd_gemm_c_data  <= (others => '0');   -- no bias

                        -- Single-cycle start pulse
                        if hd_gemm_start = '0' and hd_gemm_done = '0' then
                            hd_gemm_start <= '1';
                            hd_cap_cnt    <= 0;
                        end if;

                        -- Capture matrix multiply output stream into Q_buf
                        if hd_gemm_o_valid = '1' then
                            if hd_cap_cnt < SEQ_LEN * HEAD_DIM then
                                v_row := hd_cap_cnt / HEAD_DIM;
                                v_col := hd_cap_cnt mod HEAD_DIM;
                                Q_buf(v_row, v_col) <= hd_gemm_o_data;
                                hd_cap_cnt <= hd_cap_cnt + 1;
                            end if;
                        end if;

                        if hd_gemm_done = '1' then
                            state           <= ST_PROJ_K;
                            hd_gemm_active  <= '0';
                            hd_gemm_a_valid <= '0';
                            hd_gemm_b_valid <= '0';
                            hd_gemm_c_valid <= '0';
                            hd_cap_cnt      <= 0;
                            report "MHA: Head " & integer'image(head_idx) & " Q-Projection finished" severity note;
                        end if;

                    ------------------------------------------------------------
                    -- ST_PROJ_K  -- K = X * W_K[head]
                    ------------------------------------------------------------
                    when ST_PROJ_K =>
                        hd_gemm_a_valid <= '1';
                        hd_gemm_b_valid <= '1';
                        hd_gemm_c_valid <= '1';
                        hd_gemm_c_data  <= (others => '0');

                        if hd_gemm_active = '0' and hd_gemm_done = '0' then
                            hd_gemm_start  <= '1';
                            hd_gemm_active <= '1';
                            hd_cap_cnt     <= 0;
                        end if;

                        -- Capture matrix multiply output stream into K_buf
                        if hd_gemm_o_valid = '1' then
                            if hd_cap_cnt < SEQ_LEN * HEAD_DIM then
                                v_row := hd_cap_cnt / HEAD_DIM;
                                v_col := hd_cap_cnt mod HEAD_DIM;
                                K_buf(v_row, v_col) <= hd_gemm_o_data;
                                hd_cap_cnt <= hd_cap_cnt + 1;
                            end if;
                        end if;

                        if hd_gemm_done = '1' then
                            state           <= ST_PROJ_V;
                            hd_gemm_active  <= '0';
                            hd_gemm_a_valid <= '0';
                            hd_gemm_b_valid <= '0';
                            hd_gemm_c_valid <= '0';
                            hd_cap_cnt      <= 0;
                            report "MHA: Head " & integer'image(head_idx) & " K-Projection finished" severity note;
                        end if;

                    ------------------------------------------------------------
                    -- ST_PROJ_V  -- V = X * W_V[head]
                    ------------------------------------------------------------
                    when ST_PROJ_V =>
                        hd_gemm_a_valid <= '1';
                        hd_gemm_b_valid <= '1';
                        hd_gemm_c_valid <= '1';
                        hd_gemm_c_data  <= (others => '0');

                        if hd_gemm_active = '0' and hd_gemm_done = '0' then
                            hd_gemm_start  <= '1';
                            hd_gemm_active <= '1';
                            hd_cap_cnt     <= 0;
                        end if;

                        -- Capture matrix multiply output stream into V_buf
                        if hd_gemm_o_valid = '1' then
                            if hd_cap_cnt < SEQ_LEN * HEAD_DIM then
                                v_row := hd_cap_cnt / HEAD_DIM;
                                v_col := hd_cap_cnt mod HEAD_DIM;
                                V_buf(v_row, v_col) <= hd_gemm_o_data;
                                hd_cap_cnt <= hd_cap_cnt + 1;
                            end if;
                        end if;

                        if hd_gemm_done = '1' then
                            state           <= ST_COMPUTE_SCORES;
                            cs_i            <= 0;
                            cs_j            <= 0;
                            cs_d            <= 0;
                            cs_acc          <= (others => '0');
                            hd_gemm_active  <= '0';
                            hd_gemm_a_valid <= '0';
                            hd_gemm_b_valid <= '0';
                            hd_gemm_c_valid <= '0';
                            hd_cap_cnt      <= 0;
                            report "MHA: Head " & integer'image(head_idx) & " V-Projection finished" severity note;
                        end if;

                    ------------------------------------------------------------
                    -- ST_COMPUTE_SCORES  -- S = (Q * K^T) / sqrt(HEAD_DIM)
                    --
                    -- Triple loop: i -> j -> d
                    -- S_buf[i][j] holds the running accumulator across cs_d.
                    ------------------------------------------------------------
                    when ST_COMPUTE_SCORES =>
                        v_prod := signed(Q_buf(cs_i, cs_d)) * signed(K_buf(cs_j, cs_d));

                        if cs_d = 0 then
                            v_acc := resize(v_prod, 64);
                        else
                            v_acc := cs_acc + resize(v_prod, 64);
                        end if;

                        if cs_d = HEAD_DIM - 1 then
                            -- Product is Q2.30.  Sum is still Q2.30.
                            -- Convert back to Q1.15 and apply sqrt(head_dim).
                            S_buf(cs_i, cs_j) <= std_logic_vector(
                                sat16(shift_right(v_acc, DATA_WIDTH - 1 + LOG_SQRT_HD))
                            );
                            cs_acc <= (others => '0');

                            cs_d <= 0;
                            if cs_j = SEQ_LEN - 1 then
                                cs_j <= 0;
                                if cs_i = SEQ_LEN - 1 then
                                    state        <= ST_APPLY_SOFTMAX;
                                    as_row       <= 0;
                                    sm_feeding   <= '0';
                                    sm_capturing <= '0';
                                    report "MHA: Head " & integer'image(head_idx) & " Scores computed" severity note;
                                else
                                    cs_i <= cs_i + 1;
                                end if;
                            else
                                cs_j <= cs_j + 1;
                            end if;
                        else
                            cs_acc <= v_acc;
                            cs_d <= cs_d + 1;
                        end if;

                    ------------------------------------------------------------
                    -- ST_APPLY_SOFTMAX  -- row-wise softmax on S_buf
                    --
                    -- For each row as_row:
                    --   1. Feed phase: stream S_buf[as_row][*] into softmax
                    --   2. Capture phase: stream softmax output back to S_buf[as_row][*]
                    ------------------------------------------------------------
                    when ST_APPLY_SOFTMAX =>
                        -- Feed phase
                        if sm_feeding = '0' and sm_capturing = '0' then
                            sm_feeding   <= '1';
                            sm_feed_cnt  <= 0;
                            sm_i_valid   <= '1';
                            sm_i_last    <= '0';
                            sm_i_data    <= S_buf(as_row, 0);
                            sm_i_channel <= 0;
                            report "MHA: Softmax feeding row " & integer'image(as_row) severity note;

                        elsif sm_feeding = '1' then
                            if sm_feed_cnt = SEQ_LEN - 2 then
                                -- Final element of this row
                                sm_i_data   <= S_buf(as_row, SEQ_LEN - 1);
                                sm_i_last   <= '1';
                                sm_i_valid  <= '1';
                                sm_feeding  <= '0';
                                sm_capturing <= '1';
                                sm_cap_cnt  <= 0;
                            else
                                sm_i_data   <= S_buf(as_row, sm_feed_cnt + 1);
                                sm_i_valid  <= '1';
                                sm_feed_cnt <= sm_feed_cnt + 1;
                            end if;
                        else
                            sm_i_valid <= '0';
                        end if;

                        -- Capture phase
                        if sm_capturing = '1' and sm_o_valid = '1' then
                            if sm_cap_cnt = 0 then
                                report "MHA: Softmax capturing row " & integer'image(as_row) severity note;
                            end if;
                            
                            if sm_cap_cnt < SEQ_LEN then
                                S_buf(as_row, sm_cap_cnt) <= sm_o_data;
                            end if;
                            
                            if sm_o_last = '1' then
                                sm_capturing <= '0';
                                if as_row = SEQ_LEN - 1 then
                                    state <= ST_ATTEND_V;
                                    av_i  <= 0;
                                    av_d  <= 0;
                                    av_j  <= 0;
                                    av_acc <= (others => '0');
                                    report "MHA: Head " & integer'image(head_idx) & " Softmax finished" severity note;
                                else
                                    as_row <= as_row + 1;
                                end if;
                            else
                                if sm_cap_cnt < SEQ_LEN then
                                    sm_cap_cnt <= sm_cap_cnt + 1;
                                end if;
                            end if;
                        end if;

                    ------------------------------------------------------------
                    -- ST_ATTEND_V  -- result = softmax(S) * V
                    --
                    -- concat_buf[i][head_idx * HEAD_DIM + d]
                    --   = sum_j S[i][j] * V[j][d]
                    --
                    -- Triple loop: i -> d -> j
                    ------------------------------------------------------------
                    when ST_ATTEND_V =>
                        v_prod := signed(S_buf(av_i, av_j)) * signed(V_buf(av_j, av_d));

                        if av_j = 0 then
                            v_acc := resize(v_prod, 64);
                        else
                            v_acc := av_acc + resize(v_prod, 64);
                        end if;

                        if av_j = SEQ_LEN - 1 then
                            concat_buf(av_i, head_idx * HEAD_DIM + av_d)
                                <= std_logic_vector(sat16(shift_right(v_acc, DATA_WIDTH - 1)));
                            av_acc <= (others => '0');

                            av_j <= 0;
                            if av_d = HEAD_DIM - 1 then
                                av_d <= 0;
                                if av_i = SEQ_LEN - 1 then
                                    if head_idx = NUM_HEADS - 1 then
                                        state <= ST_OUTPUT_PROJ;
                                        out_gemm_active <= '0';
                                        report "MHA: All heads finished, starting Output Projection" severity note;
                                    else
                                        head_idx <= head_idx + 1;
                                        state    <= ST_PROJ_Q;
                                        hd_gemm_active <= '0';
                                        report "MHA: Head " & integer'image(head_idx) & " finished, starting Head " & integer'image(head_idx + 1) severity note;
                                    end if;
                                else
                                    av_i <= av_i + 1;
                                end if;
                            else
                                av_d <= av_d + 1;
                            end if;
                        else
                            av_acc <= v_acc;
                            av_j <= av_j + 1;
                        end if;

                    ------------------------------------------------------------
                    -- ST_OUTPUT_PROJ  -- final = concat * W_O
                    ------------------------------------------------------------
                    when ST_OUTPUT_PROJ =>
                        out_gemm_a_valid <= '1';
                        out_gemm_b_valid <= '1';
                        out_gemm_c_valid <= '1';
                        out_gemm_c_data  <= (others => '0');

                        if out_gemm_active = '0' and out_gemm_done = '0' then
                            out_gemm_start  <= '1';
                            out_gemm_active <= '1';
                        end if;

                        -- Forward gemm output directly
                        if out_gemm_o_valid = '1' then
                            o_data    <= out_gemm_o_data;
                            o_valid   <= '1';
                            o_last    <= out_gemm_o_last;
                            o_channel <= out_gemm_o_chan;
                        end if;

                        if out_gemm_done = '1' then
                            state            <= ST_DONE;
                            out_gemm_a_valid <= '0';
                            out_gemm_b_valid <= '0';
                            out_gemm_c_valid <= '0';
                            o_valid          <= '0';
                        end if;

                    ------------------------------------------------------------
                    -- ST_DONE  -- one-cycle done assertion, then IDLE
                    ------------------------------------------------------------
                    when ST_DONE =>
                        state <= ST_IDLE;

                    ------------------------------------------------------------
                    -- Safety
                    ------------------------------------------------------------
                    when others =>
                        state <= ST_IDLE;

                end case;
            end if;
        end if;
    end process p_fsm;

end architecture rtl;
