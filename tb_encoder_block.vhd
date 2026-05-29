-- ============================================================================
-- tb_encoder_block.vhd  --  Testbench for Post-LN Transformer Encoder Block
-- ============================================================================
-- Tests the full encoder_block top-level entity: Multi-Head Attention (MHA),
-- first residual add + LayerNorm, Feed-Forward Network (FFN), second residual
-- add + LayerNorm.
--
-- Data:      SEQ_LEN = 64 tokens, MODEL_DIM = 512 elements per token
--            Token values generated via 16-bit Fibonacci LFSR (deterministic,
--            repeatable pseudo-random).  Channel carries the token index.
-- Clock:     10 ns period (5 ns half-cycle)
-- Reset:     rstn asserted after 30 ns (matching tb_sigmoid_requant.vhd)
-- File I/O:  std.textio — MHA / FFN / final encoder outputs logged to
--            "mha_out.txt", "ffn_out.txt", "encoder_out.txt".
-- Assertions: valid/last consistency, per-token element count, channel range.
-- Termination: std.env.finish after all outputs captured.
-- ============================================================================

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;
    use std.textio.all;
    use std.env.finish;
    use ieee.math_real.all;

    use work.utilities.all;
    use work.clog2_pkg.all;

entity tb_encoder_block is
end entity tb_encoder_block;

architecture sim of tb_encoder_block is

    ---------------------------------------------------------------------------
    -- Constants
    ---------------------------------------------------------------------------
    constant DATA_WIDTH : positive := 16;
    constant MODEL_DIM  : positive := 32;
    constant SEQ_LEN    : positive := 16;
    constant NUM_HEADS  : positive := 8;
    constant HEAD_DIM   : positive := 4;
    constant HIDDEN_DIM : positive := 128;
    constant WEIGHT_SCALE : integer := 1024;

    -- Total number of elements across all tokens
    constant TOTAL_ELEMENTS : positive := SEQ_LEN * MODEL_DIM;

    -- Output file names (relative paths for portability)
    constant MHA_OUT_FILE     : string := "mha_out.txt";
    constant FFN_OUT_FILE     : string := "ffn_out.txt";
    constant ENCODER_OUT_FILE : string := "encoder_out.txt";

    -- LFSR initial seed (deterministic, repeatable)
    constant LFSR_SEED : std_logic_vector(15 downto 0) := x"ACE1";

    constant max_size_x : positive := 512;

    function tb_weight (
        addr : natural;
        salt : natural
    ) return std_logic_vector is
        variable raw : integer;
        variable val : integer;
    begin
        raw := (addr * 37 + salt * 101) mod 17;
        val := raw - 8;
        if val = 0 then
            val := 1;
        end if;
        return std_logic_vector(to_signed(val * WEIGHT_SCALE, DATA_WIDTH));
    end function tb_weight;

    function tb_bias (
        addr : natural;
        salt : natural
    ) return std_logic_vector is
        variable raw : integer;
        variable val : integer;
    begin
        raw := (addr * 11 + salt * 23) mod 5;
        val := raw - 2;
        return std_logic_vector(to_signed(val * (WEIGHT_SCALE / 4), DATA_WIDTH));
    end function tb_bias;

    ---------------------------------------------------------------------------
    -- Component declaration for encoder_block
    ---------------------------------------------------------------------------
    component encoder_block is
        generic (
            DATA_WIDTH : positive := 16;
            MODEL_DIM  : positive := 512;
            NUM_HEADS  : positive := 8;
            HEAD_DIM   : positive := 64;
            HIDDEN_DIM : positive := 128;
            SEQ_LEN    : positive := 64
        );
        port (
            clk   : in std_logic;
            rstn : in std_logic;

            -- Input stream
            i_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            i_valid   : in  std_logic;
            i_last    : in  std_logic;
            i_channel : in  integer;

            -- MHA sub-block output (debug)
            o_mha_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
            o_mha_valid   : out std_logic;
            o_mha_last    : out std_logic;
            o_mha_channel : out integer;

            -- FFN sub-block output (debug)
            o_ffn_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
            o_ffn_valid   : out std_logic;
            o_ffn_last    : out std_logic;
            o_ffn_channel : out integer;

            -- Final encoder output stream
            o_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
            o_valid   : out std_logic;
            o_last    : out std_logic;
            o_channel : out integer;

            -- MHA weight memory (unused in testbench, tied off)
            w_q_addr : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
            w_q_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
            w_q_re   : out std_logic;
            w_k_addr : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
            w_k_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
            w_k_re   : out std_logic;
            w_v_addr : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
            w_v_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
            w_v_re   : out std_logic;
            w_o_addr : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
            w_o_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
            w_o_re   : out std_logic;

            -- FFN weight memory (unused in testbench, tied off)
            ffn_w1_addr  : out std_logic_vector(clog2(HIDDEN_DIM * MODEL_DIM) - 1 downto 0);
            ffn_w1_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
            ffn_w1_re    : out std_logic;
            ffn_b1_addr  : out std_logic_vector(clog2(HIDDEN_DIM) - 1 downto 0);
            ffn_b1_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
            ffn_b1_re    : out std_logic;
            ffn_w2_addr  : out std_logic_vector(clog2(MODEL_DIM * HIDDEN_DIM) - 1 downto 0);
            ffn_w2_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
            ffn_w2_re    : out std_logic;
            ffn_b2_addr  : out std_logic_vector(clog2(MODEL_DIM) - 1 downto 0);
            ffn_b2_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
            ffn_b2_re    : out std_logic;

            -- LayerNorm parameter loading (unused in testbench, tied off)
            ln_params_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
            ln_params_valid : in  std_logic := '0';
            ln_params_addr  : in  std_logic_vector(clog2(MODEL_DIM) - 1 downto 0) := (others => '0');
            ln_params_sel   : in  std_logic := '0'
        );
    end component encoder_block;

    ---------------------------------------------------------------------------
    -- Signals
    ---------------------------------------------------------------------------
    signal clk  : std_logic := '0';
    signal rstn : std_logic := '0';

    -- Input stream
    signal i_data    : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal i_valid   : std_logic := '0';
    signal i_last    : std_logic := '0';
    signal i_channel : integer := 0;

    -- MHA debug output
    signal o_mha_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal o_mha_valid   : std_logic;
    signal o_mha_last    : std_logic;
    signal o_mha_channel : integer := 0;

    -- FFN debug output
    signal o_ffn_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal o_ffn_valid   : std_logic;
    signal o_ffn_last    : std_logic;
    signal o_ffn_channel : integer := 0;

    -- Final encoder output
    signal o_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal o_valid   : std_logic;
    signal o_last    : std_logic;
    signal o_channel : integer := 0;

    -- MHA weight memory (tied off)
    signal w_q_addr : std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
    signal w_q_data : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal w_q_re   : std_logic;
    signal w_k_addr : std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
    signal w_k_data : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal w_k_re   : std_logic;
    signal w_v_addr : std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
    signal w_v_data : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal w_v_re   : std_logic;
    signal w_o_addr : std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
    signal w_o_data : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal w_o_re   : std_logic;

    -- FFN weight memory (tied off)
    signal ffn_w1_addr : std_logic_vector(clog2(HIDDEN_DIM * MODEL_DIM) - 1 downto 0);
    signal ffn_w1_data : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal ffn_w1_re   : std_logic;
    signal ffn_b1_addr : std_logic_vector(clog2(HIDDEN_DIM) - 1 downto 0);
    signal ffn_b1_data : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal ffn_b1_re   : std_logic;
    signal ffn_w2_addr : std_logic_vector(clog2(MODEL_DIM * HIDDEN_DIM) - 1 downto 0);
    signal ffn_w2_data : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal ffn_w2_re   : std_logic;
    signal ffn_b2_addr : std_logic_vector(clog2(MODEL_DIM) - 1 downto 0);
    signal ffn_b2_data : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal ffn_b2_re   : std_logic;

    -- LayerNorm params (tied off)
    signal ln_params_data  : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    signal ln_params_valid : std_logic := '0';
    signal ln_params_addr  : std_logic_vector(clog2(MODEL_DIM) - 1 downto 0) := (others => '0');
    signal ln_params_sel   : std_logic := '0';

    -- Completion flags — set when each capture process has written all
    -- expected outputs.  The assertion process watches encoder_capture_done
    -- to know when to terminate.
    signal encoder_capture_done : std_logic := '0';

begin

    ---------------------------------------------------------------------------
    -- Clock generation: 10 ns period, 5 ns half-cycle
    -- (matches existing project testbench style)
    ---------------------------------------------------------------------------
    clk <= not clk after 5 ns;

    ---------------------------------------------------------------------------
    -- DUT instantiation
    ---------------------------------------------------------------------------
    dut : entity work.encoder_block(structural)
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            MODEL_DIM  => MODEL_DIM,
            NUM_HEADS  => NUM_HEADS,
            HEAD_DIM   => HEAD_DIM,
            HIDDEN_DIM => HIDDEN_DIM,
            SEQ_LEN    => SEQ_LEN
        )
        port map (
            clk           => clk,
            rstn         => rstn,
            i_data        => i_data,
            i_valid       => i_valid,
            i_last        => i_last,
            i_channel     => i_channel,
            o_mha_data    => o_mha_data,
            o_mha_valid   => o_mha_valid,
            o_mha_last    => o_mha_last,
            o_mha_channel => o_mha_channel,
            o_ffn_data    => o_ffn_data,
            o_ffn_valid   => o_ffn_valid,
            o_ffn_last    => o_ffn_last,
            o_ffn_channel => o_ffn_channel,
            o_data        => o_data,
            o_valid       => o_valid,
            o_last        => o_last,
            o_channel     => o_channel,
            -- Deterministic test memories
            w_q_addr => w_q_addr, w_q_data => w_q_data, w_q_re => w_q_re,
            w_k_addr => w_k_addr, w_k_data => w_k_data, w_k_re => w_k_re,
            w_v_addr => w_v_addr, w_v_data => w_v_data, w_v_re => w_v_re,
            w_o_addr => w_o_addr, w_o_data => w_o_data, w_o_re => w_o_re,
            ffn_w1_addr => ffn_w1_addr, ffn_w1_data => ffn_w1_data, ffn_w1_re => ffn_w1_re,
            ffn_b1_addr => ffn_b1_addr, ffn_b1_data => ffn_b1_data, ffn_b1_re => ffn_b1_re,
            ffn_w2_addr => ffn_w2_addr, ffn_w2_data => ffn_w2_data, ffn_w2_re => ffn_w2_re,
            ffn_b2_addr => ffn_b2_addr, ffn_b2_data => ffn_b2_data, ffn_b2_re => ffn_b2_re,
            -- LayerNorm params tied off
            ln_params_data => (others => '0'), ln_params_valid => '0',
            ln_params_addr => (others => '0'), ln_params_sel => '0'
        );

    w_q_data <= tb_weight(to_integer(unsigned(w_q_addr)), 1) when w_q_re = '1' else (others => '0');
    w_k_data <= tb_weight(to_integer(unsigned(w_k_addr)), 2) when w_k_re = '1' else (others => '0');
    w_v_data <= tb_weight(to_integer(unsigned(w_v_addr)), 3) when w_v_re = '1' else (others => '0');
    w_o_data <= tb_weight(to_integer(unsigned(w_o_addr)), 4) when w_o_re = '1' else (others => '0');

    ffn_w1_data <= tb_weight(to_integer(unsigned(ffn_w1_addr)), 5) when ffn_w1_re = '1' else (others => '0');
    ffn_b1_data <= tb_bias(to_integer(unsigned(ffn_b1_addr)), 6) when ffn_b1_re = '1' else (others => '0');
    ffn_w2_data <= tb_weight(to_integer(unsigned(ffn_w2_addr)), 7) when ffn_w2_re = '1' else (others => '0');
    ffn_b2_data <= tb_bias(to_integer(unsigned(ffn_b2_addr)), 8) when ffn_b2_re = '1' else (others => '0');

    ---------------------------------------------------------------------------
    -- Reset process: assert rstn after 30 ns
    -- (matching tb_sigmoid_requant.vhd pattern)
    ---------------------------------------------------------------------------
    p_reset : process is
    begin
        wait for 30 ns;
        wait until rising_edge(clk);
        rstn <= '1';
        wait;
    end process p_reset;

    ---------------------------------------------------------------------------
    -- Stimulus process
    --
    -- Generates SEQ_LEN tokens, each containing MODEL_DIM elements.
    -- Element values are produced by a 16-bit Fibonacci LFSR seeded with
    -- LFSR_SEED, giving deterministic pseudo-random activations.
    --
    -- Handshaking conventions (accel-library streaming style):
    --   valid  — asserted with every data element
    --   last   — marks the final element of each token
    --   channel — carries the token index (0 .. SEQ_LEN-1)
    --
    -- The process waits 20 ns after reset release, then drives one element
    -- per clock cycle until the full sequence has been transmitted.
    ---------------------------------------------------------------------------
    p_stimulus : process is
        variable lfsr_v   : std_logic_vector(15 downto 0) := LFSR_SEED;
        variable feedback : std_logic;
        variable data_val : signed(DATA_WIDTH - 1 downto 0);
    begin
        -- Safe idle state before reset
        i_valid   <= '0';
        i_last    <= '0';
        i_data    <= (others => '0');
        i_channel <= 0;

        -- Wait for reset release (matching tb_sigmoid_requant delay)
        wait until rstn = '1';
        wait for 20 ns;
        wait until rising_edge(clk);

        -- Generate SEQ_LEN tokens
        for token_idx in 0 to SEQ_LEN - 1 loop
            -- Generate MODEL_DIM elements for this token
            for elem_idx in 0 to MODEL_DIM - 1 loop
                -- ----------------------------------------------------------
                -- Advance 16-bit Fibonacci LFSR
                -- Polynomial: x^16 + x^14 + x^13 + x^11 + 1
                -- Tap positions (0-indexed): 15, 13, 12, 10
                -- ----------------------------------------------------------
                feedback := lfsr_v(15) xor lfsr_v(13)
                            xor lfsr_v(12) xor lfsr_v(10);
                lfsr_v   := lfsr_v(14 downto 0) & feedback;

                -- Use raw LFSR value as signed data (covers full 16-bit range
                -- when interpreted as two's-complement)
                data_val := signed(lfsr_v);

                -- Avoid producing exactly zero for better visibility in logs;
                -- a zero LFSR state is a lock-up condition anyway and will
                -- never occur with a non-zero seed in a maximal-length LFSR.
                if data_val = 0 then
                    data_val := to_signed(1, DATA_WIDTH);
                end if;

                -- Drive the streaming interface
                i_data    <= std_logic_vector(data_val);
                i_channel <= token_idx;
                i_valid   <= '1';

                -- Assert last on the final element of each token
                if elem_idx = MODEL_DIM - 1 then
                    i_last <= '1';
                else
                    i_last <= '0';
                end if;

                    wait until rising_edge(clk);
                end loop;
                report "TB: Finished driving token " & integer'image(token_idx) severity note;
            end loop;
            -- Wait one more cycle for the last element to be sampled by the FSM
            wait until rising_edge(clk);
            i_valid <= '0';
            i_last  <= '0';
            report "TB: All tokens driven" severity note;

        -- Entire sequence transmitted; deassert handshake
        i_valid   <= '0';
        i_last    <= '0';
        i_data    <= (others => '0');
        i_channel <= 0;

        wait;
    end process p_stimulus;

    ---------------------------------------------------------------------------
    -- Assertion / protocol-checking process
    --
    -- Monitors the final encoder output stream (o_valid, o_last, o_channel)
    -- and flags violations:
    --   (1) o_last must never be asserted without o_valid.
    --   (2) Each token must produce exactly MODEL_DIM valid outputs between
    --       consecutive o_last assertions.
    --   (3) o_channel must remain constant across all elements of a token.
    --   (4) o_channel must lie in the valid range [0 .. SEQ_LEN-1].
    --
    -- The process runs until encoder_capture_done is set, then exits cleanly.
    ---------------------------------------------------------------------------
    p_assertions : process
        variable token_elem_cnt : integer := 0;
        variable capture_chan   : integer := -1;
        variable current_chan   : integer;
    begin
        -- Wait until the DUT is out of reset
        wait until rstn = '1';
        wait until rising_edge(clk);

        token_elem_cnt := 0;
        capture_chan   := -1;

        loop
            wait until rising_edge(clk);

            ------------------------------------------------------------------
            -- (1) Check: last must not be asserted without valid
            ------------------------------------------------------------------
            if o_last = '1' and o_valid = '0' then
                report "ERROR: o_last asserted without o_valid at "
                    & time'image(now) severity error;
            end if;

            ------------------------------------------------------------------
            -- (2,3,4) Token-level checks
            ------------------------------------------------------------------
            if o_valid = '1' then
                token_elem_cnt := token_elem_cnt + 1;
                current_chan   := o_channel;

                -- On the first element of a new token, capture the channel
                if token_elem_cnt = 1 then
                    capture_chan := current_chan;
                else
                    -- (3) Verify channel stays constant within a token
                    if current_chan /= capture_chan then
                        report "ERROR: o_channel changed mid-token: "
                            & integer'image(capture_chan) & " -> "
                            & integer'image(current_chan)
                            severity error;
                        capture_chan := current_chan;
                    end if;
                end if;

                -- (4) Channel must be in the valid range
                if capture_chan < 0 or capture_chan >= SEQ_LEN then
                    report "ERROR: o_channel out of range (0.."
                        & integer'image(SEQ_LEN - 1) & "): "
                        & integer'image(capture_chan)
                        severity error;
                end if;

                -- (2) When o_last fires, verify that the token had exactly
                --     MODEL_DIM elements.
                if o_last = '1' then
                    if token_elem_cnt /= MODEL_DIM then
                        report "ERROR: Token element count mismatch. "
                            & "Expected " & integer'image(MODEL_DIM)
                            & ", got " & integer'image(token_elem_cnt)
                            & " (channel " & integer'image(capture_chan) & ")"
                            severity error;
                    end if;
                    -- Reset counter for the next token
                    token_elem_cnt := 0;
                end if;
            end if;

            -- Stop the loop once all outputs have been captured
            if encoder_capture_done = '1' then
                exit;
            end if;
        end loop;

        report "Assertion checks completed." severity note;
        wait;
    end process p_assertions;

    p_capture_mha : process is
        file out_file : text open write_mode is MHA_OUT_FILE;
        variable l    : line;
    begin
        wait until rstn = '1';
        loop
            wait until rising_edge(clk);
            if o_mha_valid = '1' then
                write(l, o_mha_channel);
                write(l, string'(" "));
                write(l, to_integer(signed(o_mha_data)));
                write(l, string'(" "));
                if o_mha_last = '1' then
                    write(l, 1);
                else
                    write(l, 0);
                end if;
                writeline(out_file, l);
            end if;
        end loop;
    end process p_capture_mha;

    p_capture_ffn : process is
        file out_file : text open write_mode is FFN_OUT_FILE;
        variable l    : line;
    begin
        wait until rstn = '1';
        loop
            wait until rising_edge(clk);
            if o_ffn_valid = '1' then
                write(l, o_ffn_channel);
                write(l, string'(" "));
                write(l, to_integer(signed(o_ffn_data)));
                write(l, string'(" "));
                if o_ffn_last = '1' then
                    write(l, 1);
                else
                    write(l, 0);
                end if;
                writeline(out_file, l);
            end if;
        end loop;
    end process p_capture_ffn;

    p_capture_encoder : process is
        file out_file : text open write_mode is ENCODER_OUT_FILE;
        variable l     : line;
        variable count : integer := 0;
    begin
        wait until rstn = '1';
        loop
            wait until rising_edge(clk);
            if o_valid = '1' then
                write(l, o_channel);
                write(l, string'(" "));
                write(l, to_integer(signed(o_data)));
                write(l, string'(" "));
                if o_last = '1' then
                    write(l, 1);
                else
                    write(l, 0);
                end if;
                writeline(out_file, l);

                count := count + 1;
                if count = TOTAL_ELEMENTS then
                    encoder_capture_done <= '1';
                    report "TB: Captured all encoder outputs" severity note;
                    wait until rising_edge(clk);
                    report "TB: Simulation completed" severity note;
                    finish;
                end if;
            end if;
        end loop;
    end process p_capture_encoder;

end architecture sim;

-- ============================================================================
-- End of file tb_encoder_block.vhd
-- ============================================================================
