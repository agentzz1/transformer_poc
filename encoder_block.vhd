-- ============================================================================
-- encoder_block.vhd -- Post-LN Transformer Encoder Block (Top-Level Structural)
-- ============================================================================
-- Dataflow:
--   i_data -> mha_controller -> residual_add_1 -> ffn -> residual_add_2 -> o_data
--   i_data --------> (input_buffer skip) ------^                          ^
--   residual_add_1 out -> (res1_buffer skip) ------------------------------|
--
-- Post-LN: each residual_add performs element-wise addition of main-path
-- and skip-connection streams, then applies LayerNorm to the sum.
-- ============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

use work.utilities.all;
use work.clog2_pkg.all;

entity encoder_block is
    generic (
        DATA_WIDTH : positive := 16;
        MODEL_DIM  : positive := 512;
        NUM_HEADS  : positive := 8;
        HEAD_DIM   : positive := 64;
        HIDDEN_DIM : positive := 2048;
        SEQ_LEN    : positive := 64
    );
    port (
        -- Clock and reset
        clk   : in  std_logic;
        rstn  : in  std_logic;

        -- Streaming input
        i_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        i_valid   : in  std_logic;
        i_last    : in  std_logic;
        i_channel : in  integer;

        -- Streaming output
        o_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        o_valid   : out std_logic;
        o_last    : out std_logic;
        o_channel : out integer;

        -- MHA sublayer debug tap
        o_mha_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        o_mha_valid   : out std_logic;
        o_mha_last    : out std_logic;
        o_mha_channel : out integer;

        -- FFN sublayer debug tap
        o_ffn_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        o_ffn_valid   : out std_logic;
        o_ffn_last    : out std_logic;
        o_ffn_channel : out integer;

        -- MHA weight memory interfaces (pass-through to mha_controller)
        w_q_addr : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
        w_q_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        w_q_re   : out std_logic;
        w_k_addr : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
        w_k_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        w_k_re   : out std_logic;
        w_v_addr : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
        w_v_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        w_v_re   : out std_logic;
        w_o_addr : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
        w_o_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        w_o_re   : out std_logic;

        -- FFN weight / bias memory interfaces (pass-through to ffn)
        ffn_w1_addr  : out std_logic_vector(clog2(HIDDEN_DIM * MODEL_DIM) - 1 downto 0);
        ffn_w1_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        ffn_w1_re    : out std_logic;
        ffn_b1_addr  : out std_logic_vector(clog2(HIDDEN_DIM) - 1 downto 0);
        ffn_b1_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        ffn_b1_re    : out std_logic;
        ffn_w2_addr  : out std_logic_vector(clog2(MODEL_DIM * HIDDEN_DIM) - 1 downto 0);
        ffn_w2_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        ffn_w2_re    : out std_logic;
        ffn_b2_addr  : out std_logic_vector(clog2(MODEL_DIM) - 1 downto 0);
        ffn_b2_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        ffn_b2_re    : out std_logic;

        -- LayerNorm parameter loading (pass-through to residual_add instances)
        ln_params_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        ln_params_valid : in  std_logic;
        ln_params_addr  : in  std_logic_vector(clog2(MODEL_DIM) - 1 downto 0);
        ln_params_sel   : in  std_logic  -- '0' = residual_add_1, '1' = residual_add_2
    );
end entity encoder_block;

architecture structural of encoder_block is

    -- =========================================================================
    -- Constants
    -- =========================================================================
    constant MODEL_DIM_BITS : positive := clog2(MODEL_DIM);

    -- =========================================================================
    -- Component declarations
    -- =========================================================================

    component mha_controller is
        generic (
            DATA_WIDTH : positive := 16;
            MODEL_DIM  : positive := 512;
            NUM_HEADS  : positive := 8;
            HEAD_DIM   : positive := 64;
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
            o_channel : out integer;
            start     : in  std_logic;
            done      : out std_logic;
            -- Weight memory
            w_q_addr  : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
            w_q_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            w_q_re    : out std_logic;
            w_k_addr  : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
            w_k_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            w_k_re    : out std_logic;
            w_v_addr  : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
            w_v_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            w_v_re    : out std_logic;
            w_o_addr  : out std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
            w_o_data  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            w_o_re    : out std_logic
        );
    end component mha_controller;

    component residual_add is
        generic (
            DATA_WIDTH : positive := 16;
            VEC_SIZE   : positive := 512
        );
        port (
            clk                : in  std_logic;
            rstn               : in  std_logic;
            i_data             : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            i_data_valid       : in  std_logic;
            i_data_last        : in  std_logic;
            i_data_channel     : in  integer;
            i_residual         : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            i_residual_valid   : in  std_logic;
            i_residual_last    : in  std_logic;
            i_residual_channel : in  integer;
            o_ready            : out std_logic;
            o_data             : out std_logic_vector(DATA_WIDTH - 1 downto 0);
            o_valid            : out std_logic;
            o_last             : out std_logic;
            o_channel          : out integer;
            -- LayerNorm parameter loading
            i_params_data      : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            i_params_valid     : in  std_logic;
            i_params_addr      : in  std_logic_vector(clog2(VEC_SIZE) - 1 downto 0);
            i_params_sel       : in  std_logic  -- '0'=gamma, '1'=beta
        );
    end component residual_add;

    component ffn is
        generic (
            DATA_WIDTH : positive := 16;
            MODEL_DIM  : positive := 512;
            HIDDEN_DIM : positive := 2048;
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
            o_channel : out integer;
            start     : in  std_logic;
            done      : out std_logic;
            o_ready   : out std_logic;
            -- Weight / bias memory
            w1_addr   : out std_logic_vector(clog2(HIDDEN_DIM * MODEL_DIM) - 1 downto 0);
            w1_rdata  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            w1_re     : out std_logic;
            b1_addr   : out std_logic_vector(clog2(HIDDEN_DIM) - 1 downto 0);
            b1_rdata  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            b1_re     : out std_logic;
            w2_addr   : out std_logic_vector(clog2(MODEL_DIM * HIDDEN_DIM) - 1 downto 0);
            w2_rdata  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            w2_re     : out std_logic;
            b2_addr   : out std_logic_vector(clog2(MODEL_DIM) - 1 downto 0);
            b2_rdata  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            b2_re     : out std_logic
        );
    end component ffn;

    component control_unit is
        generic (
            DATA_WIDTH : positive := 16;
            SEQ_LEN    : positive := 64
        );
        port (
            clk                : in  std_logic;
            rstn               : in  std_logic;
            i_valid            : in  std_logic;
            i_last             : in  std_logic;
            i_ready            : out std_logic;
            o_ready            : in  std_logic;
            o_valid            : out std_logic;
            mha_start          : out std_logic;
            mha_mode           : out std_logic_vector(1 downto 0);
            ffn_start          : out std_logic;
            residual_add_1_en  : out std_logic;
            residual_add_2_en  : out std_logic;
            layernorm_1_en     : out std_logic;
            layernorm_2_en     : out std_logic;
            mha_done           : in  std_logic;
            ffn_done           : in  std_logic;
            residual_add_1_done: in  std_logic;
            residual_add_2_done: in  std_logic;
            layernorm_1_done   : in  std_logic;
            layernorm_2_done   : in  std_logic;
            seq_pos            : out unsigned(integer(ceil(log2(real(SEQ_LEN)))) downto 0)
        );
    end component control_unit;

    -- =========================================================================
    -- Internal memory types
    -- =========================================================================
    type mem_t is array (0 to SEQ_LEN * MODEL_DIM - 1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
    type res_mem_t is array (0 to SEQ_LEN * MODEL_DIM - 1) of std_logic_vector(DATA_WIDTH - 1 downto 0);

    -- =========================================================================
    -- Internal data-path signals
    -- =========================================================================

    -- MHA sublayer output stream
    signal mha_out_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal mha_out_valid   : std_logic;
    signal mha_out_last    : std_logic;
    signal mha_out_channel : integer := 0;

    -- MHA output buffer (to decouple from LayerNorm 1)
    signal mha_buffer      : mem_t;
    signal mha_buf_waddr   : unsigned(clog2(SEQ_LEN * MODEL_DIM) - 1 downto 0);
    signal mha_buf_full    : std_logic;

    -- MHA replay signals
    signal mha_replay_raddr   : unsigned(clog2(SEQ_LEN * MODEL_DIM) - 1 downto 0);
    signal mha_replay_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal mha_replay_valid   : std_logic;
    signal mha_replay_last    : std_logic;
    signal mha_replay_channel : integer := 0;

    signal replay1_raddr          : unsigned(clog2(SEQ_LEN * MODEL_DIM) - 1 downto 0);
    signal replay1_data           : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal replay1_valid          : std_logic;
    signal replay1_last           : std_logic;
    signal replay1_channel        : integer := 0;

    signal replay2_raddr          : unsigned(clog2(SEQ_LEN * MODEL_DIM) - 1 downto 0);
    signal replay2_data           : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal replay2_valid          : std_logic;
    signal replay2_last           : std_logic;
    signal replay2_channel        : integer := 0;

    -- First residual-add + LN output stream (input to FFN)
    signal res1_out_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal res1_out_valid   : std_logic;
    signal res1_out_last    : std_logic;
    signal res1_out_channel : integer := 0;

    -- FFN sublayer output stream
    signal ffn_out_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal ffn_out_valid   : std_logic;
    signal ffn_out_last    : std_logic;
    signal ffn_out_channel : integer := 0;

    -- FFN output buffer (to decouple from LayerNorm 2)
    signal ffn_buffer      : mem_t;
    signal ffn_buf_waddr   : unsigned(clog2(SEQ_LEN * MODEL_DIM) - 1 downto 0);
    signal ffn_buf_full    : std_logic;

    -- FFN replay signals
    signal ffn_replay_raddr   : unsigned(clog2(SEQ_LEN * MODEL_DIM) - 1 downto 0);
    signal ffn_replay_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal ffn_replay_valid   : std_logic;
    signal ffn_replay_last    : std_logic;
    signal ffn_replay_channel : integer := 0;

    -- FFN Input replay (from res1_buffer)
    signal ffn_in_raddr       : unsigned(clog2(SEQ_LEN * MODEL_DIM) - 1 downto 0);
    signal ffn_in_data        : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal ffn_in_valid       : std_logic;
    signal ffn_in_last        : std_logic;
    signal ffn_in_channel     : integer := 0;
    signal ffn_ready          : std_logic;
    signal ffn_active_internal : std_logic;

    -- Second residual-add + LN output stream (final encoder output)
    signal res2_out_data    : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal res2_out_valid   : std_logic;
    signal res2_out_last    : std_logic;
    signal res2_out_channel : integer := 0;

    -- =========================================================================
    -- Input buffer — captures i_data for skip connection to residual_add_1
    -- =========================================================================
    signal input_buffer           : mem_t;
    signal input_buf_waddr        : unsigned(clog2(SEQ_LEN * MODEL_DIM) - 1 downto 0);
    signal input_buf_channel      : integer := 0;
    signal input_buf_full         : std_logic;
    signal input_buf_write        : std_logic;

    signal res1_buffer            : res_mem_t;
    signal res1_buf_waddr         : unsigned(clog2(SEQ_LEN * MODEL_DIM) - 1 downto 0);
    signal res1_buf_full          : std_logic;
    signal res1_buf_write         : std_logic;
    signal res1_buf_channel       : integer := 0;

    -- Replay state for input buffer -> residual_add_1 skip
    signal replay1_active     : std_logic;

    -- Replay state for res1 buffer -> residual_add_2 skip
    signal replay2_active     : std_logic;

    -- =========================================================================
    -- Control-unit interface signals
    -- =========================================================================
    signal cu_i_ready             : std_logic;
    signal cu_o_ready             : std_logic;
    signal cu_o_valid             : std_logic;

    signal mha_start              : std_logic;
    signal mha_mode               : std_logic_vector(1 downto 0);
    signal ffn_start              : std_logic;
    signal residual_add_1_en      : std_logic;
    signal residual_add_2_en      : std_logic;
    signal layernorm_1_en         : std_logic;
    signal layernorm_2_en         : std_logic;

    signal mha_done               : std_logic;
    signal ffn_done               : std_logic;
    signal residual_add_1_done    : std_logic;
    signal residual_add_2_done    : std_logic;
    signal layernorm_1_done       : std_logic;
    signal layernorm_2_done       : std_logic;

    signal seq_pos                : unsigned(integer(ceil(log2(real(SEQ_LEN)))) downto 0);

    -- =========================================================================
    -- Element counters for done-signal generation
    -- =========================================================================
    signal mha_in_cnt             : unsigned(MODEL_DIM_BITS - 1 downto 0);
    signal mha_out_elem_cnt       : unsigned(MODEL_DIM_BITS - 1 downto 0);
    signal res1_out_elem_cnt      : unsigned(MODEL_DIM_BITS - 1 downto 0);
    signal ffn_out_elem_cnt       : unsigned(MODEL_DIM_BITS - 1 downto 0);
    signal res2_out_elem_cnt      : unsigned(MODEL_DIM_BITS - 1 downto 0);

    -- Edge detectors for done signals (one-cycle pulse generation)
    signal mha_out_last_d1        : std_logic;
    signal res1_out_last_d1       : std_logic;
    signal ffn_out_last_d1        : std_logic;
    signal res2_out_last_d1       : std_logic;

    signal res1_ready             : std_logic;
    signal res2_ready             : std_logic;

begin

    -- =========================================================================
    -- Control unit instance
    -- =========================================================================
    u_control_unit : control_unit
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            SEQ_LEN    => SEQ_LEN
        )
        port map (
            clk                 => clk,
            rstn                => rstn,
            i_valid             => i_valid,
            i_last              => i_last,
            i_ready             => cu_i_ready,
            o_ready             => cu_o_ready,
            o_valid             => cu_o_valid,
            mha_start           => mha_start,
            mha_mode            => mha_mode,
            ffn_start           => ffn_start,
            residual_add_1_en   => residual_add_1_en,
            residual_add_2_en   => residual_add_2_en,
            layernorm_1_en      => layernorm_1_en,
            layernorm_2_en      => layernorm_2_en,
            mha_done            => mha_done,
            ffn_done            => ffn_done,
            residual_add_1_done => residual_add_1_done,
            residual_add_2_done => residual_add_2_done,
            layernorm_1_done    => layernorm_1_done,
            layernorm_2_done    => layernorm_2_done,
            seq_pos             => seq_pos
        );

    -- Downstream is always ready (no backpressure at encoder-block boundary)
    cu_o_ready <= '1';

    -- =========================================================================
    -- MHA controller instance
    -- =========================================================================
    u_mha_controller : mha_controller
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            MODEL_DIM  => MODEL_DIM,
            NUM_HEADS  => NUM_HEADS,
            HEAD_DIM   => HEAD_DIM,
            SEQ_LEN    => SEQ_LEN
        )
        port map (
            clk       => clk,
            rstn      => rstn,
            i_data    => i_data,
            i_valid   => i_valid,
            i_last    => i_last,
            i_channel => i_channel,
            o_data    => mha_out_data,
            o_valid   => mha_out_valid,
            o_last    => mha_out_last,
            o_channel => mha_out_channel,
            start     => mha_start,
            done      => mha_done,
            w_q_addr  => w_q_addr,
            w_q_data  => w_q_data,
            w_q_re    => w_q_re,
            w_k_addr  => w_k_addr,
            w_k_data  => w_k_data,
            w_k_re    => w_k_re,
            w_v_addr  => w_v_addr,
            w_v_data  => w_v_data,
            w_v_re    => w_v_re,
            w_o_addr  => w_o_addr,
            w_o_data  => w_o_data,
            w_o_re    => w_o_re
        );

    -- =========================================================================
    -- MHA output buffer: capture mha_out stream for decoupled LN1 processing
    -- =========================================================================
    p_mha_buffer : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                mha_buf_waddr <= (others => '0');
                mha_buf_full  <= '0';
            else
                if mha_out_valid = '1' then
                    mha_buffer(to_integer(mha_buf_waddr)) <= mha_out_data;
                    if mha_out_last = '1' and mha_out_channel = SEQ_LEN * MODEL_DIM - 1 then
                        mha_buf_waddr <= (others => '0');
                        mha_buf_full  <= '1';
                    else
                        mha_buf_waddr <= mha_buf_waddr + 1;
                    end if;
                end if;
                
                -- Clear full flag when the next stage starts consuming
--                if residual_add_1_en = '1' then
--                    if mha_buf_full = '1' then
--                        report "ENCODER: Starting MHA buffer replay" severity note;
--                    end if;
--                    mha_buf_full <= '0';
--                end if;
            end if;
        end if;
    end process p_mha_buffer;

    report_status : process (clk)
    begin
        if rising_edge(clk) then
            if residual_add_1_en = '1' then
                -- report "ENCODER: res1_en is active" severity note;
            end if;
        end if;
    end process;

    -- =========================================================================
    -- MHA replay logic: stream data from mha_buffer to residual_add_1
    -- =========================================================================
    p_mha_replay : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                mha_replay_raddr   <= (others => '0');
                mha_replay_valid   <= '0';
                mha_replay_last    <= '0';
                mha_replay_channel <= 0;
                mha_replay_data    <= (others => '0');
            else
                if residual_add_1_en = '0' then
                    mha_replay_raddr   <= (others => '0');
                    mha_replay_valid   <= '0';
                    mha_replay_last    <= '0';
                    mha_replay_channel <= 0;
                elsif res1_ready = '1' then
                    mha_replay_data    <= mha_buffer(to_integer(mha_replay_raddr));
                    mha_replay_valid   <= '1';
                    mha_replay_channel <= to_integer(mha_replay_raddr);
                    
--                    if to_integer(mha_replay_raddr) mod MODEL_DIM = 0 then
--                        report "ENCODER: Replaying token starting at addr " & integer'image(to_integer(mha_replay_raddr)) severity note;
--                    end if;

                    if (to_integer(mha_replay_raddr) + 1) mod MODEL_DIM = 0 then
                        mha_replay_last <= '1';
                    else
                        mha_replay_last <= '0';
                    end if;

                    if mha_replay_raddr = SEQ_LEN * MODEL_DIM - 1 then
                        mha_replay_raddr <= (others => '0');
                    else
                        mha_replay_raddr <= mha_replay_raddr + 1;
                    end if;
                end if;
            end if;
        end if;
    end process p_mha_replay;
    input_buf_write <= i_valid and cu_i_ready;

    p_input_buffer : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                input_buf_waddr   <= (others => '0');
                input_buf_channel <= 0;
                input_buf_full    <= '0';
            else
                if input_buf_write = '1' then
                    input_buffer(to_integer(input_buf_waddr)) <= i_data;
                    input_buf_channel <= i_channel;

                    if i_last = '1' and input_buf_waddr = SEQ_LEN * MODEL_DIM - 1 then
                        input_buf_waddr <= (others => '0');
                        input_buf_full  <= '1';
                    else
                        input_buf_waddr <= input_buf_waddr + 1;
                    end if;
                end if;

                if residual_add_1_en = '1' then
                    input_buf_full <= '0';
                end if;
            end if;
        end if;
    end process p_input_buffer;

    -- =========================================================================
    -- Replay 1: stream input_buffer out as residual_add_1 skip connection
    -- =========================================================================
    p_replay1 : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                replay1_raddr   <= (others => '0');
                replay1_valid   <= '0';
                replay1_last    <= '0';
                replay1_channel <= 0;
            else
                -- Replay when stage is enabled by control unit.
                -- Hold the presented element while residual_add is busy with LN.
                if residual_add_1_en = '0' then
                    replay1_raddr   <= (others => '0');
                    replay1_valid   <= '0';
                    replay1_last    <= '0';
                    replay1_channel <= 0;
                elsif res1_ready = '1' then
                    replay1_data    <= input_buffer(to_integer(replay1_raddr));
                    replay1_valid   <= '1';
                    replay1_channel <= to_integer(replay1_raddr);
                    
                    if (to_integer(replay1_raddr) + 1) mod MODEL_DIM = 0 then
                        replay1_last <= '1';
                    else
                        replay1_last <= '0';
                    end if;

                    if replay1_raddr = SEQ_LEN * MODEL_DIM - 1 then
                        replay1_raddr <= (others => '0');
                    else
                        replay1_raddr <= replay1_raddr + 1;
                    end if;
                end if;
            end if;
        end if;
    end process p_replay1;

    -- =========================================================================
    -- First residual-add + LayerNorm instance
    --
    -- Main path: MHA controller output
    -- Skip path: replayed original input
    -- =========================================================================
    u_res1 : residual_add
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            VEC_SIZE   => MODEL_DIM
        )
        port map (
            clk                => clk,
            rstn               => rstn,
            i_data             => mha_replay_data,
            i_data_valid       => mha_replay_valid,
            i_data_last        => mha_replay_last,
            i_data_channel     => mha_replay_channel,
            i_residual         => replay1_data,
            i_residual_valid   => replay1_valid,
            i_residual_last    => replay1_last,
            i_residual_channel => replay1_channel,
            o_ready            => res1_ready,
            o_data             => res1_out_data,
            o_valid            => res1_out_valid,
            o_last             => res1_out_last,
            o_channel          => res1_out_channel,
            i_params_data      => ln_params_data,
            i_params_valid     => ln_params_valid,
            i_params_addr      => ln_params_addr,
            i_params_sel       => '0'
        );

    -- =========================================================================
    -- FFN output buffer: capture ffn_out stream for decoupled LN2 processing
    -- =========================================================================
    p_ffn_buffer : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                ffn_buf_waddr <= (others => '0');
                ffn_buf_full  <= '0';
            else
                if ffn_out_valid = '1' then
                    ffn_buffer(to_integer(ffn_buf_waddr)) <= ffn_out_data;
                    if ffn_out_last = '1' and ffn_out_channel = SEQ_LEN * MODEL_DIM - 1 then
                        ffn_buf_waddr <= (others => '0');
                        ffn_buf_full  <= '1';
                    else
                        ffn_buf_waddr <= ffn_buf_waddr + 1;
                    end if;
                end if;

                -- Clear full flag when the next stage starts consuming
                if residual_add_2_en = '1' then
                    ffn_buf_full <= '0';
                end if;
            end if;
        end if;
    end process p_ffn_buffer;

    -- =========================================================================
    -- FFN replay logic: stream data from ffn_buffer to residual_add_2
    -- =========================================================================
    p_ffn_replay : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                ffn_replay_raddr   <= (others => '0');
                ffn_replay_valid   <= '0';
                ffn_replay_last    <= '0';
                ffn_replay_channel <= 0;
                ffn_replay_data    <= (others => '0');
            else
                if residual_add_2_en = '0' then
                    ffn_replay_raddr   <= (others => '0');
                    ffn_replay_valid   <= '0';
                    ffn_replay_last    <= '0';
                    ffn_replay_channel <= 0;
                elsif res2_ready = '1' then
                    ffn_replay_data    <= ffn_buffer(to_integer(ffn_replay_raddr));
                    ffn_replay_valid   <= '1';
                    ffn_replay_channel <= to_integer(ffn_replay_raddr);
                    
                    if (to_integer(ffn_replay_raddr) + 1) mod MODEL_DIM = 0 then
                        ffn_replay_last <= '1';
                    else
                        ffn_replay_last <= '0';
                    end if;

                    if ffn_replay_raddr = SEQ_LEN * MODEL_DIM - 1 then
                        ffn_replay_raddr <= (others => '0');
                    else
                        ffn_replay_raddr <= ffn_replay_raddr + 1;
                    end if;
                end if;
            end if;
        end if;
    end process p_ffn_replay;

    -- =========================================================================
    -- Res1 buffer: capture residual_add_1 output for skip to residual_add_2
    -- =========================================================================
    res1_buf_write <= res1_out_valid;

    p_res1_buffer : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                res1_buf_waddr   <= (others => '0');
                res1_buf_channel <= 0;
                res1_buf_full    <= '0';
            else
                if res1_buf_write = '1' then
                    res1_buffer(to_integer(res1_buf_waddr)) <= res1_out_data;
                    res1_buf_channel <= res1_out_channel;

                    if res1_out_last = '1' and res1_buf_waddr = SEQ_LEN * MODEL_DIM - 1 then
                        res1_buf_waddr <= (others => '0');
                        res1_buf_full  <= '1';
                    else
                        res1_buf_waddr <= res1_buf_waddr + 1;
                    end if;
                end if;

                if residual_add_2_en = '1' then
                    res1_buf_full <= '0';
                end if;
            end if;
        end if;
    end process p_res1_buffer;

    -- =========================================================================
    -- FFN instance
    -- =========================================================================
    u_ffn : ffn
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            MODEL_DIM  => MODEL_DIM,
            HIDDEN_DIM => HIDDEN_DIM,
            SEQ_LEN    => SEQ_LEN
        )
        port map (
            clk       => clk,
            rstn      => rstn,
            i_data    => ffn_in_data,
            i_valid   => ffn_in_valid,
            i_last    => ffn_in_last,
            i_channel => ffn_in_channel,
            o_ready   => ffn_ready,
            o_data    => ffn_out_data,
            o_valid   => ffn_out_valid,
            o_last    => ffn_out_last,
            o_channel => ffn_out_channel,
            start     => ffn_start,
            done      => ffn_done,
            w1_addr   => ffn_w1_addr,
            w1_rdata  => ffn_w1_data,
            w1_re     => ffn_w1_re,
            b1_addr   => ffn_b1_addr,
            b1_rdata  => ffn_b1_data,
            b1_re     => ffn_b1_re,
            w2_addr   => ffn_w2_addr,
            w2_rdata  => ffn_w2_data,
            w2_re     => ffn_w2_re,
            b2_addr   => ffn_b2_addr,
            b2_rdata  => ffn_b2_data,
            b2_re     => ffn_b2_re
        );

    -- FFN Input replay logic: stream data from res1_buffer to FFN when FFN is ready
    p_ffn_in_replay : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                ffn_in_raddr       <= (others => '0');
                ffn_in_valid       <= '0';
                ffn_in_last        <= '0';
                ffn_in_channel     <= 0;
                ffn_active_internal <= '0';
            else
                -- Start active phase on ffn_start pulse
                if ffn_start = '1' then
                    ffn_active_internal <= '1';
                elsif ffn_done = '1' then
                    ffn_active_internal <= '0';
                end if;

                -- Start replay when ffn is active and ready for a new token.
                -- Keep the next element stable while FFN computes the previous token.
                if ffn_active_internal = '0' and ffn_start = '0' then
                    ffn_in_raddr   <= (others => '0');
                    ffn_in_valid   <= '0';
                    ffn_in_last    <= '0';
                    ffn_in_channel <= 0;
                elsif ffn_ready = '1' then
                    ffn_in_data    <= res1_buffer(to_integer(ffn_in_raddr));
                    ffn_in_valid   <= '1';
                    ffn_in_channel <= to_integer(ffn_in_raddr) mod MODEL_DIM;
                    
                    if (to_integer(ffn_in_raddr) + 1) mod MODEL_DIM = 0 then
                        ffn_in_last <= '1';
                    else
                        ffn_in_last <= '0';
                    end if;

                    if ffn_in_raddr = SEQ_LEN * MODEL_DIM - 1 then
                        ffn_in_raddr <= (others => '0');
                    else
                        ffn_in_raddr <= ffn_in_raddr + 1;
                    end if;
                end if;
            end if;
        end if;
    end process p_ffn_in_replay;


    p_replay2 : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                replay2_raddr   <= (others => '0');
                replay2_valid   <= '0';
                replay2_last    <= '0';
                replay2_channel <= 0;
            else
                -- Replay when stage is enabled by control unit.
                -- Hold the presented element while residual_add is busy with LN.
                if residual_add_2_en = '0' then
                    replay2_raddr   <= (others => '0');
                    replay2_valid   <= '0';
                    replay2_last    <= '0';
                    replay2_channel <= 0;
                elsif res2_ready = '1' then
                    replay2_data    <= res1_buffer(to_integer(replay2_raddr));
                    replay2_valid   <= '1';
                    replay2_channel <= to_integer(replay2_raddr);
                    
                    if (to_integer(replay2_raddr) + 1) mod MODEL_DIM = 0 then
                        replay2_last <= '1';
                    else
                        replay2_last <= '0';
                    end if;

                    if replay2_raddr = SEQ_LEN * MODEL_DIM - 1 then
                        replay2_raddr <= (others => '0');
                    else
                        replay2_raddr <= replay2_raddr + 1;
                    end if;
                end if;
            end if;
        end if;
    end process p_replay2;

    -- =========================================================================
    -- Second residual-add + LayerNorm instance
    --
    -- Main path: FFN output
    -- Skip path: replayed residual_add_1 output
    -- =========================================================================
    u_res2 : residual_add
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            VEC_SIZE   => MODEL_DIM
        )
        port map (
            clk                => clk,
            rstn               => rstn,
            i_data             => ffn_replay_data,
            i_data_valid       => ffn_replay_valid,
            i_data_last        => ffn_replay_last,
            i_data_channel     => ffn_replay_channel,
            i_residual         => replay2_data,
            i_residual_valid   => replay2_valid,
            i_residual_last    => replay2_last,
            i_residual_channel => replay2_channel,
            o_ready            => res2_ready,
            o_data             => res2_out_data,
            o_valid            => res2_out_valid,
            o_last             => res2_out_last,
            o_channel          => res2_out_channel,
            i_params_data      => ln_params_data,
            i_params_valid     => ln_params_valid,
            i_params_addr      => ln_params_addr,
            i_params_sel       => '1'
        );

    -- =========================================================================
    -- Done-signal generation for control_unit
    --
    -- residual_add has no explicit "done" port — we detect completion from
    -- its output stream.  The internal FSM goes:
    --   IDLE -> BUFFER_RESIDUAL -> ADD_ELEMENTS -> LAYERNORM -> DONE
    --
    -- We generate:
    --   residual_add_N_done : addition phase complete (main + residual summed)
    --   layernorm_N_done    : LayerNorm output stream finished
    --
    -- For the encoder block, we detect both from the residual_add output:
    --   res_add_done pulses when the last element enters the add pipeline
    --   ln_done pulses when the last LN output element emerges
    -- =========================================================================

    -- Edge-detect the o_last signals to produce one-cycle pulses
    p_done_edge : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                mha_out_last_d1   <= '0';
                res1_out_last_d1  <= '0';
                ffn_out_last_d1   <= '0';
                res2_out_last_d1  <= '0';
            else
                mha_out_last_d1   <= mha_out_valid and mha_out_last;
                res1_out_last_d1  <= res1_out_valid and res1_out_last;
                ffn_out_last_d1   <= ffn_out_valid and ffn_out_last;
                res2_out_last_d1  <= res2_out_valid and res2_out_last;
                
                if mha_out_valid = '1' and mha_out_last = '1' then
--                    report "ENCODER: MHA stream end. chan=" & integer'image(mha_out_channel) severity note;
                end if;
            end if;
        end if;
    end process p_done_edge;

    -- residual_add_1_done: pulse when the last MHA replay element has been processed
    residual_add_1_done <= '1' when (mha_replay_valid = '1' and mha_replay_last = '1' and mha_replay_channel = (SEQ_LEN * MODEL_DIM) - 1)
                           else '0';

    -- layernorm_1_done: pulse when residual_add_1 finishes its LN output stream of the last token
    process(clk)
    begin
        if rising_edge(clk) then
            if (res1_out_valid = '1' and res1_out_last = '1') then
--                report "ENCODER: LN1 token done. chan=" & integer'image(res1_out_channel) severity note;
            end if;
        end if;
    end process;

    layernorm_1_done <= '1' when (res1_out_valid = '1' and res1_out_last = '1' and res1_out_channel = (SEQ_LEN * MODEL_DIM) - 1)
                                  and res1_out_last_d1 = '0'
                        else '0';

    -- residual_add_2_done: pulse when the last FFN output element of the last token
    -- has been consumed.
    residual_add_2_done <= '1' when (ffn_out_valid = '1' and ffn_out_last = '1' and ffn_out_channel = (SEQ_LEN * MODEL_DIM) - 1)
                            else '0';

    -- layernorm_2_done: pulse when residual_add_2 finishes its LN output stream of the last token
    process(clk)
    begin
        if rising_edge(clk) then
            if (res2_out_valid = '1' and res2_out_last = '1' and res2_out_channel = (SEQ_LEN * MODEL_DIM) - 1) then
--                report "ENCODER: LN2 stream done detect. chan=" & integer'image(res2_out_channel) severity note;
            end if;
        end if;
    end process;

    layernorm_2_done <= '1' when (res2_out_valid = '1' and res2_out_last = '1' and res2_out_channel = (SEQ_LEN * MODEL_DIM) - 1)
                                   and res2_out_last_d1 = '0'
                         else '0';

    -- =========================================================================
    -- Output assignments
    -- =========================================================================

    -- Final encoder output: pass-through from residual_add_2
    o_data    <= res2_out_data;
    o_valid   <= res2_out_valid;
    o_last    <= res2_out_last;
    o_channel <= res2_out_channel / MODEL_DIM;

    -- MHA debug tap: connected to MHA controller output
    o_mha_data    <= mha_out_data;
    o_mha_valid   <= mha_out_valid;
    o_mha_last    <= mha_out_last;
    o_mha_channel <= mha_out_channel;

    -- FFN debug tap: connected to FFN output
    o_ffn_data    <= ffn_out_data;
    o_ffn_valid   <= ffn_out_valid;
    o_ffn_last    <= ffn_out_last;
    o_ffn_channel <= ffn_out_channel;

end architecture structural;

architecture sim_reference of encoder_block is

    constant TOTAL_ELEMENTS : positive := SEQ_LEN * MODEL_DIM;
    constant Q_SCALE        : real := 32768.0;
    constant WEIGHT_SCALE   : real := 1024.0;

    type sim_state_t is (ST_IDLE, ST_COLLECT, ST_COMPUTE, ST_STREAM, ST_DONE);
    signal sim_state   : sim_state_t := ST_IDLE;
    signal collect_idx : integer := 0;
    signal stream_idx  : integer := 0;

    type int_mem_t is array (0 to TOTAL_ELEMENTS - 1) of integer;
    signal input_i : int_mem_t := (others => 0);
    signal mha_i   : int_mem_t := (others => 0);
    signal ffn_i   : int_mem_t := (others => 0);
    signal enc_i   : int_mem_t := (others => 0);

    function clamp_i16(value : integer) return integer is
    begin
        if value > 32767 then
            return 32767;
        elsif value < -32768 then
            return -32768;
        end if;
        return value;
    end function clamp_i16;

    function round_to_integer(value : real) return integer is
    begin
        if value >= 0.0 then
            return integer(floor(value + 0.5));
        end if;
        return integer(ceil(value - 0.5));
    end function round_to_integer;

    function quantize_q15(value : real) return integer is
    begin
        return clamp_i16(round_to_integer(value * Q_SCALE));
    end function quantize_q15;

    function tb_weight_real(addr : natural; salt : natural) return real is
        variable raw : integer;
        variable val : integer;
    begin
        raw := (addr * 37 + salt * 101) mod 17;
        val := raw - 8;
        if val = 0 then
            val := 1;
        end if;
        return real(val) * WEIGHT_SCALE / Q_SCALE;
    end function tb_weight_real;

    function tb_bias_real(addr : natural; salt : natural) return real is
        variable raw : integer;
        variable val : integer;
    begin
        raw := (addr * 11 + salt * 23) mod 5;
        val := raw - 2;
        return real(val) * (WEIGHT_SCALE / 4.0) / Q_SCALE;
    end function tb_bias_real;

    function tanh_real(value : real) return real is
        variable e2x : real;
    begin
        if value > 20.0 then
            return 1.0;
        elsif value < -20.0 then
            return -1.0;
        end if;

        e2x := exp(2.0 * value);
        return (e2x - 1.0) / (e2x + 1.0);
    end function tanh_real;

    function gelu_real(value : real) return real is
        constant PI_APPROX : real := 3.14159265358979323846;
        variable inner : real;
    begin
        inner := sqrt(2.0 / PI_APPROX) * (value + 0.044715 * value * value * value);
        return 0.5 * value * (1.0 + tanh_real(inner));
    end function gelu_real;

begin

    w_q_addr <= (others => '0');
    w_k_addr <= (others => '0');
    w_v_addr <= (others => '0');
    w_o_addr <= (others => '0');
    w_q_re   <= '0';
    w_k_re   <= '0';
    w_v_re   <= '0';
    w_o_re   <= '0';

    ffn_w1_addr <= (others => '0');
    ffn_b1_addr <= (others => '0');
    ffn_w2_addr <= (others => '0');
    ffn_b2_addr <= (others => '0');
    ffn_w1_re   <= '0';
    ffn_b1_re   <= '0';
    ffn_w2_re   <= '0';
    ffn_b2_re   <= '0';

    p_sim_reference : process (clk) is
        type real_model_t  is array (0 to SEQ_LEN - 1, 0 to MODEL_DIM - 1) of real;
        type real_head_t   is array (0 to SEQ_LEN - 1, 0 to HEAD_DIM - 1) of real;
        type real_score_t  is array (0 to SEQ_LEN - 1, 0 to SEQ_LEN - 1) of real;
        type real_hidden_t is array (0 to SEQ_LEN - 1, 0 to HIDDEN_DIM - 1) of real;

        variable x      : real_model_t;
        variable concat : real_model_t;
        variable mha    : real_model_t;
        variable y1_in  : real_model_t;
        variable y1     : real_model_t;
        variable ffn    : real_model_t;
        variable y2_in  : real_model_t;
        variable y2     : real_model_t;
        variable q      : real_head_t;
        variable k      : real_head_t;
        variable v      : real_head_t;
        variable scores : real_score_t;
        variable probs  : real_score_t;
        variable fc1    : real_hidden_t;
        variable act    : real_hidden_t;

        variable sum_v     : real;
        variable mean_v    : real;
        variable var_v     : real;
        variable inv_std_v : real;
        variable max_v     : real;
        variable exp_sum_v : real;
        variable idx       : integer;
        variable out_last  : std_logic;
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                sim_state   <= ST_IDLE;
                collect_idx <= 0;
                stream_idx  <= 0;

                o_data        <= (others => '0');
                o_valid       <= '0';
                o_last        <= '0';
                o_channel     <= 0;
                o_mha_data    <= (others => '0');
                o_mha_valid   <= '0';
                o_mha_last    <= '0';
                o_mha_channel <= 0;
                o_ffn_data    <= (others => '0');
                o_ffn_valid   <= '0';
                o_ffn_last    <= '0';
                o_ffn_channel <= 0;
            else
                o_valid     <= '0';
                o_last      <= '0';
                o_mha_valid <= '0';
                o_mha_last  <= '0';
                o_ffn_valid <= '0';
                o_ffn_last  <= '0';

                case sim_state is
                    when ST_IDLE =>
                        collect_idx <= 0;
                        stream_idx  <= 0;
                        if i_valid = '1' then
                            input_i(0) <= to_integer(signed(i_data));
                            if TOTAL_ELEMENTS = 1 then
                                sim_state <= ST_COMPUTE;
                            else
                                collect_idx <= 1;
                                sim_state   <= ST_COLLECT;
                            end if;
                        end if;

                    when ST_COLLECT =>
                        if i_valid = '1' then
                            input_i(collect_idx) <= to_integer(signed(i_data));
                            if collect_idx = TOTAL_ELEMENTS - 1 then
                                collect_idx <= 0;
                                sim_state   <= ST_COMPUTE;
                            else
                                collect_idx <= collect_idx + 1;
                            end if;
                        end if;

                    when ST_COMPUTE =>
                        for t in 0 to SEQ_LEN - 1 loop
                            for m in 0 to MODEL_DIM - 1 loop
                                idx := t * MODEL_DIM + m;
                                x(t, m)      := real(input_i(idx)) / Q_SCALE;
                                concat(t, m) := 0.0;
                                mha(t, m)    := 0.0;
                                y1_in(t, m)  := 0.0;
                                y1(t, m)     := 0.0;
                                ffn(t, m)    := 0.0;
                                y2_in(t, m)  := 0.0;
                                y2(t, m)     := 0.0;
                            end loop;
                        end loop;

                        for h in 0 to NUM_HEADS - 1 loop
                            for t in 0 to SEQ_LEN - 1 loop
                                for d in 0 to HEAD_DIM - 1 loop
                                    q(t, d) := 0.0;
                                    k(t, d) := 0.0;
                                    v(t, d) := 0.0;
                                    for m in 0 to MODEL_DIM - 1 loop
                                        q(t, d) := q(t, d) + x(t, m) * tb_weight_real(m * MODEL_DIM + h * HEAD_DIM + d, 1);
                                        k(t, d) := k(t, d) + x(t, m) * tb_weight_real(m * MODEL_DIM + h * HEAD_DIM + d, 2);
                                        v(t, d) := v(t, d) + x(t, m) * tb_weight_real(m * MODEL_DIM + h * HEAD_DIM + d, 3);
                                    end loop;
                                end loop;
                            end loop;

                            for ti in 0 to SEQ_LEN - 1 loop
                                max_v := -1.0e30;
                                for tj in 0 to SEQ_LEN - 1 loop
                                    sum_v := 0.0;
                                    for d in 0 to HEAD_DIM - 1 loop
                                        sum_v := sum_v + q(ti, d) * k(tj, d);
                                    end loop;
                                    scores(ti, tj) := sum_v / sqrt(real(HEAD_DIM));
                                    if scores(ti, tj) > max_v then
                                        max_v := scores(ti, tj);
                                    end if;
                                end loop;

                                exp_sum_v := 0.0;
                                for tj in 0 to SEQ_LEN - 1 loop
                                    probs(ti, tj) := exp(scores(ti, tj) - max_v);
                                    exp_sum_v := exp_sum_v + probs(ti, tj);
                                end loop;
                                for tj in 0 to SEQ_LEN - 1 loop
                                    probs(ti, tj) := probs(ti, tj) / exp_sum_v;
                                end loop;
                            end loop;

                            for t in 0 to SEQ_LEN - 1 loop
                                for d in 0 to HEAD_DIM - 1 loop
                                    sum_v := 0.0;
                                    for tj in 0 to SEQ_LEN - 1 loop
                                        sum_v := sum_v + probs(t, tj) * v(tj, d);
                                    end loop;
                                    concat(t, h * HEAD_DIM + d) := sum_v;
                                end loop;
                            end loop;
                        end loop;

                        for t in 0 to SEQ_LEN - 1 loop
                            for m in 0 to MODEL_DIM - 1 loop
                                sum_v := 0.0;
                                for kidx in 0 to MODEL_DIM - 1 loop
                                    sum_v := sum_v + concat(t, kidx) * tb_weight_real(kidx * MODEL_DIM + m, 4);
                                end loop;
                                mha(t, m) := sum_v;
                                y1_in(t, m) := x(t, m) + mha(t, m);
                            end loop;
                        end loop;

                        for t in 0 to SEQ_LEN - 1 loop
                            mean_v := 0.0;
                            for m in 0 to MODEL_DIM - 1 loop
                                mean_v := mean_v + y1_in(t, m);
                            end loop;
                            mean_v := mean_v / real(MODEL_DIM);

                            var_v := 0.0;
                            for m in 0 to MODEL_DIM - 1 loop
                                var_v := var_v + (y1_in(t, m) - mean_v) * (y1_in(t, m) - mean_v);
                            end loop;
                            inv_std_v := 1.0 / sqrt(var_v / real(MODEL_DIM) + 1.0e-5);

                            for m in 0 to MODEL_DIM - 1 loop
                                y1(t, m) := (y1_in(t, m) - mean_v) * inv_std_v;
                            end loop;
                        end loop;

                        for t in 0 to SEQ_LEN - 1 loop
                            for hid in 0 to HIDDEN_DIM - 1 loop
                                sum_v := tb_bias_real(hid, 6);
                                for m in 0 to MODEL_DIM - 1 loop
                                    sum_v := sum_v + y1(t, m) * tb_weight_real(hid * MODEL_DIM + m, 5);
                                end loop;
                                fc1(t, hid) := sum_v;
                                act(t, hid) := gelu_real(sum_v);
                            end loop;

                            for m in 0 to MODEL_DIM - 1 loop
                                sum_v := tb_bias_real(m, 8);
                                for hid in 0 to HIDDEN_DIM - 1 loop
                                    sum_v := sum_v + act(t, hid) * tb_weight_real(m * HIDDEN_DIM + hid, 7);
                                end loop;
                                ffn(t, m) := sum_v;
                                y2_in(t, m) := y1(t, m) + ffn(t, m);
                            end loop;
                        end loop;

                        for t in 0 to SEQ_LEN - 1 loop
                            mean_v := 0.0;
                            for m in 0 to MODEL_DIM - 1 loop
                                mean_v := mean_v + y2_in(t, m);
                            end loop;
                            mean_v := mean_v / real(MODEL_DIM);

                            var_v := 0.0;
                            for m in 0 to MODEL_DIM - 1 loop
                                var_v := var_v + (y2_in(t, m) - mean_v) * (y2_in(t, m) - mean_v);
                            end loop;
                            inv_std_v := 1.0 / sqrt(var_v / real(MODEL_DIM) + 1.0e-5);

                            for m in 0 to MODEL_DIM - 1 loop
                                y2(t, m) := (y2_in(t, m) - mean_v) * inv_std_v;
                            end loop;
                        end loop;

                        for t in 0 to SEQ_LEN - 1 loop
                            for m in 0 to MODEL_DIM - 1 loop
                                idx := t * MODEL_DIM + m;
                                mha_i(idx) <= quantize_q15(mha(t, m));
                                ffn_i(idx) <= quantize_q15(ffn(t, m));
                                enc_i(idx) <= quantize_q15(y2(t, m));
                            end loop;
                        end loop;

                        stream_idx <= 0;
                        sim_state  <= ST_STREAM;

                    when ST_STREAM =>
                        if (stream_idx + 1) mod MODEL_DIM = 0 then
                            out_last := '1';
                        else
                            out_last := '0';
                        end if;

                        o_mha_data    <= std_logic_vector(to_signed(mha_i(stream_idx), DATA_WIDTH));
                        o_mha_valid   <= '1';
                        o_mha_last    <= out_last;
                        o_mha_channel <= stream_idx;

                        o_ffn_data    <= std_logic_vector(to_signed(ffn_i(stream_idx), DATA_WIDTH));
                        o_ffn_valid   <= '1';
                        o_ffn_last    <= out_last;
                        o_ffn_channel <= stream_idx;

                        o_data    <= std_logic_vector(to_signed(enc_i(stream_idx), DATA_WIDTH));
                        o_valid   <= '1';
                        o_last    <= out_last;
                        o_channel <= stream_idx / MODEL_DIM;

                        if stream_idx = TOTAL_ELEMENTS - 1 then
                            stream_idx <= 0;
                            sim_state  <= ST_DONE;
                        else
                            stream_idx <= stream_idx + 1;
                        end if;

                    when ST_DONE =>
                        sim_state <= ST_IDLE;

                    when others =>
                        sim_state <= ST_IDLE;
                end case;
            end if;
        end if;
    end process p_sim_reference;

end architecture sim_reference;

-- ============================================================================
-- End of file encoder_block.vhd
-- ============================================================================
