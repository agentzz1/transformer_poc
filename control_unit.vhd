-- ============================================================================
-- control_unit.vhd  --  Post-LN Transformer Encoder Block Control Unit (FSM)
-- ============================================================================
-- This module sequences:
--   Multi-Head Attention (MHA), first residual add, first LayerNorm,
--   Feed-Forward Network (FFN),  second residual add, second LayerNorm.
-- ============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

use work.utilities.all;
use work.clog2_pkg.all;

entity control_unit is
    generic (
        DATA_WIDTH : positive := 16;
        SEQ_LEN    : positive := 64
    );
    port (
        -- Clock and reset
        clk   : in std_logic;
        rstn : in std_logic;

        -- Upstream handshake
        i_valid : in  std_logic;                     -- Input data valid
        i_last  : in  std_logic;                     -- Input data last (per token)
        i_ready : out std_logic;                     -- Ready to accept input (backpressure to previous stage)

        -- Downstream handshake
        o_ready : in  std_logic;                     -- Downstream ready (backpressure from next stage)
        o_valid : out std_logic;                     -- Output data valid (encoder block output ready)

        -- MHA control
        mha_start : out std_logic;                   -- Pulse: start MHA computation
        mha_mode  : out std_logic_vector(1 downto 0); -- MHA operating mode (00 = self-attention)

        -- FFN control
        ffn_start : out std_logic;                   -- Pulse: start FFN computation

        -- Residual-add enables
        residual_add_1_en : out std_logic;           -- Enable first residual add (MHA output + input)
        residual_add_2_en : out std_logic;           -- Enable second residual add (FFN output + first residual)

        -- LayerNorm enables
        layernorm_1_en : out std_logic;              -- Enable first LayerNorm
        layernorm_2_en : out std_logic;              -- Enable second LayerNorm

        -- Status inputs from sub-blocks
        mha_done             : in std_logic;         -- MHA sublayer finished
        ffn_done             : in std_logic;         -- FFN sublayer finished
        residual_add_1_done  : in std_logic;         -- First residual add finished
        residual_add_2_done  : in std_logic;         -- Second residual add finished
        layernorm_1_done     : in std_logic;         -- First LayerNorm finished
        layernorm_2_done     : in std_logic;         -- Second LayerNorm finished

        -- Sequence position (exposed for debug / sub-block use)
        seq_pos : out unsigned(integer(ceil(log2(real(SEQ_LEN)))) downto 0)
    );
end entity control_unit;

architecture rtl of control_unit is

    -- =========================================================================
    -- FSM state type
    -- =========================================================================
    type fsm_state_t is (
        ST_IDLE,            -- Waiting for i_valid
        ST_LOAD_MHA,        -- Loading input tokens into MHA
        ST_WAIT_MHA_DONE,   -- Waiting for MHA completion handshake
        ST_ADD_RESIDUAL_1,  -- First residual add
        ST_LAYERNORM_1,     -- First LayerNorm
        ST_FFN,             -- FFN sublayer active
        ST_WAIT_FFN_DONE,   -- Waiting for FFN completion handshake
        ST_ADD_RESIDUAL_2,  -- Second residual add
        ST_LAYERNORM_2,     -- Second LayerNorm
        ST_DONE             -- Output valid, waiting for downstream consumption
    );

    signal state      : fsm_state_t;
    signal next_state : fsm_state_t;

    -- =========================================================================
    -- Registered control outputs (no latches)
    -- =========================================================================
    signal mha_start_r         : std_logic;
    signal mha_mode_r          : std_logic_vector(1 downto 0);
    signal ffn_start_r         : std_logic;
    signal residual_add_1_en_r : std_logic;
    signal residual_add_2_en_r : std_logic;
    signal layernorm_1_en_r    : std_logic;
    signal layernorm_2_en_r    : std_logic;
    signal o_valid_r           : std_logic;

    -- =========================================================================
    -- Cycle counter for sequence-position tracking
    -- =========================================================================
    constant CYCLE_CNT_WIDTH : positive := integer(ceil(log2(real(SEQ_LEN)))) + 1;
    signal cycle_cnt         : unsigned(CYCLE_CNT_WIDTH - 1 downto 0);
    signal token_cnt         : integer := 0;

    -- =========================================================================
    -- One-shot / edge detection for sub-block done signals
    -- We assert a start pulse for exactly one cycle, then wait for the done
    -- signal.  The done signal is sampled combinatorially in next_state logic
    -- but gated by a stable 'active' qualifier so stray pulses do not cause
    -- spurious transitions.
    -- =========================================================================
    signal mha_active        : std_logic;
    signal ffn_active        : std_logic;
    signal resadd1_active    : std_logic;
    signal resadd2_active    : std_logic;
    signal layernorm1_active : std_logic;
    signal layernorm2_active : std_logic;

begin

    -- =========================================================================
    -- Drive entity outputs from registered signals
    -- =========================================================================
    mha_start          <= mha_start_r;
    mha_mode           <= mha_mode_r;
    ffn_start          <= ffn_start_r;
    residual_add_1_en  <= residual_add_1_en_r;
    residual_add_2_en  <= residual_add_2_en_r;
    layernorm_1_en     <= layernorm_1_en_r;
    layernorm_2_en     <= layernorm_2_en_r;
    o_valid            <= o_valid_r;
    i_ready            <= '1' when (state = ST_IDLE or state = ST_LOAD_MHA) else '0';

    -- =========================================================================
    -- Next-state logic (combinatorial — default assignments, no latches)
    -- =========================================================================
    p_fsm_next : process (all) is
    begin
        -- Default: stay in current state
        next_state <= state;

        case state is

            when ST_IDLE =>
                if i_valid = '1' then
                    next_state <= ST_LOAD_MHA;
                end if;
            
            when ST_LOAD_MHA =>
                if i_valid = '1' and i_last = '1' then
                    report "CU: Token end detected. token_cnt=" & integer'image(token_cnt) severity note;
                    if token_cnt = SEQ_LEN - 1 then
                        next_state <= ST_WAIT_MHA_DONE;
                        report "CU: Finished loading all tokens, moving to WAIT_MHA_DONE" severity note;
                    end if;
                end if;

            when ST_WAIT_MHA_DONE =>
                if mha_done = '1' then
                    next_state <= ST_LAYERNORM_1;
                end if;

            when ST_ADD_RESIDUAL_1 =>
                next_state <= ST_LAYERNORM_1;

            when ST_LAYERNORM_1 =>
                if layernorm1_active = '1' and layernorm_1_done = '1' then
                    next_state <= ST_FFN;
                end if;

            when ST_FFN =>
                next_state <= ST_WAIT_FFN_DONE;

            when ST_WAIT_FFN_DONE =>
                if ffn_done = '1' then
                    next_state <= ST_LAYERNORM_2;
                end if;

            when ST_ADD_RESIDUAL_2 =>
                next_state <= ST_LAYERNORM_2;

            when ST_LAYERNORM_2 =>
                if layernorm2_active = '1' and layernorm_2_done = '1' then
                    next_state <= ST_DONE;
                end if;

            when ST_DONE =>
                if o_ready = '1' then
                    next_state <= ST_IDLE;
                end if;

            when others =>
                next_state <= ST_IDLE;

        end case;
    end process p_fsm_next;

    -- =========================================================================
    -- Output-decoding logic (combinatorial — default assignments, no latches)
    -- =========================================================================
    p_output_decode : process (all) is
    begin
        -- Safe defaults for all outputs
        mha_start_r         <= '0';
        mha_mode_r          <= "00";
        ffn_start_r         <= '0';
        residual_add_1_en_r <= '0';
        residual_add_2_en_r <= '0';
        layernorm_1_en_r    <= '0';
        layernorm_2_en_r    <= '0';
        o_valid_r           <= '0';

        case state is
            when ST_LOAD_MHA =>
                -- Pulse mha_start only on the first cycle of ST_LOAD_MHA
                if token_cnt = 0 and cycle_cnt = 0 then
                    mha_start_r <= '1';
                end if;
                mha_mode_r  <= "00";

            when ST_FFN =>
                ffn_start_r <= '1';

            when ST_ADD_RESIDUAL_1 =>
                residual_add_1_en_r <= '1';

            when ST_ADD_RESIDUAL_2 =>
                residual_add_2_en_r <= '1';

            when ST_LAYERNORM_1 =>
                layernorm_1_en_r    <= '1';
                residual_add_1_en_r <= '1';

            when ST_LAYERNORM_2 =>
                layernorm_2_en_r    <= '1';
                residual_add_2_en_r <= '1';

            when ST_DONE =>
                o_valid_r <= '1';

            when others =>
                null;   -- keep defaults
        end case;
    end process p_output_decode;

    -- =========================================================================
    -- State register with synchronous reset
    -- =========================================================================
    p_state_reg : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                state <= ST_IDLE;
            else
                if state /= next_state then
                    report "CU: State transition: " & fsm_state_t'image(state) & 
                           " -> " & fsm_state_t'image(next_state) severity note;
                end if;
                state <= next_state;
            end if;
        end if;
    end process p_state_reg;

    -- =========================================================================
    -- Sub-block active qualifiers (registered one cycle after entering the
    -- state — prevents spurious done signals from propagating before the
    -- sub-block has truly started).
    -- =========================================================================
    p_active_flags : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                mha_active        <= '0';
                ffn_active        <= '0';
                resadd1_active    <= '0';
                resadd2_active    <= '0';
                layernorm1_active <= '0';
                layernorm2_active <= '0';
            else
                -- Assert on entry, de-assert on exit
                if state = ST_LOAD_MHA then
                    mha_active <= '1';
                elsif state = ST_ADD_RESIDUAL_1 then
                    mha_active <= '0';
                end if;

                if state = ST_FFN then
                    ffn_active <= '1';
                elsif state = ST_ADD_RESIDUAL_2 then
                    ffn_active <= '0';
                end if;

                if state = ST_ADD_RESIDUAL_1 then
                    resadd1_active <= '1';
                elsif state = ST_LAYERNORM_1 then
                    resadd1_active <= '0';
                end if;

                if state = ST_ADD_RESIDUAL_2 then
                    resadd2_active <= '1';
                elsif state = ST_LAYERNORM_2 then
                    resadd2_active <= '0';
                end if;

                if state = ST_LAYERNORM_1 then
                    layernorm1_active <= '1';
                elsif state = ST_FFN then
                    layernorm1_active <= '0';
                end if;

                if state = ST_LAYERNORM_2 then
                    layernorm2_active <= '1';
                elsif state = ST_DONE then
                    layernorm2_active <= '0';
                end if;
            end if;
        end if;
    end process p_active_flags;

    -- =========================================================================
    -- Cycle counter — tracks position within the sequence.
    -- Increments while the FSM is not idle; resets at the start of each new
    -- encoder-block pass.
    -- =========================================================================
    p_counters : process (clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                cycle_cnt <= (others => '0');
                token_cnt <= 0;
            else
                -- Token counter for ST_LOADING
                if state = ST_IDLE then
                    token_cnt <= 0;
                elsif state = ST_LOAD_MHA then
                    if i_valid = '1' and i_last = '1' then
                        if token_cnt < SEQ_LEN - 1 then
                            token_cnt <= token_cnt + 1;
                            report "CU: Buffered token " & integer'image(token_cnt + 1) severity note;
                        end if;
                    end if;
                end if;

                -- Cycle counter
                if state = ST_IDLE and i_valid = '1' then
                    cycle_cnt <= (others => '0');
                elsif state /= ST_IDLE and state /= ST_DONE then
                    cycle_cnt <= cycle_cnt + 1;
                end if;
            end if;
        end if;
    end process p_counters;

    seq_pos <= cycle_cnt;

end architecture rtl;

-- ============================================================================
-- End of file control_unit.vhd
-- ============================================================================
