-- =============================================================================
-- Layer Normalization (Post-LN) -- Synthesizable VHDL Module
-- Compatible with the transformer encoder project
-- =============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

library work;
use work.utilities.all;
use work.clog2_pkg.all;

entity layernorm is
    generic (
        DATA_WIDTH : positive := 16;
        VEC_SIZE   : positive := 512   -- must be power of 2
    );
    port (
        clk              : in  std_logic;
        rstn             : in  std_logic;

        -- Streaming input
        i_data           : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        i_valid          : in  std_logic;
        i_last           : in  std_logic;
        i_channel        : in  integer;

        -- Streaming output
        o_data           : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        o_valid          : out std_logic;
        o_last           : out std_logic;
        o_channel        : out integer;

        -- Parameter loading (gamma / beta)
        i_params_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        i_params_valid   : in  std_logic;
        i_params_addr    : in  std_logic_vector(clog2(VEC_SIZE) - 1 downto 0);
        i_params_sel     : in  std_logic                      -- '0' = gamma, '1' = beta
    );
end entity layernorm;

architecture rtl of layernorm is

    -- Constant for bit-width of internal indices
    constant VEC_BITS : positive := clog2(VEC_SIZE);

    -- FSM States
    type state_t is (IDLE, ACCUMULATE, COMPUTE, NORMALIZE, DONE);
    signal state, next_state : state_t;

    -- Local memories (registers)
    type data_array_t is array (0 to VEC_SIZE - 1) of signed(DATA_WIDTH - 1 downto 0);
    signal data_buf : data_array_t;
    signal gamma    : data_array_t := (others => to_signed(16384, DATA_WIDTH)); -- Q2.14 scale=1.0
    signal beta     : data_array_t := (others => to_signed(0, DATA_WIDTH));

    -- Accumulation registers
    signal sum_reg     : signed(31 downto 0);
    signal sum_sq_reg  : signed(47 downto 0);
    signal acc_cnt     : unsigned(VEC_BITS - 1 downto 0);
    signal acc_done    : std_logic;

    -- Statistics registers
    signal mean_reg    : signed(DATA_WIDTH - 1 downto 0);
    signal var_reg     : signed(31 downto 0);
    signal inv_std_reg : signed(DATA_WIDTH - 1 downto 0); -- Q2.14
    
    -- Normalization controls
    signal norm_cnt     : unsigned(VEC_BITS - 1 downto 0);
    signal norm_done    : std_logic;
    signal saved_chan   : integer := 0;

    -- Computation internal signals
    signal compute_done : std_logic;
    signal comp_step    : integer := 0;

begin

    ---------------------------------------------------------------------------
    -- FSM: State Transitions (Synchronous)
    ---------------------------------------------------------------------------
    p_fsm_seq : process(clk) is
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                state <= IDLE;
            else
                state <= next_state;
            end if;
        end if;
    end process p_fsm_seq;

    ---------------------------------------------------------------------------
    -- FSM: Next State Logic (Combinatorial)
    ---------------------------------------------------------------------------
    p_fsm_comb : process(state, i_valid, acc_done, compute_done, norm_done) is
    begin
        next_state <= state;
        case state is
            when IDLE =>
                if i_valid = '1' then
                    next_state <= ACCUMULATE;
                end if;
            
            when ACCUMULATE =>
                if acc_done = '1' then
                    next_state <= COMPUTE;
                end if;
            
            when COMPUTE =>
                if compute_done = '1' then
                    next_state <= NORMALIZE;
                end if;
            
            when NORMALIZE =>
                if i_valid = '1' then
                    report "LN: Starting normalization output for channel " & integer'image(i_channel);
                end if;
                if norm_done = '1' then
                    next_state <= DONE;
                end if;
            
            when DONE =>
                next_state <= IDLE;
            
            when others =>
                next_state <= IDLE;
        end case;
    end process p_fsm_comb;

    ---------------------------------------------------------------------------
    -- Data Path: Accumulation and Parameter Loading
    ---------------------------------------------------------------------------
    p_data_input : process(clk) is
        variable addr : integer;
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                acc_cnt    <= (others => '0');
                acc_done   <= '0';
                sum_reg    <= (others => '0');
                sum_sq_reg <= (others => '0');
                saved_chan <= 0;
            else
                -- Parameter Loading (Gamma/Beta)
                if i_params_valid = '1' then
                    addr := to_integer(unsigned(i_params_addr));
                    if i_params_sel = '0' then
                        gamma(addr) <= signed(i_params_data);
                    else
                        beta(addr)  <= signed(i_params_data);
                    end if;
                end if;

                -- Data Accumulation
                acc_done <= '0';
                if state = IDLE and i_valid = '1' then
                    acc_cnt    <= (others => '0');
                    sum_reg    <= resize(signed(i_data), 32);
                    sum_sq_reg <= resize(signed(i_data) * signed(i_data), 48);
                    data_buf(0) <= signed(i_data);
                    saved_chan <= i_channel;
                    acc_cnt    <= to_unsigned(1, VEC_BITS);
                elsif state = ACCUMULATE and i_valid = '1' then
                    data_buf(to_integer(acc_cnt)) <= signed(i_data);
                    sum_reg    <= sum_reg + resize(signed(i_data), 32);
                    sum_sq_reg <= sum_sq_reg + resize(signed(i_data) * signed(i_data), 48);
                    
                    if acc_cnt = to_unsigned(VEC_SIZE - 1, VEC_BITS) then
                        acc_done <= '1';
                    else
                        acc_cnt  <= acc_cnt + 1;
                    end if;
                end if;
            end if;
        end if;
    end process p_data_input;

    ---------------------------------------------------------------------------
    -- Data Path: Statistics Computation (Mean, Variance, InvStd)
    ---------------------------------------------------------------------------
    p_stats_compute : process(clk) is
        variable var_long : signed(63 downto 0);
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                mean_reg     <= (others => '0');
                var_reg      <= (others => '0');
                inv_std_reg  <= (others => '0');
                compute_done <= '0';
                comp_step    <= 0;
            else
                compute_done <= '0';
                if state = COMPUTE then
                    case comp_step is
                        when 0 =>
                            -- Mean = sum / VEC_SIZE (VEC_SIZE is power of 2)
                            mean_reg  <= resize(shift_right(sum_reg, VEC_BITS), DATA_WIDTH);
                            comp_step <= 1;
                        
                        when 1 =>
                            -- Var = (sum_sq / VEC_SIZE) - mean^2
                            var_long := resize(shift_right(sum_sq_reg, VEC_BITS), 64) - 
                                       resize(mean_reg * mean_reg, 64);
                            var_reg   <= resize(var_long, 32);
                            comp_step <= 2;
                        
                        when 2 =>
                            -- InvStd = 1 / sqrt(var + epsilon)
                            -- Simplified: Using a constant or simple approx for now
                            -- to focus on FSM flow. (Wait 5 cycles)
                            comp_step <= comp_step + 1;
                        
                        when 3 | 4 | 5 | 6 =>
                            comp_step <= comp_step + 1;
                            
                        when 7 =>
                            inv_std_reg  <= to_signed(163, DATA_WIDTH); -- Mock value
                            compute_done <= '1';
                            comp_step    <= 0;
                            
                        when others =>
                            comp_step <= 0;
                    end case;
                else
                    comp_step <= 0;
                end if;
            end if;
        end if;
    end process p_stats_compute;

    ---------------------------------------------------------------------------
    -- Data Path: Normalization Output
    ---------------------------------------------------------------------------
    p_normalize_out : process(clk) is
        variable x_val    : signed(DATA_WIDTH - 1 downto 0);
        variable diff     : signed(DATA_WIDTH downto 0);
        variable scaled   : signed(2 * DATA_WIDTH downto 0);
        variable weighted : signed(63 downto 0);
        variable y_val    : signed(DATA_WIDTH - 1 downto 0);
        variable idx      : integer;
    begin
        if rising_edge(clk) then
            if rstn = '0' then
                norm_cnt  <= (others => '0');
                norm_done <= '0';
                o_valid   <= '0';
                o_last    <= '0';
                o_data    <= (others => '0');
                o_channel <= 0;
            else
                o_valid <= '0';
                o_last  <= '0';
                norm_done <= '0';

                if state = NORMALIZE and norm_done = '0' then
                    idx   := to_integer(norm_cnt);
                    x_val := data_buf(idx);

                    -- y = (x - mean) * inv_std * gamma + beta
                    diff     := resize(x_val, DATA_WIDTH+1) - resize(mean_reg, DATA_WIDTH+1);
                    scaled   := diff * inv_std_reg;
                    weighted := resize(gamma(idx) * scaled, 64);
                    
                    y_val    := resize(shift_right(weighted, 14), DATA_WIDTH) + beta(idx);

                    o_data    <= std_logic_vector(y_val);
                    o_valid   <= '1';
                    o_channel <= saved_chan + idx;

                    if norm_cnt = to_unsigned(VEC_SIZE - 1, VEC_BITS) then
                        o_last    <= '1';
                        norm_done <= '1';
                        norm_cnt  <= (others => '0');
                    else
                        norm_cnt <= norm_cnt + 1;
                    end if;
                end if;
            end if;
        end if;
    end process p_normalize_out;

end architecture rtl;
