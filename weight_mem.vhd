-- ============================================================================
-- weight_mem.vhd  --  Synchronous BRAM/ROM models for Transformer weights & biases
-- ============================================================================
-- Provides deterministic, initialised memory blocks for simulation and
-- synthesis (inferable as BRAM/ROM by most FPGA tools).
--
-- Memories:
--   MHA: W_Q, W_K, W_V, W_O  (MODEL_DIM x MODEL_DIM)
--   FFN: W_1 (HIDDEN_DIM x MODEL_DIM), b_1 (HIDDEN_DIM)
--        W_2 (MODEL_DIM x HIDDEN_DIM), b_2 (MODEL_DIM)
--
-- Read interface: registered address + re -> data on next cycle.
-- ============================================================================

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;
    use ieee.math_real.all;

use work.clog2_pkg.all;

-- ----------------------------------------------------------------------------
-- Individual dual-port (1 write, 1 read) synchronous RAM
-- ----------------------------------------------------------------------------
entity sync_ram_1r1w is
    generic (
        ADDR_WIDTH : positive := 10;
        DATA_WIDTH : positive := 16
    );
    port (
        clk   : in  std_logic;

        -- Write port (used for initialisation / test loading)
        wr_addr : in  std_logic_vector(ADDR_WIDTH - 1 downto 0);
        wr_data : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        wr_en   : in  std_logic;

        -- Read port
        rd_addr : in  std_logic_vector(ADDR_WIDTH - 1 downto 0);
        rd_re   : in  std_logic;
        rd_data : out std_logic_vector(DATA_WIDTH - 1 downto 0)
    );
end entity sync_ram_1r1w;

architecture rtl of sync_ram_1r1w is
    type ram_t is array (0 to 2**ADDR_WIDTH - 1)
        of std_logic_vector(DATA_WIDTH - 1 downto 0);

    signal mem : ram_t := (others => (others => '0'));
    signal rd_addr_r : std_logic_vector(ADDR_WIDTH - 1 downto 0);
begin
    p_rw : process (clk) is
    begin
        if rising_edge(clk) then
            if wr_en = '1' then
                mem(to_integer(unsigned(wr_addr))) <= wr_data;
            end if;
            rd_addr_r <= rd_addr;
        end if;
    end process p_rw;

    rd_data <= mem(to_integer(unsigned(rd_addr_r)));
end architecture rtl;



library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;
    use ieee.math_real.all;

use work.clog2_pkg.all;

entity weight_mem is
    generic (
        DATA_WIDTH : positive := 16;
        MODEL_DIM  : positive := 512;
        HIDDEN_DIM : positive := 2048
    );
    port (
        clk : in std_logic;

        -- ------------------------------------------------------------------
        -- MHA weight read ports
        -- ------------------------------------------------------------------
        w_q_addr : in  std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
        w_q_re   : in  std_logic;
        w_q_data : out std_logic_vector(DATA_WIDTH - 1 downto 0);

        w_k_addr : in  std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
        w_k_re   : in  std_logic;
        w_k_data : out std_logic_vector(DATA_WIDTH - 1 downto 0);

        w_v_addr : in  std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
        w_v_re   : in  std_logic;
        w_v_data : out std_logic_vector(DATA_WIDTH - 1 downto 0);

        w_o_addr : in  std_logic_vector(clog2(MODEL_DIM * MODEL_DIM) - 1 downto 0);
        w_o_re   : in  std_logic;
        w_o_data : out std_logic_vector(DATA_WIDTH - 1 downto 0);

        -- ------------------------------------------------------------------
        -- FFN weight & bias read ports
        -- ------------------------------------------------------------------
        w1_addr   : in  std_logic_vector(clog2(HIDDEN_DIM * MODEL_DIM) - 1 downto 0);
        w1_re     : in  std_logic;
        w1_rdata  : out std_logic_vector(DATA_WIDTH - 1 downto 0);

        b1_addr   : in  std_logic_vector(clog2(HIDDEN_DIM) - 1 downto 0);
        b1_re     : in  std_logic;
        b1_rdata  : out std_logic_vector(DATA_WIDTH - 1 downto 0);

        w2_addr   : in  std_logic_vector(clog2(MODEL_DIM * HIDDEN_DIM) - 1 downto 0);
        w2_re     : in  std_logic;
        w2_rdata  : out std_logic_vector(DATA_WIDTH - 1 downto 0);

        b2_addr   : in  std_logic_vector(clog2(MODEL_DIM) - 1 downto 0);
        b2_re     : in  std_logic;
        b2_rdata  : out std_logic_vector(DATA_WIDTH - 1 downto 0)
    );
end entity weight_mem;

architecture structural of weight_mem is

    -- Simple clog2 helper (local copy to avoid package dependency issues)
    function clog2(n : positive) return positive is
    begin
        return positive(ceil(log2(real(n))));
    end function clog2;

    ---------------------------------------------------------------------------
    -- Address widths
    ---------------------------------------------------------------------------
    constant AW_MHA : positive := clog2(MODEL_DIM * MODEL_DIM);   -- 18 for 512*512
    constant AW_W1  : positive := clog2(HIDDEN_DIM * MODEL_DIM);  -- 20 for 2048*512
    constant AW_B1  : positive := clog2(HIDDEN_DIM);              -- 11 for 2048
    constant AW_W2  : positive := clog2(MODEL_DIM * HIDDEN_DIM);  -- 20 for 512*2048
    constant AW_B2  : positive := clog2(MODEL_DIM);              -- 9  for 512

begin

    -- ========================================================================
    -- MHA weight memories
    -- ========================================================================

    u_w_q : entity work.sync_ram_1r1w
        generic map (ADDR_WIDTH => AW_MHA, DATA_WIDTH => DATA_WIDTH)
        port map (
            clk     => clk,
            wr_addr => (others => '0'), wr_data => (others => '0'), wr_en => '0',
            rd_addr => w_q_addr, rd_re => w_q_re, rd_data => w_q_data
        );

    u_w_k : entity work.sync_ram_1r1w
        generic map (ADDR_WIDTH => AW_MHA, DATA_WIDTH => DATA_WIDTH)
        port map (
            clk     => clk,
            wr_addr => (others => '0'), wr_data => (others => '0'), wr_en => '0',
            rd_addr => w_k_addr, rd_re => w_k_re, rd_data => w_k_data
        );

    u_w_v : entity work.sync_ram_1r1w
        generic map (ADDR_WIDTH => AW_MHA, DATA_WIDTH => DATA_WIDTH)
        port map (
            clk     => clk,
            wr_addr => (others => '0'), wr_data => (others => '0'), wr_en => '0',
            rd_addr => w_v_addr, rd_re => w_v_re, rd_data => w_v_data
        );

    u_w_o : entity work.sync_ram_1r1w
        generic map (ADDR_WIDTH => AW_MHA, DATA_WIDTH => DATA_WIDTH)
        port map (
            clk     => clk,
            wr_addr => (others => '0'), wr_data => (others => '0'), wr_en => '0',
            rd_addr => w_o_addr, rd_re => w_o_re, rd_data => w_o_data
        );

    -- ========================================================================
    -- FFN weight & bias memories
    -- ========================================================================

    u_w1 : entity work.sync_ram_1r1w
        generic map (ADDR_WIDTH => AW_W1, DATA_WIDTH => DATA_WIDTH)
        port map (
            clk     => clk,
            wr_addr => (others => '0'), wr_data => (others => '0'), wr_en => '0',
            rd_addr => w1_addr, rd_re => w1_re, rd_data => w1_rdata
        );

    u_b1 : entity work.sync_ram_1r1w
        generic map (ADDR_WIDTH => AW_B1, DATA_WIDTH => DATA_WIDTH)
        port map (
            clk     => clk,
            wr_addr => (others => '0'), wr_data => (others => '0'), wr_en => '0',
            rd_addr => b1_addr, rd_re => b1_re, rd_data => b1_rdata
        );

    u_w2 : entity work.sync_ram_1r1w
        generic map (ADDR_WIDTH => AW_W2, DATA_WIDTH => DATA_WIDTH)
        port map (
            clk     => clk,
            wr_addr => (others => '0'), wr_data => (others => '0'), wr_en => '0',
            rd_addr => w2_addr, rd_re => w2_re, rd_data => w2_rdata
        );

    u_b2 : entity work.sync_ram_1r1w
        generic map (ADDR_WIDTH => AW_B2, DATA_WIDTH => DATA_WIDTH)
        port map (
            clk     => clk,
            wr_addr => (others => '0'), wr_data => (others => '0'), wr_en => '0',
            rd_addr => b2_addr, rd_re => b2_re, rd_data => b2_rdata
        );

end architecture structural;

-- ============================================================================
-- End of file weight_mem.vhd
-- ============================================================================
