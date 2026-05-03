library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.utilities.all;
use work.clog2_pkg.all;

entity ffn is
    generic (
        DATA_WIDTH : integer := 16;
        HIDDEN_DIM : integer := 2048;
        MODEL_DIM  : integer := 512;
        max_size_x : integer := 512
    );
    port (
        clk       : in  std_logic;
        rstn     : in  std_logic;

        -- Control
        start     : in  std_logic;
        done      : out std_logic;

        -- Streaming input (sequential, one element per cycle, MODEL_DIM total)
        i_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        i_valid   : in  std_logic;
        i_last    : in  std_logic;
        i_channel : in  integer range 0 to max_size_x - 1;

        -- Streaming output (sequential, one element per cycle, MODEL_DIM total)
        o_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        o_valid   : out std_logic;
        o_last    : out std_logic;
        o_channel : out integer range 0 to max_size_x - 1;

        -- W_1 weight memory: (HIDDEN_DIM x MODEL_DIM) -- FC1 weights
        w1_addr   : out std_logic_vector(clog2(HIDDEN_DIM * MODEL_DIM) - 1 downto 0);
        w1_rdata  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        w1_re     : out std_logic;

        -- b_1 bias memory: (HIDDEN_DIM x 1) -- FC1 biases
        b1_addr   : out std_logic_vector(clog2(HIDDEN_DIM) - 1 downto 0);
        b1_rdata  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        b1_re     : out std_logic;

        -- W_2 weight memory: (MODEL_DIM x HIDDEN_DIM) -- FC2 weights
        w2_addr   : out std_logic_vector(clog2(MODEL_DIM * HIDDEN_DIM) - 1 downto 0);
        w2_rdata  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        w2_re     : out std_logic;

        -- b_2 bias memory: (MODEL_DIM x 1) -- FC2 biases
        b2_addr   : out std_logic_vector(clog2(MODEL_DIM) - 1 downto 0);
        b2_rdata  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
        b2_re     : out std_logic
    );
end entity ffn;

architecture rtl of ffn is

    ---------------------------------------------------------------------------
    -- FSM state declaration
    ---------------------------------------------------------------------------
    type fsm_state_t is (
        IDLE,
        FC1_MATMUL,
        GELU_ACTIVATE,
        FC2_MATMUL,
        DONE
    );

    signal state      : fsm_state_t;
    signal next_state : fsm_state_t;

    ---------------------------------------------------------------------------
    -- Component declarations from accel library
    ---------------------------------------------------------------------------

    -- Output-stationary GEMM: computes C = A * B + C
    -- Matrix dimensions: A(M x K), B(K x N), C(M x N)
    -- All ports use memory-mapped read interface: component drives _addr and _re,
    -- expects _data to be valid on the next cycle (pipelined read).
    component gemm_mm is
        generic (
            DATA_WIDTH : integer;
            M          : integer;
            K          : integer;
            N          : integer
        );
        port (
            clk       : in  std_logic;
            rstn     : in  std_logic;
            start     : in  std_logic;
            done      : out std_logic;

            -- A matrix (row-major)
            a_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            a_addr    : out std_logic_vector(clog2(M * K) - 1 downto 0);
            a_re      : out std_logic;
            a_valid   : in  std_logic;

            -- B matrix (row-major)
            b_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            b_addr    : out std_logic_vector(clog2(K * N) - 1 downto 0);
            b_re      : out std_logic;
            b_valid   : in  std_logic;

            -- C matrix (row-major, bias-add)
            c_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            c_addr    : out std_logic_vector(clog2(M * N) - 1 downto 0);
            c_re      : out std_logic;
            c_valid   : in  std_logic;

            -- Output matrix (row-major, streaming)
            o_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
            o_valid   : out std_logic;
            o_last    : out std_logic;
            o_channel : out integer range 0 to max_size_x - 1
        );
    end component;

    -- Per-element activation function applied to a stream of partial sums.
    component psum_activation is
        generic (
            DATA_WIDTH   : integer;
            NUM_ELEMENTS : integer;
            MODE         : string
        );
        port (
            clk       : in  std_logic;
            rstn     : in  std_logic;
            start     : in  std_logic;
            done      : out std_logic;

            i_data    : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            i_valid   : in  std_logic;
            i_last    : in  std_logic;
            i_channel : in  integer range 0 to max_size_x - 1;

            o_data    : out std_logic_vector(DATA_WIDTH - 1 downto 0);
            o_valid   : out std_logic;
            o_last    : out std_logic;
            o_channel : out integer range 0 to max_size_x - 1
        );
    end component;

    ---------------------------------------------------------------------------
    -- Address width constants
    ---------------------------------------------------------------------------
    constant AW_W1 : integer := clog2(HIDDEN_DIM * MODEL_DIM);     -- W_1 address width
    constant AW_B1 : integer := clog2(HIDDEN_DIM);                  -- b_1 address width
    constant AW_W2 : integer := clog2(MODEL_DIM * HIDDEN_DIM);     -- W_2 address width
    constant AW_B2 : integer := clog2(MODEL_DIM);                   -- b_2 address width
    constant AW_FC1_B : integer := clog2(MODEL_DIM);                -- FC1 B-matrix addr (K*N = MODEL_DIM*1)
    constant AW_FC2_B : integer := clog2(HIDDEN_DIM);              -- FC2 B-matrix addr (K*N = HIDDEN_DIM*1)

    ---------------------------------------------------------------------------
    -- Internal buffers (vectors stored as arrays for addressable access)
    ---------------------------------------------------------------------------
    type vec_model_t  is array (0 to MODEL_DIM - 1)  of std_logic_vector(DATA_WIDTH - 1 downto 0);
    type vec_hidden_t is array (0 to HIDDEN_DIM - 1) of std_logic_vector(DATA_WIDTH - 1 downto 0);

    -- Input buffer: captured from i_data stream during IDLE
    signal input_buffer     : vec_model_t;
    signal input_cnt        : integer range 0 to MODEL_DIM - 1;
    signal input_captured   : std_logic;

    -- GELU output buffer: captured during GELU_ACTIVATE, read by FC2 GEMM
    signal gelu_out_buffer  : vec_hidden_t;
    signal gelu_out_cnt     : integer range 0 to HIDDEN_DIM - 1;
    signal gelu_capture_done : std_logic;

    ---------------------------------------------------------------------------
    -- Internal address buses from gemm_os B-ports (not exposed at top level)
    ---------------------------------------------------------------------------
    signal fc1_b_addr       : std_logic_vector(AW_FC1_B - 1 downto 0);
    signal fc2_b_addr       : std_logic_vector(AW_FC2_B - 1 downto 0);

    ---------------------------------------------------------------------------
    -- Start / done handshake signals for each stage
    ---------------------------------------------------------------------------
    signal fc1_start   : std_logic;
    signal fc1_done    : std_logic;
    signal gelu_start  : std_logic;
    signal gelu_done   : std_logic;
    signal fc2_start   : std_logic;
    signal fc2_done    : std_logic;

    ---------------------------------------------------------------------------
    -- Data paths between stages
    ---------------------------------------------------------------------------
    signal fc1_odata  : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal fc1_ovalid : std_logic;
    signal fc1_olast  : std_logic;
    signal fc1_ochan  : integer range 0 to max_size_x - 1;

    signal gelu_odata  : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal gelu_ovalid : std_logic;
    signal gelu_olast  : std_logic;
    signal gelu_ochan  : integer range 0 to max_size_x - 1;

    signal fc2_odata  : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal fc2_ovalid : std_logic;
    signal fc2_olast  : std_logic;
    signal fc2_ochan  : integer range 0 to max_size_x - 1;

    -- Pipelined data for B-port reads (registered output of buffer lookup)
    signal fc1_b_data_reg : std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal fc2_b_data_reg : std_logic_vector(DATA_WIDTH - 1 downto 0);

begin

    ---------------------------------------------------------------------------
    -- FSM: state register
    ---------------------------------------------------------------------------
    proc_fsm_reg : process(clk, rstn)
    begin
        if rstn = '0' then
            state <= IDLE;
        elsif rising_edge(clk) then
            state <= next_state;
        end if;
    end process proc_fsm_reg;

    ---------------------------------------------------------------------------
    -- FSM: next-state logic
    ---------------------------------------------------------------------------
    proc_fsm_next : process(all)
    begin
        next_state <= state;

        case state is
            when IDLE =>
                if start = '1' or input_captured = '1' then
                    next_state <= FC1_MATMUL;
                end if;

            when FC1_MATMUL =>
                if fc1_done = '1' then
                    next_state <= GELU_ACTIVATE;
                end if;

            when GELU_ACTIVATE =>
                if gelu_done = '1' and gelu_capture_done = '1' then
                    next_state <= FC2_MATMUL;
                end if;

            when FC2_MATMUL =>
                if fc2_done = '1' then
                    next_state <= DONE;
                end if;

            when DONE =>
                next_state <= IDLE;

        end case;
    end process proc_fsm_next;

    ---------------------------------------------------------------------------
    -- Input capture: accumulate MODEL_DIM elements from i_data stream
    ---------------------------------------------------------------------------
    proc_input_capture : process(clk, rstn)
    begin
        if rstn = '0' then
            input_cnt      <= 0;
            input_captured <= '0';
            input_buffer   <= (others => (others => '0'));
        elsif rising_edge(clk) then
            if state = IDLE then
                if i_valid = '1' then
                    input_buffer(input_cnt) <= i_data;
                    if input_cnt = MODEL_DIM - 1 then
                        input_cnt      <= 0;
                        input_captured <= '1';
                    else
                        input_cnt <= input_cnt + 1;
                    end if;
                end if;
            else
                input_captured <= '0';
            end if;
        end if;
    end process proc_input_capture;

    ---------------------------------------------------------------------------
    -- FC1 B-data pipelined read from input_buffer
    -- gemm_os drives fc1_b_addr; we register the lookup result to match the
    -- pipelined read timing (addr out -> data in on next cycle).
    ---------------------------------------------------------------------------
    proc_fc1_b_read : process(clk, rstn)
    begin
        if rstn = '0' then
            fc1_b_data_reg <= (others => '0');
        elsif rising_edge(clk) then
            fc1_b_data_reg <= input_buffer(to_integer(unsigned(fc1_b_addr)));
        end if;
    end process proc_fc1_b_read;

    ---------------------------------------------------------------------------
    -- Done output
    ---------------------------------------------------------------------------
    done <= '1' when state = DONE else '0';

    ---------------------------------------------------------------------------
    -- FC1 start (combinatorial pulse when input fully captured)
    ---------------------------------------------------------------------------
    fc1_start <= '1' when state = IDLE and (start = '1' or input_captured = '1') else '0';

    ---------------------------------------------------------------------------
    -- GELU start (registered pulse on FC1 completion)
    ---------------------------------------------------------------------------
    proc_gelu_start : process(clk, rstn)
    begin
        if rstn = '0' then
            gelu_start <= '0';
        elsif rising_edge(clk) then
            if state = FC1_MATMUL and fc1_done = '1' then
                gelu_start <= '1';
            else
                gelu_start <= '0';
            end if;
        end if;
    end process proc_gelu_start;

    ---------------------------------------------------------------------------
    -- GELU output capture: buffer streaming GELU output for FC2 B-port reads
    ---------------------------------------------------------------------------
    proc_gelu_output_capture : process(clk, rstn)
    begin
        if rstn = '0' then
            gelu_out_buffer  <= (others => (others => '0'));
            gelu_out_cnt     <= 0;
            gelu_capture_done <= '0';
        elsif rising_edge(clk) then
            if state = GELU_ACTIVATE then
                if gelu_ovalid = '1' then
                    gelu_out_buffer(gelu_out_cnt) <= gelu_odata;
                    if gelu_out_cnt = HIDDEN_DIM - 1 then
                        gelu_out_cnt     <= 0;
                        gelu_capture_done <= '1';
                    else
                        gelu_out_cnt <= gelu_out_cnt + 1;
                    end if;
                end if;
            else
                gelu_capture_done <= '0';
                gelu_out_cnt      <= 0;
            end if;
        end if;
    end process proc_gelu_output_capture;

    ---------------------------------------------------------------------------
    -- FC2 start (registered pulse when GELU output fully buffered)
    ---------------------------------------------------------------------------
    proc_fc2_start : process(clk, rstn)
    begin
        if rstn = '0' then
            fc2_start <= '0';
        elsif rising_edge(clk) then
            if state = GELU_ACTIVATE and gelu_done = '1' and gelu_capture_done = '1' then
                fc2_start <= '1';
            else
                fc2_start <= '0';
            end if;
        end if;
    end process proc_fc2_start;

    ---------------------------------------------------------------------------
    -- FC2 B-data pipelined read from gelu_out_buffer
    ---------------------------------------------------------------------------
    proc_fc2_b_read : process(clk, rstn)
    begin
        if rstn = '0' then
            fc2_b_data_reg <= (others => '0');
        elsif rising_edge(clk) then
            fc2_b_data_reg <= gelu_out_buffer(to_integer(unsigned(fc2_b_addr)));
        end if;
    end process proc_fc2_b_read;

    ---------------------------------------------------------------------------
    -- FC1 GEMM instance
    --   Result(hid) = sum_k W_1(hid, k) * x(k) + b_1(hid)
    --   A = W_1:  (HIDDEN_DIM x MODEL_DIM)  -- from external memory
    --   B = x:    (MODEL_DIM x 1)            -- from input_buffer
    --   C = b_1:  (HIDDEN_DIM x 1)           -- from external memory
    ---------------------------------------------------------------------------
    u_gemm_fc1 : gemm_mm
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            M          => HIDDEN_DIM,
            K          => MODEL_DIM,
            N          => 1
        )
        port map (
            clk       => clk,
            rstn     => rstn,
            start     => fc1_start,
            done      => fc1_done,

            -- A-port: external W_1 memory
            a_data    => w1_rdata,
            a_addr    => w1_addr,
            a_re      => w1_re,
            a_valid   => '1',

            -- B-port: internal input_buffer (pipelined lookup)
            b_data    => fc1_b_data_reg,
            b_addr    => fc1_b_addr,
            b_re      => open,
            b_valid   => '1',

            -- C-port: external b_1 memory
            c_data    => b1_rdata,
            c_addr    => b1_addr,
            c_re      => b1_re,
            c_valid   => '1',

            -- Output stream
            o_data    => fc1_odata,
            o_valid   => fc1_ovalid,
            o_last    => fc1_olast,
            o_channel => fc1_ochan
        );

    ---------------------------------------------------------------------------
    -- GELU activation instance
    --   Applies GELU element-wise to the FC1 output stream.
    ---------------------------------------------------------------------------
    u_gelu : psum_activation
        generic map (
            DATA_WIDTH   => DATA_WIDTH,
            NUM_ELEMENTS => HIDDEN_DIM,
            MODE         => "GELU"
        )
        port map (
            clk       => clk,
            rstn     => rstn,
            start     => gelu_start,
            done      => gelu_done,

            i_data    => fc1_odata,
            i_valid   => fc1_ovalid,
            i_last    => fc1_olast,
            i_channel => fc1_ochan,

            o_data    => gelu_odata,
            o_valid   => gelu_ovalid,
            o_last    => gelu_olast,
            o_channel => gelu_ochan
        );

    ---------------------------------------------------------------------------
    -- FC2 GEMM instance
    --   Result(mdl) = sum_hid W_2(mdl, hid) * GELU(hid) + b_2(mdl)
    --   A = W_2:        (MODEL_DIM x HIDDEN_DIM)  -- from external memory
    --   B = GELU_out:   (HIDDEN_DIM x 1)           -- from gelu_out_buffer
    --   C = b_2:        (MODEL_DIM x 1)            -- from external memory
    ---------------------------------------------------------------------------
    u_gemm_fc2 : gemm_mm
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            M          => MODEL_DIM,
            K          => HIDDEN_DIM,
            N          => 1
        )
        port map (
            clk       => clk,
            rstn     => rstn,
            start     => fc2_start,
            done      => fc2_done,

            -- A-port: external W_2 memory
            a_data    => w2_rdata,
            a_addr    => w2_addr,
            a_re      => w2_re,
            a_valid   => '1',

            -- B-port: internal gelu_out_buffer (pipelined lookup)
            b_data    => fc2_b_data_reg,
            b_addr    => fc2_b_addr,
            b_re      => open,
            b_valid   => '1',

            -- C-port: external b_2 memory
            c_data    => b2_rdata,
            c_addr    => b2_addr,
            c_re      => b2_re,
            c_valid   => '1',

            -- Output stream
            o_data    => fc2_odata,
            o_valid   => fc2_ovalid,
            o_last    => fc2_olast,
            o_channel => fc2_ochan
        );

    ---------------------------------------------------------------------------
    -- Output forwarding: stream FC2 GEMM output during FC2_MATMUL state.
    -- Once fc2_done asserts, the FSM moves to DONE for one cycle then IDLE.
    ---------------------------------------------------------------------------
    o_data    <= fc2_odata;
    o_valid   <= fc2_ovalid when (state = FC2_MATMUL or state = DONE) else '0';
    o_last    <= fc2_olast;
    o_channel <= fc2_ochan;

end architecture rtl;
