--------------------------------------------------------------------------------
-- Dummy package to satisfy "library accel; use accel.utilities.all;" in PoC
--------------------------------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;

package utilities is

    -- max_size_x used throughout the project as a generic upper-bound constant
    constant max_size_x : integer := 512;

end package utilities;

--------------------------------------------------------------------------------
-- End of file utilities.vhd
--------------------------------------------------------------------------------
