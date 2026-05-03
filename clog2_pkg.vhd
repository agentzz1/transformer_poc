library ieee;
use ieee.math_real.all;

package clog2_pkg is
    function clog2(n : positive) return positive;
end package clog2_pkg;

package body clog2_pkg is
    function clog2(n : positive) return positive is
    begin
        return positive(ceil(log2(real(n))));
    end function clog2;
end package body clog2_pkg;
