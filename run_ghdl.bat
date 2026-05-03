@echo off
REM ============================================================================
REM run_ghdl.bat — Windows batch script to compile and simulate the Transformer
REM encoder testbench using GHDL.
REM ============================================================================
REM Prerequisites: GHDL installed and on PATH
REM                mingw32-make (optional, for Makefile usage)
REM ============================================================================

echo [INFO] Cleaning previous build artefacts...
if exist work rmdir /s /q work
if exist *.cf del /q *.cf
if exist *.vcd del /q *.vcd
if exist *.ghw del /q *.ghw
if exist mha_out.txt del /q mha_out.txt
if exist ffn_out.txt del /q ffn_out.txt
if exist encoder_out.txt del /q encoder_out.txt

mkdir work 2>nul

echo [INFO] Analysing VHDL source files...

REM ---------------------------------------------------------------------------
REM Dependency order: packages / low-level -> high-level
REM ---------------------------------------------------------------------------

ghdl -a --workdir=work --std=08 clog2_pkg.vhd
if %errorlevel% neq 0 goto ERROR

ghdl -a --workdir=work --std=08 gemm_os.vhd
if %errorlevel% neq 0 goto ERROR

ghdl -a --workdir=work --std=08 gemm_os_adapter.vhd
if %errorlevel% neq 0 goto ERROR

ghdl -a --workdir=work --std=08 softmax.vhd
if %errorlevel% neq 0 goto ERROR

ghdl -a --workdir=work --std=08 scalar_ops.vhd
if %errorlevel% neq 0 goto ERROR

ghdl -a --workdir=work --std=08 layernorm.vhd
if %errorlevel% neq 0 goto ERROR

ghdl -a --workdir=work --std=08 weight_mem.vhd
if %errorlevel% neq 0 goto ERROR

ghdl -a --workdir=work --std=08 mha_controller.vhd
if %errorlevel% neq 0 goto ERROR

ghdl -a --workdir=work --std=08 ffn.vhd
if %errorlevel% neq 0 goto ERROR

ghdl -a --workdir=work --std=08 residual_add.vhd
if %errorlevel% neq 0 goto ERROR

ghdl -a --workdir=work --std=08 control_unit.vhd
if %errorlevel% neq 0 goto ERROR

ghdl -a --workdir=work --std=08 encoder_block.vhd
if %errorlevel% neq 0 goto ERROR

ghdl -a --workdir=work --std=08 tb_encoder_block.vhd
if %errorlevel% neq 0 goto ERROR

echo [INFO] Elaborating testbench...
ghdl -e --workdir=work --std=08 tb_encoder_block
if %errorlevel% neq 0 goto ERROR

echo [INFO] Running simulation (VCD dump)...
ghdl -r --workdir=work --std=08 tb_encoder_block --wave=wave.ghw --stop-time=100us
if %errorlevel% neq 0 goto ERROR

echo.
echo [SUCCESS] Simulation completed.
echo   Waveform : wave.ghw   (open with GTKWave)
echo   MHA out  : mha_out.txt
echo   FFN out  : ffn_out.txt
echo   Encoder  : encoder_out.txt
goto END

:ERROR
echo.
echo [ERROR] GHDL step failed (exit code %errorlevel%).
pause
goto END

:END
