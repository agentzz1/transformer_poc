@echo off
REM run_basys_sim.bat - Windows batch script to run top-level simulation of tb_basys3_top

echo [INFO] Cleaning work_basys...
if exist work_basys rmdir /s /q work_basys
if exist *_vhdl.txt del /q *_vhdl.txt
if exist basys_sim.log del /q basys_sim.log

mkdir work_basys 2>nul

echo [INFO] Analyzing unisim_dummy...
ghdl -a --std=08 --workdir=work_basys -P=work_basys --work=unisim unisim_dummy.vhd
if %errorlevel% neq 0 goto ERROR

echo [INFO] Analyzing design files in dependency order...
for %%f in (clog2_pkg.vhd weights_pkg.vhd utilities.vhd gemm_os.vhd gemm_os_adapter.vhd softmax.vhd scalar_ops.vhd layernorm.vhd weight_mem.vhd gemm_mm.vhd psum_activation.vhd mha_controller.vhd ffn.vhd residual_add.vhd control_unit.vhd encoder_block.vhd patch_embed.vhd classifier.vhd basys3_top.vhd tb_basys3_top.vhd) do (
    echo [a] %%f
    ghdl -a --std=08 --workdir=work_basys -P=work_basys %%f
    if %errorlevel% neq 0 goto ERROR
)

echo [INFO] Elaborating tb_basys3_top...
ghdl -e --std=08 --workdir=work_basys -P=work_basys tb_basys3_top
if %errorlevel% neq 0 goto ERROR

echo [INFO] Running simulation (this may take a minute due to UART timing)...
ghdl -r --std=08 --workdir=work_basys -P=work_basys tb_basys3_top --ieee-asserts=disable --stop-time=120ms > basys_sim.log 2>&1
if %errorlevel% neq 0 goto ERROR

echo.
echo [SUCCESS] Top-level simulation completed successfully.
echo Log: basys_sim.log
echo.
echo --- Last 20 lines of basys_sim.log ---
powershell -Command "Get-Content basys_sim.log -Tail 20"
echo.
goto END

:ERROR
echo.
echo [ERROR] GHDL step failed (exit code %errorlevel%).
exit /b %errorlevel%

:END
