@echo off
REM ============================================================================
REM flash_transformer.bat - Flash the final ViT Transformer bitstream onto Basys 3
REM ============================================================================

set VIVADO_BAT="C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat"
set TCL_SCRIPT="C:\Users\maogo\OneDrive\transformer\transformer_poc\vivado_synth_test\flash_basys3.tcl"
set LOG_FILE="C:\Users\maogo\vivado_work\flash.log"
set JOU_FILE="C:\Users\maogo\vivado_work\flash.jou"

echo ============================================================================
echo [INFO] Ready to flash the Transformer bitstream to Basys 3
echo [INFO] Please ensure the board is plugged in via USB and powered on.
echo ============================================================================
echo.

if not exist %VIVADO_BAT% (
    echo [ERROR] Vivado not found at %VIVADO_BAT%
    echo Please verify your Vivado 2025.2 installation path.
    pause
    exit /b 1
)

echo [INFO] Releasing any locked JTAG / hw_server processes...
powershell -Command "Stop-Process -Name vivado, hw_server, cs_server -Force -ErrorAction SilentlyContinue" >nul 2>&1

echo [INFO] Starting Vivado JTAG programmer...
call %VIVADO_BAT% -mode batch -source %TCL_SCRIPT% -log %LOG_FILE% -journal %JOU_FILE%

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Flashing failed! Check the log: %LOG_FILE%
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================================
echo [SUCCESS] Basys 3 successfully programmed!
echo ============================================================================
echo.
pause
