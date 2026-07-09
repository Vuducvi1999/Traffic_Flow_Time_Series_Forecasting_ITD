@echo off
cd /d "%~dp0"

set SERVICE_NAME=FastApi_Traffic_Forecasting
set NSSM_EXE="%~dp0nssm\win64\nssm.exe"

:: Kiểm tra quyền Admin
net session >nul 2>&1
if %errorLevel% neq 0 (
    if "%1"=="" (
        powershell Start-Process -FilePath "%~f0" -Verb RunAs
    ) else (
        powershell Start-Process -FilePath "%~f0" -ArgumentList "'%1'" -Verb RunAs
    )
    exit /b
)

echo 1. Cai dat va Khoi dong Service
echo 2. Go bo (Delete) Service
echo ===================================================
set /p choise="Nhap lua chon cua ban (1 hoac 2): "

if "%choise%"=="1" goto :install_service
if "%choise%"=="2" goto :uninstall_service
goto :eof

:install_service
echo Dang cai dat Service...

%NSSM_EXE% install %SERVICE_NAME% "%~dp0.venv\Scripts\python.exe" "src/api_server.py"
%NSSM_EXE% set %SERVICE_NAME% AppDirectory "%~dp0"
%NSSM_EXE% set %SERVICE_NAME% Description "Dich vu du bao luu luong giao thong FastAPI"
%NSSM_EXE% start %SERVICE_NAME%
echo Cai dat va khoi dong thanh cong!
pause
goto :eof

:uninstall_service
echo Dang dung va xoa Service...
%NSSM_EXE% stop %SERVICE_NAME% >nul 2>&1
%NSSM_EXE% remove %SERVICE_NAME% confirm
echo Da go bo Service thanh cong!
pause
goto :eof