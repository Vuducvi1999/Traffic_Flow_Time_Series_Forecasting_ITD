@echo off
cd /d "%~dp0"

:: TÊN SERVICE (Bạn có thể đổi tùy ý)
set SERVICE_NAME=FastApi_Traffic_Forecasting

:: KIỂM TRA: Nếu Service gọi file này, tham số %1 sẽ là "run"
if "%1"=="run" goto :start_api_server

:: Kiểm tra quyền Admin, nếu không có sẽ tự động đòi quyền Admin (Đã fix lỗi rỗng tham số trên PowerShell)
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
sc create %SERVICE_NAME% binPath= "\"%~f0\" run" start= auto
sc description %SERVICE_NAME% "Dich vu du bao luu luong giao thong FastAPI"
sc start %SERVICE_NAME%
echo Cai dat va khoi dong thanh cong!
pause
goto :eof

:uninstall_service
echo Dang dung va xoa Service...
sc stop %SERVICE_NAME% >nul 2>&1
sc delete %SERVICE_NAME%
echo Da go bo Service thanh cong!
pause
goto :eof

:start_api_server
call .venv\Scripts\activate.bat

python src/api_server.py