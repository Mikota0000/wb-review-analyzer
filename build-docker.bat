@echo off
chcp 65001 >nul
echo on

setlocal EnableDelayedExpansion

rem === Configuration ===
set "PROJECT_NAME=wildberries-review-analyzer"
set "IMAGE_NAME=wb-analyzer"
set "CONTAINER_NAME=wb-review-analyzer"

:check_files
echo [INFO] Checking required files...
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found!
    exit /b 1
)
if not exist "app" (
    echo [ERROR] Directory app\ not found!
    exit /b 1
)
if not exist "models" (
    echo [WARNING] Directory models\ not found. Make sure model is trained.
)
echo [SUCCESS] All required files are present.
goto :eof

:build
call :check_files
echo [INFO] Building Docker image...
docker build --target production ^
    --tag %IMAGE_NAME%:latest ^
    --tag %IMAGE_NAME%:%DATE:~6,4%%DATE:~3,2%%DATE:~0,2%-%%TIME:~0,2%%TIME:~3,2%%TIME:~6,2% ^
    .
if errorlevel 1 exit /b 1
echo [SUCCESS] Docker image built.
goto :eof

:run
call :check_files
call :build
echo [INFO] Starting container...
for /f "tokens=*" %%i in ('docker ps -q -f name=%CONTAINER_NAME%') do (
    if defined %%i (
        echo [WARNING] Container already running. Removing...
        docker stop %CONTAINER_NAME%
        docker rm %CONTAINER_NAME%
    )
)
mkdir logs 2>nul
mkdir temp 2>nul

docker run -d --name %CONTAINER_NAME% ^
    -p 8000:8000 ^
    -v "%cd%\models":/app/models:ro ^
    -v "%cd%\logs":/app/logs ^
    -v "%cd%\temp":/app/temp ^
    --restart unless-stopped ^
    %IMAGE_NAME%:latest

if errorlevel 1 exit /b 1
echo [SUCCESS] Container started.
echo [INFO] API: http://localhost:8000
echo [INFO] Swagger: http://localhost:8000/docs

call :status
goto :eof

:compose
call :check_files
echo [INFO] Starting via docker-compose...
mkdir logs 2>nul & mkdir temp 2>nul & mkdir models 2>nul
docker-compose up --build -d
if errorlevel 1 exit /b 1
echo [SUCCESS] Services started with docker-compose.
echo [INFO] API: http://localhost:8000
goto :eof

:stop
echo [INFO] Stopping container...
for /f "tokens=*" %%i in ('docker ps -q -f name=%CONTAINER_NAME%') do (
    if defined %%i (
        docker stop %CONTAINER_NAME%
        docker rm %CONTAINER_NAME%
        echo [SUCCESS] Container stopped.
        goto :eof
    )
)
echo [WARNING] Container not running.
goto :eof

:stop-compose
echo [INFO] Stopping docker-compose...
docker-compose down
echo [SUCCESS] All services stopped.
goto :eof

:logs
echo [INFO] Showing logs...
docker logs -f %CONTAINER_NAME%
goto :eof

:status
echo [INFO] Container status:
docker ps -f name=%CONTAINER_NAME%
echo [INFO] Health check...
timeout /t 5 >nul
curl -f http://localhost:8000/api/v1/health >nul 2>&1
if errorlevel 1 (
    echo [ERROR] API unreachable.
) else (
    echo [SUCCESS] API is up.
)
goto :eof

:clean
echo [INFO] Cleaning Docker resources...
call :stop
for /f "tokens=*" %%i in ('docker images -q %IMAGE_NAME%') do (
    if defined %%i (
        docker rmi %%i
        echo [SUCCESS] Docker images removed.
    )
)
docker system prune -f
echo [SUCCESS] Cleanup done.
goto :eof

:help
echo Usage: build-docker.bat [command]
echo.
echo Commands:
echo   build          Build Docker image
echo   run            Build & run container (default)
echo   compose        Start via docker-compose
echo   stop           Stop container
echo   stop-compose   Stop docker-compose
echo   logs           Show container logs
echo   status         Check status & health
echo   clean          Clean Docker resources
echo   help           Show this help
goto :eof

rem — dispatch
if "%~1"=="" goto run
if "%~1"=="build" goto build
if "%~1"=="run" goto run
if "%~1"=="compose" goto compose
if "%~1"=="stop" goto stop
if "%~1"=="stop-compose" goto stop-compose
if "%~1"=="logs" goto logs
if "%~1"=="status" goto status
if "%~1"=="clean" goto clean
if "%~1"=="help" goto help

echo [ERROR] Unknown command: %1
goto help
