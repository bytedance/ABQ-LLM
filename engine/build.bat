set BUILD_DIR=%~dp0

@echo off
echo BUILD_DIR: %BUILD_DIR%
set backup_path=%PATH%

setlocal enabledelayedexpansion
echo "build x64"
rem call "C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/VC/Auxiliary/Build/vcvars64.bat"
call "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"

set BUILD_TYPE=Release
if exist build_win rd /s /q build_win
mkdir build_win
cd build_win

rem for release md x64
cmake %BUILD_DIR% -G "Visual Studio 17 2022" ^
        -DSM=86 ^
        -DENABLE_W2A2=ON ^
        -DENABLE_W2A4=ON ^
        -DENABLE_W2A6=ON ^
        -DENABLE_W2A8=ON ^
        -DENABLE_W3A3=ON ^
        -DENABLE_W3A8=ON ^
        -DENABLE_W4A4=ON ^
        -DENABLE_W4A8=ON ^
        -DENABLE_W5A5=ON ^
        -DENABLE_W6A6=ON ^
        -DENABLE_W7A7=ON ^
        -DENABLE_W8A8=ON ^
        -DCMAKE_BUILD_TYPE=!BUILD_TYPE! ^
        -DCMAKE_EXPORT_COMPILE_COMMANDS=on

cmake --build . --config !BUILD_TYPE! -j8
cd ..

set PATH=%backup_path%