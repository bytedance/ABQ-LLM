

@REM $env:Path += ";C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64"
@REM echo %$env:Path%

set backup_path=%PATH%

@echo off
set CUTLASS_PATH=..\..\..\3rdparty\cuda\cutlass
set INCLUDE_FLAG=-I%CUTLASS_PATH%\tools\util\include -I%CUTLASS_PATH%\include -I.\

set BUILD_FLAG= -std=c++17 -g -w -arch=sm_86

nvcc.exe %INCLUDE_FLAG% ^
         %BUILD_FLAG% ^
         -o test_w4a4_int_base.exe test_w4a4_int_base.cu

set PATH=%backup_path%