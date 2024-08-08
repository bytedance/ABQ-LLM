@echo off

if exist results rd /s /q results
mkdir results

set BITS=2 3 4 5 6 7 8

set M=1
set N=4096
set K=4096
(for %%b in (%BITS%) do (
   .\bin\Release\test_any_wmma.exe %M% %N% %K% %%b %%b 1 > ./results/%M%x%N%x%K%_w%%ba%%b.txt
))
.\bin\Release\test_any_wmma.exe %M% %N% %K% 4 2 1 > ./results/%M%x%N%x%K%_w2a4.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 6 2 1 > ./results/%M%x%N%x%K%_w2a6.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 2 1 > ./results/%M%x%N%x%K%_w2a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 3 1 > ./results/%M%x%N%x%K%_w3a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 4 1 > ./results/%M%x%N%x%K%_w4a8.txt

set M=1
set N=8192
set K=1024
(for %%b in (%BITS%) do (
   .\bin\Release\test_any_wmma.exe %M% %N% %K% %%b %%b 1 > ./results/%M%x%N%x%K%_w%%ba%%b.txt
))
.\bin\Release\test_any_wmma.exe %M% %N% %K% 4 2 1 > ./results/%M%x%N%x%K%_w2a4.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 6 2 1 > ./results/%M%x%N%x%K%_w2a6.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 2 1 > ./results/%M%x%N%x%K%_w2a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 3 1 > ./results/%M%x%N%x%K%_w3a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 4 1 > ./results/%M%x%N%x%K%_w4a8.txt

set M=1
set N=11008
set K=4096
(for %%b in (%BITS%) do (
   .\bin\Release\test_any_wmma.exe %M% %N% %K% %%b %%b 1 > ./results/%M%x%N%x%K%_w%%ba%%b.txt
))
.\bin\Release\test_any_wmma.exe %M% %N% %K% 4 2 1 > ./results/%M%x%N%x%K%_w2a4.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 6 2 1 > ./results/%M%x%N%x%K%_w2a6.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 2 1 > ./results/%M%x%N%x%K%_w2a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 3 1 > ./results/%M%x%N%x%K%_w3a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 4 1 > ./results/%M%x%N%x%K%_w4a8.txt

set M=1
set N=5120
set K=5120
(for %%b in (%BITS%) do (
   .\bin\Release\test_any_wmma.exe %M% %N% %K% %%b %%b 1 > ./results/%M%x%N%x%K%_w%%ba%%b.txt
))
.\bin\Release\test_any_wmma.exe %M% %N% %K% 4 2 1 > ./results/%M%x%N%x%K%_w2a4.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 6 2 1 > ./results/%M%x%N%x%K%_w2a6.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 2 1 > ./results/%M%x%N%x%K%_w2a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 3 1 > ./results/%M%x%N%x%K%_w3a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 4 1 > ./results/%M%x%N%x%K%_w4a8.txt

set M=1
set N=4096
set K=11008
(for %%b in (%BITS%) do (
   .\bin\Release\test_any_wmma.exe %M% %N% %K% %%b %%b 1 > ./results/%M%x%N%x%K%_w%%ba%%b.txt
))
.\bin\Release\test_any_wmma.exe %M% %N% %K% 4 2 1 > ./results/%M%x%N%x%K%_w2a4.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 6 2 1 > ./results/%M%x%N%x%K%_w2a6.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 2 1 > ./results/%M%x%N%x%K%_w2a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 3 1 > ./results/%M%x%N%x%K%_w3a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 4 1 > ./results/%M%x%N%x%K%_w4a8.txt

set M=4
set N=4096
set K=4096
(for %%b in (%BITS%) do (
   .\bin\Release\test_any_wmma.exe %M% %N% %K% %%b %%b 1 > ./results/%M%x%N%x%K%_w%%ba%%b.txt
))
.\bin\Release\test_any_wmma.exe %M% %N% %K% 4 2 1 > ./results/%M%x%N%x%K%_w2a4.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 6 2 1 > ./results/%M%x%N%x%K%_w2a6.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 2 1 > ./results/%M%x%N%x%K%_w2a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 3 1 > ./results/%M%x%N%x%K%_w3a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 4 1 > ./results/%M%x%N%x%K%_w4a8.txt

set M=4
set N=8192
set K=1024
(for %%b in (%BITS%) do (
   .\bin\Release\test_any_wmma.exe %M% %N% %K% %%b %%b 1 > ./results/%M%x%N%x%K%_w%%ba%%b.txt
))
.\bin\Release\test_any_wmma.exe %M% %N% %K% 4 2 1 > ./results/%M%x%N%x%K%_w2a4.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 6 2 1 > ./results/%M%x%N%x%K%_w2a6.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 2 1 > ./results/%M%x%N%x%K%_w2a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 3 1 > ./results/%M%x%N%x%K%_w3a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 4 1 > ./results/%M%x%N%x%K%_w4a8.txt

set M=4
set N=11008
set K=4096
(for %%b in (%BITS%) do (
   .\bin\Release\test_any_wmma.exe %M% %N% %K% %%b %%b 1 > ./results/%M%x%N%x%K%_w%%ba%%b.txt
))
.\bin\Release\test_any_wmma.exe %M% %N% %K% 4 2 1 > ./results/%M%x%N%x%K%_w2a4.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 6 2 1 > ./results/%M%x%N%x%K%_w2a6.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 2 1 > ./results/%M%x%N%x%K%_w2a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 3 1 > ./results/%M%x%N%x%K%_w3a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 4 1 > ./results/%M%x%N%x%K%_w4a8.txt

set M=4
set N=5120
set K=5120
(for %%b in (%BITS%) do (
   .\bin\Release\test_any_wmma.exe %M% %N% %K% %%b %%b 1 > ./results/%M%x%N%x%K%_w%%ba%%b.txt
))
.\bin\Release\test_any_wmma.exe %M% %N% %K% 4 2 1 > ./results/%M%x%N%x%K%_w2a4.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 6 2 1 > ./results/%M%x%N%x%K%_w2a6.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 2 1 > ./results/%M%x%N%x%K%_w2a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 3 1 > ./results/%M%x%N%x%K%_w3a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 4 1 > ./results/%M%x%N%x%K%_w4a8.txt

set M=4
set N=4096
set K=11008
(for %%b in (%BITS%) do (
   .\bin\Release\test_any_wmma.exe %M% %N% %K% %%b %%b 1 > ./results/%M%x%N%x%K%_w%%ba%%b.txt
))
.\bin\Release\test_any_wmma.exe %M% %N% %K% 4 2 1 > ./results/%M%x%N%x%K%_w2a4.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 6 2 1 > ./results/%M%x%N%x%K%_w2a6.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 2 1 > ./results/%M%x%N%x%K%_w2a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 3 1 > ./results/%M%x%N%x%K%_w3a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 4 1 > ./results/%M%x%N%x%K%_w4a8.txt

set M=8
set N=4096
set K=4096
(for %%b in (%BITS%) do (
   .\bin\Release\test_any_wmma.exe %M% %N% %K% %%b %%b 1 > ./results/%M%x%N%x%K%_w%%ba%%b.txt
))
.\bin\Release\test_any_wmma.exe %M% %N% %K% 4 2 1 > ./results/%M%x%N%x%K%_w2a4.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 6 2 1 > ./results/%M%x%N%x%K%_w2a6.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 2 1 > ./results/%M%x%N%x%K%_w2a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 3 1 > ./results/%M%x%N%x%K%_w3a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 4 1 > ./results/%M%x%N%x%K%_w4a8.txt

set M=8
set N=8192
set K=1024
(for %%b in (%BITS%) do (
   .\bin\Release\test_any_wmma.exe %M% %N% %K% %%b %%b 1 > ./results/%M%x%N%x%K%_w%%ba%%b.txt
))
.\bin\Release\test_any_wmma.exe %M% %N% %K% 4 2 1 > ./results/%M%x%N%x%K%_w2a4.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 6 2 1 > ./results/%M%x%N%x%K%_w2a6.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 2 1 > ./results/%M%x%N%x%K%_w2a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 3 1 > ./results/%M%x%N%x%K%_w3a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 4 1 > ./results/%M%x%N%x%K%_w4a8.txt

set M=8
set N=11008
set K=4096
(for %%b in (%BITS%) do (
   .\bin\Release\test_any_wmma.exe %M% %N% %K% %%b %%b 1 > ./results/%M%x%N%x%K%_w%%ba%%b.txt
))
.\bin\Release\test_any_wmma.exe %M% %N% %K% 4 2 1 > ./results/%M%x%N%x%K%_w2a4.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 6 2 1 > ./results/%M%x%N%x%K%_w2a6.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 2 1 > ./results/%M%x%N%x%K%_w2a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 3 1 > ./results/%M%x%N%x%K%_w3a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 4 1 > ./results/%M%x%N%x%K%_w4a8.txt

set M=8
set N=5120
set K=5120
(for %%b in (%BITS%) do (
   .\bin\Release\test_any_wmma.exe %M% %N% %K% %%b %%b 1 > ./results/%M%x%N%x%K%_w%%ba%%b.txt
))
.\bin\Release\test_any_wmma.exe %M% %N% %K% 4 2 1 > ./results/%M%x%N%x%K%_w2a4.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 6 2 1 > ./results/%M%x%N%x%K%_w2a6.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 2 1 > ./results/%M%x%N%x%K%_w2a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 3 1 > ./results/%M%x%N%x%K%_w3a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 4 1 > ./results/%M%x%N%x%K%_w4a8.txt

set M=8
set N=4096
set K=11008
(for %%b in (%BITS%) do (
   .\bin\Release\test_any_wmma.exe %M% %N% %K% %%b %%b 1 > ./results/%M%x%N%x%K%_w%%ba%%b.txt
))
.\bin\Release\test_any_wmma.exe %M% %N% %K% 4 2 1 > ./results/%M%x%N%x%K%_w2a4.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 6 2 1 > ./results/%M%x%N%x%K%_w2a6.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 2 1 > ./results/%M%x%N%x%K%_w2a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 3 1 > ./results/%M%x%N%x%K%_w3a8.txt
.\bin\Release\test_any_wmma.exe %M% %N% %K% 8 4 1 > ./results/%M%x%N%x%K%_w4a8.txt