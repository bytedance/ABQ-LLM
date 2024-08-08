@echo off

if exist results rd /s /q results
mkdir results

BITS=2 3 4 5 6 7 8

M=1
N=4096
K=4096
(for %%b in (%BITS%) do (
   ./bin/test_any_wmma ${M} ${N} ${K} %%b %%b 1 > ./results/${M}x${N}x${K}_w%%ba%%b.txt
))
./bin/test_any_wmma ${M} ${N} ${K} 4 2 1 > ./results/${M}x${N}x${K}_w2a4.txt
./bin/test_any_wmma ${M} ${N} ${K} 6 2 1 > ./results/${M}x${N}x${K}_w2a6.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 2 1 > ./results/${M}x${N}x${K}_w2a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 3 1 > ./results/${M}x${N}x${K}_w3a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 4 1 > ./results/${M}x${N}x${K}_w4a8.txt

M=1
N=8192
K=1024
(for %%b in (%BITS%) do (
   ./bin/test_any_wmma ${M} ${N} ${K} %%b %%b 1 > ./results/${M}x${N}x${K}_w%%ba%%b.txt
))
./bin/test_any_wmma ${M} ${N} ${K} 4 2 1 > ./results/${M}x${N}x${K}_w2a4.txt
./bin/test_any_wmma ${M} ${N} ${K} 6 2 1 > ./results/${M}x${N}x${K}_w2a6.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 2 1 > ./results/${M}x${N}x${K}_w2a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 3 1 > ./results/${M}x${N}x${K}_w3a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 4 1 > ./results/${M}x${N}x${K}_w4a8.txt

M=1
N=11008
K=4096
(for %%b in (%BITS%) do (
   ./bin/test_any_wmma ${M} ${N} ${K} %%b %%b 1 > ./results/${M}x${N}x${K}_w%%ba%%b.txt
))
./bin/test_any_wmma ${M} ${N} ${K} 4 2 1 > ./results/${M}x${N}x${K}_w2a4.txt
./bin/test_any_wmma ${M} ${N} ${K} 6 2 1 > ./results/${M}x${N}x${K}_w2a6.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 2 1 > ./results/${M}x${N}x${K}_w2a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 3 1 > ./results/${M}x${N}x${K}_w3a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 4 1 > ./results/${M}x${N}x${K}_w4a8.txt

M=1
N=5120
K=5120
(for %%b in (%BITS%) do (
   ./bin/test_any_wmma ${M} ${N} ${K} %%b %%b 1 > ./results/${M}x${N}x${K}_w%%ba%%b.txt
))
./bin/test_any_wmma ${M} ${N} ${K} 4 2 1 > ./results/${M}x${N}x${K}_w2a4.txt
./bin/test_any_wmma ${M} ${N} ${K} 6 2 1 > ./results/${M}x${N}x${K}_w2a6.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 2 1 > ./results/${M}x${N}x${K}_w2a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 3 1 > ./results/${M}x${N}x${K}_w3a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 4 1 > ./results/${M}x${N}x${K}_w4a8.txt

M=1
N=4096
K=11008
(for %%b in (%BITS%) do (
   ./bin/test_any_wmma ${M} ${N} ${K} %%b %%b 1 > ./results/${M}x${N}x${K}_w%%ba%%b.txt
))
./bin/test_any_wmma ${M} ${N} ${K} 4 2 1 > ./results/${M}x${N}x${K}_w2a4.txt
./bin/test_any_wmma ${M} ${N} ${K} 6 2 1 > ./results/${M}x${N}x${K}_w2a6.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 2 1 > ./results/${M}x${N}x${K}_w2a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 3 1 > ./results/${M}x${N}x${K}_w3a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 4 1 > ./results/${M}x${N}x${K}_w4a8.txt

M=4
N=4096
K=4096
(for %%b in (%BITS%) do (
   ./bin/test_any_wmma ${M} ${N} ${K} %%b %%b 1 > ./results/${M}x${N}x${K}_w%%ba%%b.txt
))
./bin/test_any_wmma ${M} ${N} ${K} 4 2 1 > ./results/${M}x${N}x${K}_w2a4.txt
./bin/test_any_wmma ${M} ${N} ${K} 6 2 1 > ./results/${M}x${N}x${K}_w2a6.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 2 1 > ./results/${M}x${N}x${K}_w2a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 3 1 > ./results/${M}x${N}x${K}_w3a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 4 1 > ./results/${M}x${N}x${K}_w4a8.txt

M=4
N=8192
K=1024
(for %%b in (%BITS%) do (
   ./bin/test_any_wmma ${M} ${N} ${K} %%b %%b 1 > ./results/${M}x${N}x${K}_w%%ba%%b.txt
))
./bin/test_any_wmma ${M} ${N} ${K} 4 2 1 > ./results/${M}x${N}x${K}_w2a4.txt
./bin/test_any_wmma ${M} ${N} ${K} 6 2 1 > ./results/${M}x${N}x${K}_w2a6.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 2 1 > ./results/${M}x${N}x${K}_w2a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 3 1 > ./results/${M}x${N}x${K}_w3a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 4 1 > ./results/${M}x${N}x${K}_w4a8.txt

M=4
N=11008
K=4096
(for %%b in (%BITS%) do (
   ./bin/test_any_wmma ${M} ${N} ${K} %%b %%b 1 > ./results/${M}x${N}x${K}_w%%ba%%b.txt
))
./bin/test_any_wmma ${M} ${N} ${K} 4 2 1 > ./results/${M}x${N}x${K}_w2a4.txt
./bin/test_any_wmma ${M} ${N} ${K} 6 2 1 > ./results/${M}x${N}x${K}_w2a6.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 2 1 > ./results/${M}x${N}x${K}_w2a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 3 1 > ./results/${M}x${N}x${K}_w3a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 4 1 > ./results/${M}x${N}x${K}_w4a8.txt

M=4
N=5120
K=5120
(for %%b in (%BITS%) do (
   ./bin/test_any_wmma ${M} ${N} ${K} %%b %%b 1 > ./results/${M}x${N}x${K}_w%%ba%%b.txt
))
./bin/test_any_wmma ${M} ${N} ${K} 4 2 1 > ./results/${M}x${N}x${K}_w2a4.txt
./bin/test_any_wmma ${M} ${N} ${K} 6 2 1 > ./results/${M}x${N}x${K}_w2a6.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 2 1 > ./results/${M}x${N}x${K}_w2a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 3 1 > ./results/${M}x${N}x${K}_w3a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 4 1 > ./results/${M}x${N}x${K}_w4a8.txt

M=4
N=4096
K=11008
(for %%b in (%BITS%) do (
   ./bin/test_any_wmma ${M} ${N} ${K} %%b %%b 1 > ./results/${M}x${N}x${K}_w%%ba%%b.txt
))
./bin/test_any_wmma ${M} ${N} ${K} 4 2 1 > ./results/${M}x${N}x${K}_w2a4.txt
./bin/test_any_wmma ${M} ${N} ${K} 6 2 1 > ./results/${M}x${N}x${K}_w2a6.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 2 1 > ./results/${M}x${N}x${K}_w2a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 3 1 > ./results/${M}x${N}x${K}_w3a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 4 1 > ./results/${M}x${N}x${K}_w4a8.txt

M=8
N=4096
K=4096
(for %%b in (%BITS%) do (
   ./bin/test_any_wmma ${M} ${N} ${K} %%b %%b 1 > ./results/${M}x${N}x${K}_w%%ba%%b.txt
))
./bin/test_any_wmma ${M} ${N} ${K} 4 2 1 > ./results/${M}x${N}x${K}_w2a4.txt
./bin/test_any_wmma ${M} ${N} ${K} 6 2 1 > ./results/${M}x${N}x${K}_w2a6.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 2 1 > ./results/${M}x${N}x${K}_w2a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 3 1 > ./results/${M}x${N}x${K}_w3a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 4 1 > ./results/${M}x${N}x${K}_w4a8.txt

M=8
N=8192
K=1024
(for %%b in (%BITS%) do (
   ./bin/test_any_wmma ${M} ${N} ${K} %%b %%b 1 > ./results/${M}x${N}x${K}_w%%ba%%b.txt
))
./bin/test_any_wmma ${M} ${N} ${K} 4 2 1 > ./results/${M}x${N}x${K}_w2a4.txt
./bin/test_any_wmma ${M} ${N} ${K} 6 2 1 > ./results/${M}x${N}x${K}_w2a6.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 2 1 > ./results/${M}x${N}x${K}_w2a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 3 1 > ./results/${M}x${N}x${K}_w3a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 4 1 > ./results/${M}x${N}x${K}_w4a8.txt

M=8
N=11008
K=4096
(for %%b in (%BITS%) do (
   ./bin/test_any_wmma ${M} ${N} ${K} %%b %%b 1 > ./results/${M}x${N}x${K}_w%%ba%%b.txt
))
./bin/test_any_wmma ${M} ${N} ${K} 4 2 1 > ./results/${M}x${N}x${K}_w2a4.txt
./bin/test_any_wmma ${M} ${N} ${K} 6 2 1 > ./results/${M}x${N}x${K}_w2a6.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 2 1 > ./results/${M}x${N}x${K}_w2a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 3 1 > ./results/${M}x${N}x${K}_w3a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 4 1 > ./results/${M}x${N}x${K}_w4a8.txt

M=8
N=5120
K=5120
(for %%b in (%BITS%) do (
   ./bin/test_any_wmma ${M} ${N} ${K} %%b %%b 1 > ./results/${M}x${N}x${K}_w%%ba%%b.txt
))
./bin/test_any_wmma ${M} ${N} ${K} 4 2 1 > ./results/${M}x${N}x${K}_w2a4.txt
./bin/test_any_wmma ${M} ${N} ${K} 6 2 1 > ./results/${M}x${N}x${K}_w2a6.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 2 1 > ./results/${M}x${N}x${K}_w2a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 3 1 > ./results/${M}x${N}x${K}_w3a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 4 1 > ./results/${M}x${N}x${K}_w4a8.txt

M=8
N=4096
K=11008
(for %%b in (%BITS%) do (
   ./bin/test_any_wmma ${M} ${N} ${K} %%b %%b 1 > ./results/${M}x${N}x${K}_w%%ba%%b.txt
))
./bin/test_any_wmma ${M} ${N} ${K} 4 2 1 > ./results/${M}x${N}x${K}_w2a4.txt
./bin/test_any_wmma ${M} ${N} ${K} 6 2 1 > ./results/${M}x${N}x${K}_w2a6.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 2 1 > ./results/${M}x${N}x${K}_w2a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 3 1 > ./results/${M}x${N}x${K}_w3a8.txt
./bin/test_any_wmma ${M} ${N} ${K} 8 4 1 > ./results/${M}x${N}x${K}_w4a8.txt