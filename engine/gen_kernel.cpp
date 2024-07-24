#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <unordered_set>
#include <string>
using namespace std;

int main(int argc, char **argv)
{
    if (argc < 5) {
        printf("Usage: ./gen_kernel X_BITS W_BITS SIGNED K_STAGE\n");
        return -1;
    }

    int x_bits = atoi(argv[1]);
    int w_bits = atoi(argv[2]);
    bool quant_sign = atoi(argv[3]) == 1;
    int k_stages = atoi(argv[4]);
    printf("////// W%dA%d %s\n", w_bits, x_bits, quant_sign ? "int" : "uint");
    constexpr int SHARED_MEM_PER_SM = 102400; // bytes
    constexpr int MAX_SHARED_MEM_PER_BLOCK_OPTIN = 101376; //bytes
    constexpr int REGS_PER_THREAD = 255;
    constexpr int MMA_M = 8;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 128;
    constexpr int WARP_K = MMA_K;
    constexpr int MAX_BLOCK_K = 256;
    unordered_set<string> st;
    for (int BLOCK_K = MMA_K; BLOCK_K <= MAX_BLOCK_K; BLOCK_K += 128) {
        for (int X_WARPS_NUMS = 1; X_WARPS_NUMS <= 32; X_WARPS_NUMS *= 2) {
            for (int W_WARPS_NUMS = 1; W_WARPS_NUMS <= 32; W_WARPS_NUMS *= 2) {
                if (X_WARPS_NUMS * W_WARPS_NUMS > 32) // out of warps in one block
                    continue;
                for (int WARP_M = MMA_M; WARP_M <= 64; WARP_M += MMA_M) {
                    for (int WARP_N = MMA_N; WARP_N <= 64; WARP_N += MMA_N) {
                        int WARP_M_TILES = WARP_M / MMA_M;
                        int WARP_N_TILES = WARP_N / MMA_N;
                        //if (WARP_M_TILES % 2 != 0 || WARP_N_TILES % 2 != 0)	// wrong accuracy existed
                        //    continue;
                        int REGS_A = WARP_M_TILES * MMA_M * MMA_K / 32 / 32;
                        int REGS_B = WARP_N_TILES * MMA_N * MMA_K / 32 / 32;
						int REGS_C = WARP_M_TILES * WARP_N_TILES * MMA_M * MMA_N / 32;
                        if (REGS_A + REGS_B + REGS_C > REGS_PER_THREAD) // out of regs in one block
                            continue;
                        if (X_WARPS_NUMS * WARP_M_TILES * MMA_M % x_bits != 0)
                            continue;
                        if (W_WARPS_NUMS * WARP_N_TILES * MMA_N % w_bits != 0)
                            continue;
                        int BLOCK_M = X_WARPS_NUMS * WARP_M_TILES * MMA_M / x_bits;
                        int BLOCK_N = W_WARPS_NUMS * WARP_N_TILES * MMA_N / w_bits;
                        //if (BLOCK_M >= 16)	// inefficient
                        //    continue;
                        if (BLOCK_M > MMA_M * x_bits)
                            continue;
                        if (BLOCK_N < 8 || BLOCK_N % 8 != 0 || BLOCK_N > MMA_K)
                            continue;


                        int SKEW = w_bits * BLOCK_N % 16 == 0 ? 8 : 0;
                        size_t input_buffer_size =
                            2 * BLOCK_M * BLOCK_K * x_bits / 8 +
                            2 * BLOCK_N * BLOCK_K * w_bits / 8;
                        size_t output_buffer_size =
                            (BLOCK_M * x_bits) * (BLOCK_N * w_bits + SKEW) * sizeof(int);
                        size_t shared_mem_size = max(input_buffer_size, output_buffer_size);	// bytes
                        if (shared_mem_size >= SHARED_MEM_PER_SM ||
                            shared_mem_size >=
                                MAX_SHARED_MEM_PER_BLOCK_OPTIN) // out of shared memory
                            continue;

						string key = to_string(BLOCK_M) + "," + to_string(BLOCK_N) + "," +
                                     to_string(BLOCK_K) + "," + to_string(WARP_M) + "," +
                                     to_string(WARP_N) + "," + to_string(WARP_K);
                        if (st.find(key) != st.end())	// alread exists
                            continue;
						
						st.insert(key);
                        printf("// cta<%d,%d,%d> warp<%d,%d,%d> mma<%d,%d,%d>   WARPS[%dx%d]\n",
                               BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N,
                               MMA_K, X_WARPS_NUMS, W_WARPS_NUMS);
                        for (int kThreadBlockStage = 2; kThreadBlockStage <= k_stages; ++kThreadBlockStage) {
                            size_t input_buffer_size =
                                kThreadBlockStage * BLOCK_M * BLOCK_K * x_bits / 8 +
                                kThreadBlockStage * BLOCK_N * BLOCK_K * w_bits / 8;
                            size_t output_buffer_size =
                                (BLOCK_M * x_bits) * (BLOCK_N * w_bits + SKEW) * sizeof(int);
                            size_t shared_mem_size =
                                max(input_buffer_size, output_buffer_size); // bytes
                            if (shared_mem_size >= SHARED_MEM_PER_SM ||
                                shared_mem_size >=
                                    MAX_SHARED_MEM_PER_BLOCK_OPTIN) // out of shared memory
                                continue;
                            printf(
                                "AQ_INSTANTIATE_FUN(AqBWMMA, %d, %d, %s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d);\n",
                                x_bits, w_bits, quant_sign ? "true" : "false", BLOCK_M, BLOCK_N,
                                BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, kThreadBlockStage);
						}
                        
                    }
                }
            }
        }
	}
    
    return 0;
}