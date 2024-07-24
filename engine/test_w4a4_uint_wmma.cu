#include <iostream>

#include <cuda.h>
#include <cutlass/numeric_types.h>
#include <cutlass/cutlass.h>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "common/pack.h"
#include "wmma_any/wmma_any.h"

#define CUTLASS_CHECK(status)                                                   \
    {                                                                           \
        cutlass::Status error = status;                                         \
        if (error != cutlass::Status::kSuccess) {                               \
            std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                      << " at: " << __LINE__ << std::endl;                      \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    }

#define NUM_PROFILE 200

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = int32_t; // <- data type of accumulator
using ElementComputeEpilogue = int32_t; // <- data type of epilogue operations
using ElementInputA = cutlass::uint4b_t; // <- data type of elements in input matrix A
using ElementInputB = cutlass::uint4b_t; // <- data type of elements in input matrix B
using ElementOutput = int32_t; // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using ElementAccumulator = int32_t;
using ElementCompute = int32_t;

using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 128>, cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<8, 8, 32>,
    cutlass::epilogue::thread::LinearCombinationClamp<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator,
        ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

int run(int m, int n, int k)
{
    int length_m = m;
    int length_n = n;
    int length_k = k;
    constexpr int W_BIT = 4;
    constexpr int A_BIT = 4;
    uint64_t seed = 0x2019;

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

    // Initialize tensors using CUTLASS helper functions
    // Big-Endian
    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
        problem_size.mk()); // <- Create matrix A with dimensions m x k
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
        problem_size.kn()); // <- Create matrix B with dimensions k x n
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
        problem_size.mn()); // <- Create matrix C with dimensions m x n
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
        problem_size.mn()); // <- Create matrix D with dimensions m x n used to store output from
        // CUTLASS kernel
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
        problem_size.mn()); // <- Create matrix D with dimensions m x n used to store output from
        // reference kernel

    // Initialize in host memory
    cutlass::reference::host::TensorFillRandomUniform(tensor_a.host_view(), seed, 0, 15);

    cutlass::reference::host::TensorFillRandomUniform(tensor_b.host_view(), seed, 0, 15);

    cutlass::reference::host::TensorFill(tensor_c.host_view(), 0);
    // std::cout << "tensor_a:\n"
    //           << tensor_a.host_view() << std::endl;
    // std::cout << "tensor_b:\n"
    //           << tensor_b.host_view() << std::endl;

    // Copy data from host to GPU
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c.sync_device();
    tensor_d.sync_device();
    tensor_ref_d.sync_device();

    // Initialize alpha and beta for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // Split k dimension into 1 partitions
    int split_k_slices = 1;

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{
        problem_size, // <- problem size of matrix multiplication
        tensor_a.device_ref(), // <- reference to matrix A on device
        tensor_b.device_ref(), // <- reference to matrix B on device
        tensor_c.device_ref(), // <- reference to matrix C on device
        tensor_ref_d.device_ref(), // <- reference to matrix D on device
        { alpha, beta }, // <- tuple of alpha and beta
        split_k_slices
    }; // <- k-dimension split factor

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Initialize CUTLASS kernel with arguments and workspace pointer
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch initialized CUTLASS kernel
    for (int trial = 0; trial < NUM_PROFILE; trial++) {
        // printf("[%d]\n", trial);
        status = gemm_op();
        CUTLASS_CHECK(status);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUTLASS-GEMM (%d-bit). m: %6d, n: %6d, k: %6d,\t Time (ms): %.4f, TOPS: %4.4f\t\n", 4,
           m, n, k, milliseconds / NUM_PROFILE,
           static_cast<double>(NUM_PROFILE * (static_cast<double>(m) * n * k * 2) /
                               (milliseconds / 1000.)) /
               1e12);

    cudaDeviceSynchronize();
    // Initialize tensors for bmma
    // Little-Endian
    int *h_a = (int *)malloc(A_BIT * m * (k / 32) * sizeof(int));
    int *h_w = (int *)malloc(W_BIT * n * (k / 32) * sizeof(int));
    int *h_a_pack = (int *)malloc(A_BIT * m * (k / 32) * sizeof(int));
    int *h_w_pack = (int *)malloc(W_BIT * n * (k / 32) * sizeof(int));

    int *d_a;
    int *d_a_pack;
    int *d_w;
    int *d_w_pack;
    int *d_out;

    cudaMalloc(&d_a, A_BIT * m * (k / 32) * sizeof(int));
    cudaMalloc(&d_a_pack, A_BIT * m * (k / 32) * sizeof(int));
    cudaMalloc(&d_w, W_BIT * n * (k / 32) * sizeof(int));
    cudaMalloc(&d_w_pack, W_BIT * n * (k / 32) * sizeof(int));
    cudaMalloc(&d_out, m * n * sizeof(int));

    constexpr int PACK_A_NUM = 32 / A_BIT;
    constexpr int PACK_W_NUM = 32 / W_BIT;
    for (int i = 0; i < m * k / PACK_A_NUM; ++i) {
        h_a[i] = (((int *)tensor_a.host_data())[i]);
        // printf("h_a[i]=%x\n", ((int*)tensor_a.host_data())[i]);
        // printf("h_a2[i]=%x\n", h_a[i]);
    }
    for (int i = 0; i < n * k / PACK_W_NUM; ++i) {
        h_w[i] = (((int *)tensor_b.host_data())[i]);
        // printf("h_w[i]=%x\n", ((int*)tensor_b.host_data())[i]);
        // printf("h_w2[i]=%x\n", h_w[i]);
    }

    // Packing weights
    cudaMemcpy(d_w, h_w, W_BIT * n * (k / 32) * sizeof(int), cudaMemcpyHostToDevice);
    cudaError_t err = launch_pack4(d_w, d_w_pack, n, k, 4);
    if (err != cudaSuccess) {
        printf("Line %d: 'launch_pack4' failed: %s\n", __LINE__, cudaGetErrorString(err));
    }
    // cudaMemcpy(h_w_pack, d_w_pack, W_BIT * n * (k / 32) * sizeof(int),
    //            cudaMemcpyDeviceToHost);
    // for (int i = 0; i < W_BIT; i++) {
    //   for (int j = 0; j < n * (k / 32); ++j) {
    //     printf("bit:%d, w_pack[%d]:%x\n", i, j, h_w_pack[i * n * (k / 32) + j]);
    //   }
    // }
    // return 0;

    // Packing activations
    cudaMemcpy(d_a, h_a, A_BIT * m * (k / 32) * sizeof(int), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // Warm UP
    for (int trial = 0; trial < 10; trial++) {
        err = launch_pack4(d_a, d_a_pack, m, k, 4, stream);
        if (err != cudaSuccess) {
            printf("Line %d: 'launch_pack4' failed: %s\n", __LINE__, cudaGetErrorString(err));
        }
        AnyQuantParams params = { d_a_pack, d_w_pack, nullptr, d_out, false, m, n, k };
        err = launchWmmaAnyKernel<QuantType<4, 4, false>, ShapeBase<16, 16, 128>,
                                  ShapeBase<16, 32, 128>, ShapeBase<8, 8, 128>, 2>(params, stream);
        if (err != cudaSuccess) {
            printf("Line %d: 'launchWmmaAnyKernel' failed: %s\n", __LINE__,
                   cudaGetErrorString(err));
        }
    }
    cudaDeviceSynchronize();
    // Launch APMM kernel
    cudaEventRecord(start, stream);
    for (int trial = 0; trial < NUM_PROFILE; trial++) {
        err = launch_pack4(d_a, d_a_pack, m, k, 4, stream);
        if (err != cudaSuccess) {
            printf("Line %d: 'launch_pack4' failed: %s\n", __LINE__, cudaGetErrorString(err));
        }
        AnyQuantParams params = { d_a_pack, d_w_pack, nullptr, d_out, false, m, n, k };
        err = launchWmmaAnyKernel<QuantType<4, 4, false>, ShapeBase<16, 16, 128>,
                                  ShapeBase<16, 32, 128>, ShapeBase<8, 8, 128>, 2>(params, stream);
        if (err != cudaSuccess) {
            printf("Line %d: 'launchWmmaAnyKernel' failed: %s\n", __LINE__,
                   cudaGetErrorString(err));
        }
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("WMMA-AQGEMM (%d-bit). m: %6d, n: %6d, k: %6d,\t Time (ms): %.4f, TOPS: %4.4f\t\n", 4, m,
           n, k, milliseconds / NUM_PROFILE,
           static_cast<double>(NUM_PROFILE * (static_cast<double>(m) * n * k * 2) /
                               (milliseconds / 1000.)) /
               1e12);

    tensor_ref_d.sync_host();
    tensor_d.copy_in_device_to_host(d_out);
    // std::cout << "CUTLASS-GEMM result:\n"
    //           << tensor_ref_d.host_view() << std::endl;

    // std::cout << "APMM-GEMM result:\n" << tensor_d.host_view() << std::endl;

    // // Check if output from CUTLASS kernel and BMMA kernel are equal or not
    bool passed =
        cutlass::reference::host::TensorEquals(tensor_d.host_view(), tensor_ref_d.host_view());

    std::cout << (passed ? "Passed" : "Failed") << std::endl;

    return (passed ? 0 : -1);
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Usage: ./prog Dim\n");
        return -1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    bool notSupported = false;

    // Ampere Tensor Core operations exposed with mma.sync and ldmatrix are first available
    // in CUDA 11.0.
    //
    // CUTLASS must be compiled with CUDA 11.0 Toolkit to run these examples.
    if (!(__CUDACC_VER_MAJOR__ >= 11)) {
        std::cerr
            << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later."
            << std::endl;
        notSupported = true;
    }

    cudaDeviceProp props;

    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error)
                  << std::endl;
        return -1;
    }

    if (!((props.major * 10 + props.minor) >= 80)) {
        std::cerr
            << "Turing Tensor Core operations must be run on a machine with compute capability at least 80."
            << std::endl;
        notSupported = true;
    }

    if (notSupported) {
        // Returning zero so this test passes on older Toolkits. Its actions are no-op.
        return 0;
    }

    return run(m, n, k);
}