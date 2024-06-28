// #include <iostream>

__global__ void matmul_kernel(float* C,
                            const float* A,
                            const float* B,
                            int64_t N,
                            int64_t K,
                            int64_t M) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N*M; i += gridDim.x * blockDim.x) {
        C[i] = 0;
        int A_start = (i / M)*K;
        int B_bias = i % M;
        for(int j = 0; j < K; j++) {
            C[i] += A[A_start+j] * B[j*M+B_bias];
        }
    }
}

void launch_matmul(float *C,
                const float *A,
                const float *B,
                int64_t N,
                int64_t K,
                int64_t M) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0 );
    dim3 grid( N*M<prop.maxGridSize[0] ? N*M : prop.maxGridSize[0] );
    dim3 block( prop.maxThreadsPerBlock );
    matmul_kernel<<<grid, block>>>(C, A, B, N, K, M);
}