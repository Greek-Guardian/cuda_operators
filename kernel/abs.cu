__global__ void abs_kernel(float* a,
                            int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
            i < n; i += gridDim.x * blockDim.x) {
        a[i] = fabs(a[i]);
    }
}

void launch_abs(float* a,
                 int n) {
    dim3 grid((n + 1023) / 1024);
    dim3 block(1024);
    abs_kernel<<<grid, block>>>(a, n);
}