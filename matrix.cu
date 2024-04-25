#include <stdio.h>

#define N 8 // Matrix size

__global__ void matrixMul(int *a, int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int i = 0; i < n; ++i)
            sum += a[row * n + i] * b[i * n + col];
        c[row * n + col] = sum;
    }
}

int main() {
    int *a, *b, *c; // Host matrices
    int *d_a, *d_b, *d_c; // Device matrices

    // Allocate memory on the host
    a = (int *)malloc(N * N * sizeof(int));
    b = (int *)malloc(N * N * sizeof(int));
    c = (int *)malloc(N * N * sizeof(int));

    // Allocate memory on the device
    cudaMalloc(&d_a, N * N * sizeof(int));
    cudaMalloc(&d_b, N * N * sizeof(int));
    cudaMalloc(&d_c, N * N * sizeof(int));

    // Initialize matrices on the host
    for (int i = 0; i < N * N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Copy matrices from host to device
    cudaMemcpy(d_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Set grid and block dimensions
    dim3 threadsPerBlock(8, 8);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Perform matrix multiplication
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Output some elements of the resulting matrix
    printf("Resulting matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", c[i * N + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(a);
    free(b);
    free(c);

    return 0;
}
