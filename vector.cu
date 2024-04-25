#include <stdio.h>

#define N 10 // Size of the vectors

__global__ void vectorAdd(int *a, int *b, int *c) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
        c[index] = a[index] + b[index];
}

int main() {
    int *a, *b, *c; // Host vectors
    int *d_a, *d_b, *d_c; // Device vectors

    // Allocate memory on the host
    a = (int *)malloc(N * sizeof(int));
    b = (int *)malloc(N * sizeof(int));
    c = (int *)malloc(N * sizeof(int));

    // Allocate memory on the device
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    // Initialize vectors on the host
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Copy vectors from host to device
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Perform vector addition
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Output some elements of the resulting vector
    printf("First 10 elements of the resulting vector:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");

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
