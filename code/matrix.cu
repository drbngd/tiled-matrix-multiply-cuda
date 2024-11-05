#include <stdio.h>
#include "util.h"

#define A_WIDTH 1024
#define A_HEIGHT 1024
#define B_WIDTH 1024
#define B_HEIGHT 1024
#define C_WIDTH B_WIDTH
#define C_HEIGHT A_HEIGHT

#define BLOCK_SIZE 32
#define NUM_SUBS (A_WIDTH / BLOCK_SIZE)


// device-side arrays
__device__ float d_A[A_HEIGHT][A_WIDTH];
__device__ float d_B[B_HEIGHT][B_WIDTH];
__device__ float d_C[C_HEIGHT][C_WIDTH];

// host-side arrays
float h_A[A_HEIGHT][A_WIDTH];
float h_B[B_HEIGHT][B_WIDTH];
float h_C[C_HEIGHT][C_WIDTH];
float h_C_ref[C_HEIGHT][C_WIDTH];

int matrixMulTest(float C[C_HEIGHT][C_WIDTH], float Cref[C_HEIGHT][C_WIDTH]);

/*
    This is the CPU-based matrix multiply.
    It calculates output matrix C, from the input matrices A and B.
*/
void matrixMulCPU(float A[A_HEIGHT][A_WIDTH], float B[C_HEIGHT][C_WIDTH], float C[C_HEIGHT][C_WIDTH]) {
    int x, y, k;
    for (y = 0; y < C_HEIGHT; y++){
        for (x = 0; x < C_WIDTH; x++){
            C[y][x] = 0;
            for (k = 0; k < A_WIDTH; k++){
                C[y][x] += A[y][k] * B[k][x];
            }
        }
    }

}


/*
    This is a GPU-based matrix multiply.
    It calculates output matrix d_C, from the input matrices d_A and d_B.
*/
__global__ void matrixMulCUDA() {
    // TODO implement simple CUDA matrix multiply here
    // inputs: d_A, d_B (global variables)
    // output: d_C (global variable)
    // do not use shared memory
    // note the launch parameters: this kernel is called for each
    //         cell in the output matrix

    /* get x and y thread idx to iterate the matrices */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    /* perform the multiplication */
    for (int k = 0; k < A_WIDTH; k++) {
        d_C[x][y] += d_A[x][k] * d_B[k][y];
    }

}

/*
    This is a GPU-based matrix multiply.
    It calculates output matrix d_C, from the input matrices d_A and d_B.
    It uses shared memory.
*/
__global__ void matrixMulCUDATiled() {
    // TODO implement tiled CUDA matrix multiply here
    // inputs: d_A, d_B (global variables)
    // output: d_C (global variable)
    // use tiled shared memory as described in the assignment
    // note the launch parameters: this kernel is called for each
    //         cell in the output matrix

    /* statically allocate two share arrays of size */
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    /* get x and y thread idx to iterate the matrices */
    int x = threadIdx.x;
    int y = threadIdx.y;
    int col = blockIdx.x * blockDim.x + x;
    int row = blockIdx.y * blockDim.y + y;

    /* initialize the output matrix */
    float product = 0.0f;

    // Loop over the sub-matrices of A and B required to compute the element
    for (int m = 0; m < NUM_SUBS; ++m) {
        // Load the sub-matrix of A into shared memory
        s_A[y][x] = d_A[row][m * BLOCK_SIZE + x];
        // Load the sub-matrix of B into shared memory
        s_B[y][x] = d_B[m * BLOCK_SIZE + y][col];
        // Synchronize to make sure the sub-matrices are loaded
        __syncthreads();

        // Multiply the sub-matrices together
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            product += s_A[y][k] * s_B[k][x];
        }
        // Synchronize to make sure that the preceding computation is done before loading new sub-matrices
        __syncthreads();
    }

    // Write the result to the output matrix
    d_C[row][col] = product;

}


int main(int argc, char **argv) {
    unsigned int mem_size_A, mem_size_B, mem_size_C;
    unsigned int x, y;
    float msec;
    cudaEvent_t start, stop;
    int mode;

    if (argc != 2) {
        printf("Syntax: %s mode\n", argv[0]);
        printf("where mode is:\n");
        printf("\t0 - CPU\n"
               "\t1 - naive GPU\n"
               "\t2 - tiled shared memory GPU\n");
        return 1;
    }
    mode = atoi(argv[1]);

    if (A_WIDTH != B_HEIGHT){
        printf("Error: A_HEIGHT and B_WIDTH do not match\n");
        return 1;
    }

    mem_size_A = sizeof(float) * A_WIDTH * A_HEIGHT;
    mem_size_B = sizeof(float) * B_WIDTH * B_HEIGHT;
    mem_size_C = sizeof(float) * C_WIDTH * C_HEIGHT;

    // Initialise A
    for (y = 0; y < A_HEIGHT; y++)
        for (x = 0; x <A_WIDTH; x++)
            h_A[y][x] = (float)rand() / RAND_MAX;
    // Initialise B
    for (y = 0; y < B_HEIGHT; y++)
        for (x = 0; x <B_WIDTH; x++)
            h_B[y][x] = (float)rand() / RAND_MAX;

    // copy host memory to device
    if (mode > 0) {
        CHECK_ERROR(cudaMemcpyToSymbol(d_A, h_A, mem_size_A));
        CHECK_ERROR(cudaMemcpyToSymbol(d_B, h_B, mem_size_B));
    }

    // Start timing
    CHECK_ERROR(cudaEventCreate(&start));
    CHECK_ERROR(cudaEventCreate(&stop));
    CHECK_ERROR(cudaEventRecord(start));

    // Setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(C_WIDTH / BLOCK_SIZE, C_HEIGHT / BLOCK_SIZE);
    
    switch (mode) {
        case 0: printf("Running CPU version\n");
            matrixMulCPU(h_A, h_B, h_C);
            break;
        case 1: printf("Running naive GPU version\n");
            matrixMulCUDA<<<grid, threads>>>();
            check_launch("matrixMulCUDA");
            break;
        case 2: printf("Running tiled GPU version\n");
            matrixMulCUDATiled<<<grid, threads>>>();
            check_launch("matrixMulCUDATiled");
            break;
        default: printf("Unknown mode %d\n",mode);
            break;
    }
        
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&msec, start, stop);

    // Copy result from device to host
    if (mode > 0)
        CHECK_ERROR(cudaMemcpyFromSymbol(h_C, d_C, mem_size_C));

    // compare the GPU results against the CPU results
    if (mode > 0) {
        // Compute reference CPU version
        matrixMulCPU(h_A, h_B, h_C_ref);

        // Check for errors
        matrixMulTest(h_C, h_C_ref);
    }

    printf("Completed in %f msec\n", msec);
    return 0;
}

static const int maxUlps = 1000;

int matrixMulTest(float C[C_HEIGHT][C_WIDTH], float Cref[C_HEIGHT][C_WIDTH]) {
    int errors = 0;
    int y, x;

    for (y = 0; y < C_HEIGHT; y++){
        for (x = 0; x < C_WIDTH; x++){
            if (!AlmostEqual2sComplement(C[y][x], Cref[y][x], maxUlps)) {
                errors++;
                printf("Device item c[%d][%d] = %f does not match host result %f\n", y, x, C[y][x], Cref[y][x]);
                if (errors > 5) {
                    printf("Too many errors, aborting comparison\n");
                    return errors;
                }
            }
        }
    }
    if (errors)
        printf("%d errors found\n", errors);
    else
        printf("Test passed successfully\n");
    return errors;
}
