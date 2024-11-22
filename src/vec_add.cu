__global__ void 
vec_add_kernel(f32 *A, f32 *B, f32 *C, i32 N) {
    i32 Idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(Idx < N) C[Idx] = A[Idx] + B[Idx];
}

internal void
vec_add(f32 *A, f32 *B, f32 *C, i32 N) {
    usize Size = N * sizeof(f32);

    f32 *A_d, *B_d, *C_d;

    /* NOTE(abid): Allocate memory on device and copy data to device. */
    CUDA_ERR_CHECK(cudaMalloc((void **)&A_d, Size));
    CUDA_ERR_CHECK(cudaMalloc((void **)&B_d, Size));
    CUDA_ERR_CHECK(cudaMalloc((void **)&C_d, Size));

    /* NOTE(abid):
       cudaMemcpy(...)
           Params:
               void *: Destination
               void *: Source
               int   : Size
               Enum  : Type of copy
    */
    cudaMemcpy(A_d, A, Size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, Size, cudaMemcpyHostToDevice);

    /* NOTE(abid): Call kernel. */
    dim3 DimGrid(ceil(N/256.0), 1, 1);
    dim3 DimBlock(256, 1, 1);
    vec_add_kernel<<<DimGrid, DimBlock>>>(A_d, B_d, C_d, N);

    /* NOTE(abid): Copy result to host and free memory. */
    cudaMemcpy(C, C_d, Size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
