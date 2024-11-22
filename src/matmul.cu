internal inline void
print_matrix(fmatrix A) {
    printf("[\n");
    for(u64 row = 0; row < A.n; ++row) {
        printf("    ");
        for(u64 col = 0; col < A.m; ++col) {
            printf("%.3f, ", A.data[row*A.m + col]);
        }
        printf("\n");
    }
    printf("]\n");
}

internal inline void
print_vector(fvec A) {
    printf("[ ");
    for(u64 col = 0; col < A.n; ++col) {
        printf("%.3f, ", A.data[col]);
    }
    printf(" ]\n");
}

__global__ void
cuda_matmul_element_kernel(f32 *a, f32 *b, f32 *c, u32 *c_n_p, u32 *c_m_p, u32 *a_m_p) {
    /* NOTE(abid): A single thread computes 1 element of `c`. */
    u32 row = blockIdx.y*blockDim.y + threadIdx.y;
    u32 col = blockIdx.x*blockDim.x + threadIdx.x;

    /* NOTE(abid): a_m = b_n, c_n = a_n, c_m = b_m. */
    u32 c_n = *c_n_p;
    u32 c_m = *c_m_p;
    u32 a_m = *a_m_p;

    if(row < c_n && col < c_m) {
        f32 sink_value = c[row*c_m + col];
        for(u32 idx = 0; idx < a_m; ++idx) {
            sink_value += a[row*a_m + idx] * b[idx*c_m + col];
        }
        c[row*c_m + col] = sink_value;
    }
}
internal void
matmul_element_per_thread(fmatrix *A, fmatrix *B, fmatrix *C) {
    _assert(A->m == B->n, "shape mismatch between A and B");
    _assert(C->n == A->n, "shape mismatch between C and A");
    _assert(C->m == B->m, "shape mismatch between C and B");
    u64 A_size = A->n*A->m*sizeof(f32);
    u64 B_size = B->n*B->m*sizeof(f32);
    u64 C_size = C->n*C->m*sizeof(f32);

    f32 *a_d, *b_d, *c_d;
    CUDA_ERR_CHECK(cudaMalloc((void **)&a_d, A_size));
    CUDA_ERR_CHECK(cudaMalloc((void **)&b_d, B_size));
    CUDA_ERR_CHECK(cudaMalloc((void **)&c_d, C_size));
    CUDA_ERR_CHECK(cudaMemcpy(a_d, A->data, A_size, cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(b_d, B->data, B_size, cudaMemcpyHostToDevice));

    u32 *c_n, *c_m, *a_m;
    CUDA_ERR_CHECK(cudaMalloc((void **)&c_n, sizeof(u32)));
    CUDA_ERR_CHECK(cudaMalloc((void **)&c_m, sizeof(u32)));
    CUDA_ERR_CHECK(cudaMalloc((void **)&a_m, sizeof(u32)));
    CUDA_ERR_CHECK(cudaMemcpy(c_n, (void *)&C->n, sizeof(u32), cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(c_m, (void *)&C->m, sizeof(u32), cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(a_m, (void *)&A->m, sizeof(u32), cudaMemcpyHostToDevice));

    u32 max_dim = (C->n > C->m) ? C->n : C->m;
    u64 n_thread = (max_dim > 32) ? 32 : max_dim;
    dim3 block_dim(n_thread, n_thread, 1);
    /* WARNING(abid): For larger matrices, they might not fit into the grid.
     * TODO: Launch multiple grid in that case. */
    dim3 grid_dim(ceil(C->n/(f32)n_thread), ceil(C->m/(f32)n_thread), 1);
    cuda_matmul_element_kernel<<<grid_dim, block_dim>>>(a_d, b_d, c_d, c_n, c_m, a_m);
    CUDA_ERR_CHECK(cudaMemcpy(C->data, c_d, C_size, cudaMemcpyDeviceToHost));

    CUDA_ERR_CHECK(cudaFree(a_d));
    CUDA_ERR_CHECK(cudaFree(b_d));
    CUDA_ERR_CHECK(cudaFree(c_d));
    CUDA_ERR_CHECK(cudaFree(c_n));
    CUDA_ERR_CHECK(cudaFree(c_m));
    CUDA_ERR_CHECK(cudaFree(a_m));
}

__global__ void
cuda_matmul_row_kernel(f32 *a, f32 *b, f32 *c, u32 *c_n_p, u32 *c_m_p, u32 *a_m_p) {
    /* NOTE(abid): A single thread computes 1 row of `c`. */
    u32 row = blockIdx.x*blockDim.x + threadIdx.x;

    /* NOTE(abid): a_m = b_n, c_n = a_n, c_m = b_m. */
    u32 c_n = *c_n_p;
    u32 c_m = *c_m_p;
    u32 a_m = *a_m_p;

    if(row < c_n) {
        for(usize col = 0; col < c_m; ++col) {
            f32 sink_value = c[row*c_m + col];
            for(u32 idx = 0; idx < a_m; ++idx) {
                sink_value += a[row*a_m + idx] * b[idx*c_m + col];
            }
            c[row*c_m + col] = sink_value;
        }
    }
}
internal void
matmul_row_per_thread(fmatrix *A, fmatrix *B, fmatrix *C) {
    _assert(A->m == B->n, "shape mismatch between A and B");
    _assert(C->n == A->n, "shape mismatch between C and A");
    _assert(C->m == B->m, "shape mismatch between C and B");
    u64 A_size = A->n*A->m*sizeof(f32);
    u64 B_size = B->n*B->m*sizeof(f32);
    u64 C_size = C->n*C->m*sizeof(f32);

    f32 *a_d, *b_d, *c_d;
    CUDA_ERR_CHECK(cudaMalloc((void **)&a_d, A_size));
    CUDA_ERR_CHECK(cudaMalloc((void **)&b_d, B_size));
    CUDA_ERR_CHECK(cudaMalloc((void **)&c_d, C_size));
    CUDA_ERR_CHECK(cudaMemcpy(a_d, A->data, A_size, cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(b_d, B->data, B_size, cudaMemcpyHostToDevice));

    u32 *c_n, *c_m, *a_m;
    CUDA_ERR_CHECK(cudaMalloc((void **)&c_n, sizeof(u32)));
    CUDA_ERR_CHECK(cudaMalloc((void **)&c_m, sizeof(u32)));
    CUDA_ERR_CHECK(cudaMalloc((void **)&a_m, sizeof(u32)));
    CUDA_ERR_CHECK(cudaMemcpy(c_n, (void *)&C->n, sizeof(u32), cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(c_m, (void *)&C->m, sizeof(u32), cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(a_m, (void *)&A->m, sizeof(u32), cudaMemcpyHostToDevice));

    u64 n_thread = (C->n > 1024) ? 1024 : C->n;
    dim3 block_dim(n_thread);
    dim3 grid_dim(ceil(C->n/(f32)n_thread));
    cuda_matmul_row_kernel<<<grid_dim, block_dim>>>(a_d, b_d, c_d, c_n, c_m, a_m);
    CUDA_ERR_CHECK(cudaMemcpy(C->data, c_d, C_size, cudaMemcpyDeviceToHost));

    CUDA_ERR_CHECK(cudaFree(a_d));
    CUDA_ERR_CHECK(cudaFree(b_d));
    CUDA_ERR_CHECK(cudaFree(c_d));
    CUDA_ERR_CHECK(cudaFree(c_n));
    CUDA_ERR_CHECK(cudaFree(c_m));
    CUDA_ERR_CHECK(cudaFree(a_m));
}

__global__ void
cuda_matmul_col_kernel(f32 *a, f32 *b, f32 *c, u32 *c_n_p, u32 *c_m_p, u32 *a_m_p) {
    /* NOTE(abid): A single thread computes 1 column of `c`. */
    u32 col = blockIdx.x*blockDim.x + threadIdx.x;

    /* NOTE(abid): a_m = b_n, c_n = a_n, c_m = b_m. */
    u32 c_n = *c_n_p;
    u32 c_m = *c_m_p;
    u32 a_m = *a_m_p;

    if(col < c_m) {
        for(usize row = 0; row < c_n; ++row) {
            f32 sink_value = c[row*c_m + col];
            for(u32 idx = 0; idx < a_m; ++idx) {
                sink_value += a[row*a_m + idx] * b[idx*c_m + col];
            }
            c[row*c_m + col] = sink_value;
        }
    }
}
internal void
matmul_col_per_thread(fmatrix *A, fmatrix *B, fmatrix *C) {
    _assert(A->m == B->n, "shape mismatch between A and B");
    _assert(C->n == A->n, "shape mismatch between C and A");
    _assert(C->m == B->m, "shape mismatch between C and B");
    u64 A_size = A->n*A->m*sizeof(f32);
    u64 B_size = B->n*B->m*sizeof(f32);
    u64 C_size = C->n*C->m*sizeof(f32);

    f32 *a_d, *b_d, *c_d;
    CUDA_ERR_CHECK(cudaMalloc((void **)&a_d, A_size));
    CUDA_ERR_CHECK(cudaMalloc((void **)&b_d, B_size));
    CUDA_ERR_CHECK(cudaMalloc((void **)&c_d, C_size));
    CUDA_ERR_CHECK(cudaMemcpy(a_d, A->data, A_size, cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(b_d, B->data, B_size, cudaMemcpyHostToDevice));

    u32 *c_n, *c_m, *a_m;
    CUDA_ERR_CHECK(cudaMalloc((void **)&c_n, sizeof(u32)));
    CUDA_ERR_CHECK(cudaMalloc((void **)&c_m, sizeof(u32)));
    CUDA_ERR_CHECK(cudaMalloc((void **)&a_m, sizeof(u32)));
    CUDA_ERR_CHECK(cudaMemcpy(c_n, (void *)&C->n, sizeof(u32), cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(c_m, (void *)&C->m, sizeof(u32), cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(a_m, (void *)&A->m, sizeof(u32), cudaMemcpyHostToDevice));

    u64 n_thread = (C->m > 1024) ? 1024 : C->m;
    dim3 block_dim(n_thread);
    dim3 grid_dim(ceil(C->m/(f32)n_thread));
    cuda_matmul_col_kernel<<<grid_dim, block_dim>>>(a_d, b_d, c_d, c_n, c_m, a_m);
    CUDA_ERR_CHECK(cudaMemcpy(C->data, c_d, C_size, cudaMemcpyDeviceToHost));

    CUDA_ERR_CHECK(cudaFree(a_d));
    CUDA_ERR_CHECK(cudaFree(b_d));
    CUDA_ERR_CHECK(cudaFree(c_d));
    CUDA_ERR_CHECK(cudaFree(c_n));
    CUDA_ERR_CHECK(cudaFree(c_m));
    CUDA_ERR_CHECK(cudaFree(a_m));
}

__global__ void
cuda_mat_vec_kernel(f32 *W, f32 *x, f32 *result, u32 *result_n_p, u32 *x_n_p) {
    u32 row = blockIdx.x*blockDim.x + threadIdx.x;
    u32 result_n = *result_n_p;
    u32 x_n = *x_n_p;

    if(row < result_n) {
        f32 sink_value = result[row];
        for(u32 idx = 0; idx < x_n; ++idx) {
            sink_value += W[row*x_n + idx] * x[idx];
        }
        result[row] = sink_value;
    }
}
internal void
mat_vec(fmatrix *W, fvec *x, fvec *result) {
    _assert(W->n == result->n, "shape mismatch between W and `result` vector");
    _assert(W->m == x->n, "shape mismatch between W and `x` vector");

    u64 W_size = W->n*W->m*sizeof(f32);
    u64 x_size = x->n*sizeof(f32);
    u64 result_size = result->n*sizeof(f32);

    f32 *W_d, *x_d, *result_d;
    CUDA_ERR_CHECK(cudaMalloc((void **)&W_d, W_size));
    CUDA_ERR_CHECK(cudaMalloc((void **)&x_d, x_size));
    CUDA_ERR_CHECK(cudaMalloc((void **)&result_d, result_size));
    CUDA_ERR_CHECK(cudaMemcpy(W_d, W->data, W_size, cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(x_d, x->data, x_size, cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(result_d, result->data, result_size, cudaMemcpyHostToDevice));

    u32 *result_n, *x_n;
    CUDA_ERR_CHECK(cudaMalloc((void **)&result_n, sizeof(u32)));
    CUDA_ERR_CHECK(cudaMalloc((void **)&x_n, sizeof(u32)));
    CUDA_ERR_CHECK(cudaMemcpy(result_n, (void *)&result->n, sizeof(u32), cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(x_n, (void *)&x->n, sizeof(u32), cudaMemcpyHostToDevice));

    u64 n_thread = (result->n > 1024) ? 1024 : result->n;
    dim3 block_dim(n_thread);
    dim3 grid_dim(ceil(result->n/(f32)n_thread));
    cuda_mat_vec_kernel<<<grid_dim, block_dim>>>(W_d, x_d, result_d, result_n, x_n);
    CUDA_ERR_CHECK(cudaMemcpy(result->data, result_d, result_size, cudaMemcpyDeviceToHost));

    CUDA_ERR_CHECK(cudaFree(W_d));
    CUDA_ERR_CHECK(cudaFree(x_d));
    CUDA_ERR_CHECK(cudaFree(result_d));
    CUDA_ERR_CHECK(cudaFree(result_n));
    CUDA_ERR_CHECK(cudaFree(x_n));
}
