__global__ void
cuda_color_to_grayscale(u8 *out, u8 *in, i32 width, i32 height) {
    u32 col = blockIdx.x*blockDim.x + threadIdx.x;
    u32 row = blockIdx.y*blockDim.y + threadIdx.y;

    if(col < width && row < height) {
        /* NOTE(Abid): Offset on the grayscale image output */
        u32 gray_offset = row*width + col;
        /* NOTE(Abid): Offset on the RGB image input, considering the color channels */
        u32 rgb_offset = gray_offset*3;

        /* NOTE(Abid): Channel-wise operation. */
        u8 r = in[rgb_offset];
        u8 g = in[rgb_offset + 1];
        u8 b = in[rgb_offset + 2];
        out[gray_offset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

internal void
color_to_grayscale_conversion(const char *file_name) {
    i32 width, height, num_channels;

    stbi_set_flip_vertically_on_load(false);
    u8 *src_img = stbi_load(file_name, &width, &height, &num_channels, 0);

    usize src_mem_size = width*height*num_channels;
    usize dest_mem_size = width*height;
    u8 *dest_img = (u8 *)malloc(dest_mem_size);

    u8 *src_img_d = NULL; u8 *dest_img_d = NULL;
    CUDA_ERR_CHECK(cudaMalloc((void **)&src_img_d, src_mem_size));
    CUDA_ERR_CHECK(cudaMalloc((void **)&dest_img_d, dest_mem_size));
    cudaMemcpy(src_img_d, src_img, src_mem_size, cudaMemcpyHostToDevice);

    dim3 grid_dim(ceil(width/16), ceil(height/16), 1);
    dim3 block_dim(16, 16, 1);
    cuda_color_to_grayscale<<<grid_dim, block_dim>>>(dest_img_d, src_img_d, width, height);
    cudaMemcpy(dest_img, dest_img_d, dest_mem_size, cudaMemcpyDeviceToHost);

    cudaFree(src_img_d);
    cudaFree(dest_img_d);

    stbi_write_jpg("assets/grayscale_out.jpg", width, height, 1, dest_img, 20);
}
