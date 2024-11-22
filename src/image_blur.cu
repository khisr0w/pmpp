__global__ void
cuda_image_blur_kernel(u8* output, u8*input, i32 width, i32 height, u32 num_channels, i32 blur_size) {
    u32 col = blockIdx.x*blockDim.x + threadIdx.x;
    u32 row = blockIdx.y*blockDim.y + threadIdx.y;
    usize pixel_offset = (row*width + col)*num_channels;

    if(col < width && row < height) {
        u32 pixel_values[3] = {0, 0, 0};
        u32 num_pixels = 0;

        for(i32 blur_row = -blur_size; blur_row < blur_size+1; ++blur_row) {
            for(i32 blur_col = -blur_size; blur_col < blur_size+1; ++blur_col) {
                i32 current_row = row + blur_row;
                i32 current_col = col + blur_col;
                if(current_row >= 0 && current_row < height && current_col >= 0 && current_col < width) {
                    usize current_offset = (current_row*width + current_col)*num_channels;
                    for(u32 channel = 0; channel < num_channels; ++channel) {
                        pixel_values[channel] += input[current_offset + channel];
                    }
                    ++num_pixels;
                }
            }
        }
        for(u32 channel = 0; channel < num_channels; ++channel) {
            output[pixel_offset + channel] = (u8)(pixel_values[channel]/num_pixels);
        }
    }
}

internal void
image_blur(const char *file_name, u32 blur_window = 6) {
    i32 width, height, num_channels;
    i32 blur_size = (i32)(blur_window/2);

    stbi_set_flip_vertically_on_load(false);
    u8 *src_img = stbi_load(file_name, &width, &height, &num_channels, 0);
    if(!src_img) {
        printf("Couldn't load file `%s`\n", file_name);
    }

    usize img_size = width*height*num_channels;
    u8 *dest_img = (u8 *)malloc(img_size);

    u8 *src_img_d = NULL, *dest_img_d = NULL;
    CUDA_ERR_CHECK(cudaMalloc((void **)&src_img_d, img_size));
    CUDA_ERR_CHECK(cudaMalloc((void **)&dest_img_d, img_size));
    cudaMemcpy(src_img_d, src_img, img_size, cudaMemcpyHostToDevice);

    dim3 grid_dim(ceil(width/16), ceil(height/16), 1);
    dim3 block_dim(16, 16, 1);
    cuda_image_blur_kernel<<<grid_dim, block_dim>>>(dest_img_d, src_img_d, width, height, num_channels, blur_size);
    cudaMemcpy(dest_img, dest_img_d, img_size, cudaMemcpyDeviceToHost);

    cudaFree(src_img_d);
    cudaFree(dest_img_d);

    stbi_write_jpg("./assets/blur_out.jpg", width, height, num_channels, dest_img, 20);
}

