#define CUDA_ERR_CHECK(Expr)                                                                 \
    do {                                                                                     \
        cudaError_t Error = Expr;                                                            \
        if(Error != cudaSuccess) {                                                           \
            printf("%s in %s at line: %d\n", cudaGetErrorString(Error), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                              \
        }                                                                                    \
    } while(0);

#define test(name, expr, should_break)                                              \
    do {                                                                            \
        fprintf(stdout, "\033[33mTesting\033[0m [" name "] ... ");                  \
        if(!(expr)) {                                                               \
            fprintf(stderr, "\033[31mfailed\033[0m (%s:%d)\n", __FILE__, __LINE__); \
            if(should_break) { *(i32 *)0 = 0; }                                     \
        } else fprintf(stdout, "\033[32mpassed\033[0m\n");                          \
    } while(0);

#define _assert(Expr, ErrorStr, ...) \
    if((Expr)) { } \
    else { \
        fprintf(stderr, "ASSERTION ERROR (%s:%d): " ErrorStr "\n", \
                __FILE__, __LINE__, ##__VA_ARGS__); \
        *(i32 *)0 = 0; \
        exit(EXIT_FAILURE); \
    }

