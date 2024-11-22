/*  +======| File Info |===============================================================+
    |                                                                                  |
    |     Subdirectory:  /src                                                          |
    |    Creation date:  11/21/2024 4:19:29 AM                                         |
    |    Last Modified:                                                                |
    |                                                                                  |
    +======================================| Copyright © Sayed Abid Hashimi |==========+  */

#if !defined(TYPES_H)

#include <stdint.h>

typedef uint8_t u8;
typedef int8_t i8;
typedef int32_t i32;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float f32;
typedef double f64;
typedef size_t usize;

#define internal static

struct fmatrix {
    f32 *data;
    u64 n;
    u64 m;
};

struct fvec {
    f32 *data;
    u64 n;
};

#define TYPES_H
#endif
