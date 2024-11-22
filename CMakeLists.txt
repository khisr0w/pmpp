cmake_minimum_required(VERSION 3.18)

project(PMPP LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 61)

set(EXE_NAME "run")

set(SOURCES
    src/main.cu
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build (Debug, Release)" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(BIN_DIR debug)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(BIN_DIR release)
endif()

add_executable(${EXE_NAME} ${SOURCES})
set_target_properties(${EXE_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
    COMPILE_FLAGS "-Xcompiler=\"/nologo /EHs /FC /Gm- /WX\""
    COMPILE_FLAGS_DEBUG "-g -G -Xcompiler=\"/Od /MTd /Z7 /Zo\""
    COMPILE_FLAGS_RELEASE "-O3 -Xcompiler=\"/MT\""
    LINK_FLAGS "-Xlinker=\"/nologo\""
    LINK_FLAGS_DEBUG "-Xlinker=\"/DEBUG:FULL\""
)

include_directories(${CMAKE_SOURCE_DIR}/include)
# target_link_directories(${EXE_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/lib)
# target_link_libraries(${EXE_NAME} PUBLIC glfw3.lib opengl32.lib user32.lib gdi32.lib shell32.lib)

add_custom_target(exec
    COMMAND cd .. && ${CMAKE_CURRENT_BINARY_DIR}/${BIN_DIR}/${EXE_NAME}
)
