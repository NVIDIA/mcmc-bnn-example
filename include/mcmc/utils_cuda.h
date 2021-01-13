/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include <chrono>
#include <cstdio>
#include <iostream>

#define ASSERT(check, fmt, ...)                                                \
  do {                                                                         \
    if (!(check)) {                                                            \
        char errMsg[2048];                                                     \
        std::snprintf(errMsg, sizeof(errMsg), fmt, ##__VA_ARGS__);             \
        fprintf(stderr, "[%s:%d] Expression \"%s\" failed: %s\n",              \
                __FILE__, __LINE__, #check, errMsg);                           \
        abort();                                                               \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status = call;                                                 \
    ASSERT(status == cudaSuccess, "CUDA call \"%s\" failed: %s", #call,        \
           cudaGetErrorString(status));                                        \
  } while (0)

#define CUDA_CHECK_NO_THROW(call)                                              \
  do {                                                                         \
    cudaError_t status = call;                                                 \
    if (status != cudaSuccess) {                                               \
      fprintf(                                                                 \
          stderr, "[%s:%d] CUDA call \"%s\" failed: %s ",                      \
          __FILE__, __LINE__, #call, cudaGetErrorString(status));              \
    }                                                                          \
  } while (0)

#define HDI __host__ __device__ inline
#define DI __device__ inline

template <int MAT_LD, bool row_major = true, typename T>
HDI constexpr T& mget(T* mat, int row, int col) {
  if (row_major)
    return mat[row * MAT_LD + col];
  return mat[col * MAT_LD + row];
}
template <int MAT_LD, bool row_major = true, typename T>
HDI constexpr const T& mget(T const* mat, int row, int col) {
  if (row_major)
    return mat[row * MAT_LD + col];
  return mat[col * MAT_LD + row];
}

template <typename T>
HDI constexpr T lit(const int x);
template <>
HDI constexpr float lit(const int x) {
  return float(x);
}
template <>
HDI constexpr double lit(const int x) {
  return double(x);
}

template <typename T>
HDI constexpr T lit(const double x);
template <>
HDI constexpr float lit(const double x) {
  return float(x);
}
template <>
HDI constexpr double lit(const double x) {
  return x;
}

template <typename IntType>
constexpr HDI IntType ceildiv(IntType a, IntType b) {
  return (a + b - 1) / b;
}

static constexpr int WarpSize = 32;

DI int lane_id() {
  int id;
  asm("mov.s32 %0, %laneid;" : "=r"(id));
  return id;
}

template <typename T>
DI T warp_all_reduce(T val) {
#pragma unroll
  for (int i = WarpSize / 2; i > 0; i >>= 1) {
    T tmp = __shfl_xor_sync(0xffffffffu, val, i, WarpSize);
    val += tmp;
  }
  return val;
}

DI void sts(float* addr, const float (&x)[1]) { *addr = x[0]; }
DI void sts(float* addr, const float (&x)[2]) {
  float2 v2 = make_float2(x[0], x[1]);
  auto* s2 = reinterpret_cast<float2*>(addr);
  *s2 = v2;
}
DI void sts(float* addr, const float (&x)[4]) {
  float4 v4 = make_float4(x[0], x[1], x[2], x[3]);
  auto* s4 = reinterpret_cast<float4*>(addr);
  *s4 = v4;
}

DI void sts(double* addr, const double (&x)[1]) { *addr = x[0]; }
DI void sts(double* addr, const double (&x)[2]) {
  double2 v2 = make_double2(x[0], x[1]);
  auto* s2 = reinterpret_cast<double2*>(addr);
  *s2 = v2;
}
DI void sts(double* addr, const double (&x)[4]) {
  double2 v41 = make_double2(x[0], x[1]);
  double2 v42 = make_double2(x[2], x[3]);
  auto* s4 = reinterpret_cast<double2*>(addr);
  *s4 = v41;
  *(s4 + 1) = v42;
}

DI void lds(float& x, float* addr) { x = *addr; }
DI void lds(float (&x)[1], float* addr) { x[0] = *addr; }
DI void lds(float (&x)[2], float* addr) {
  auto* s2 = reinterpret_cast<float2*>(addr);
  auto v2 = *s2;
  x[0] = v2.x;
  x[1] = v2.y;
}
DI void lds(float (&x)[4], float* addr) {
  auto* s4 = reinterpret_cast<float4*>(addr);
  auto v4 = *s4;
  x[0] = v4.x;
  x[1] = v4.y;
  x[2] = v4.z;
  x[3] = v4.w;
}
DI void lds(double& x, double* addr) { x = *addr; }
DI void lds(double (&x)[1], double* addr) { x[0] = *addr; }
DI void lds(double (&x)[2], double* addr) {
  auto* s2 = reinterpret_cast<double2*>(addr);
  auto v2 = *s2;
  x[0] = v2.x;
  x[1] = v2.y;
}
DI void lds(double (&x)[4], double* addr) {
  auto* s4 = reinterpret_cast<double2*>(addr);
  auto v41 = *s4;
  auto v42 = *(s4 + 1);
  x[0] = v41.x;
  x[1] = v41.y;
  x[2] = v42.x;
  x[3] = v42.y;
}

DI void ldg(float& x, const float* addr) { x = *addr; }
DI void ldg(float (&x)[1], const float* addr) { x[0] = *addr; }
DI void ldg(float (&x)[2], const float* addr) {
  auto* s2 = reinterpret_cast<const float2*>(addr);
  auto v2 = *s2;
  x[0] = v2.x;
  x[1] = v2.y;
}
DI void ldg(float (&x)[4], const float* addr) {
  auto* s4 = reinterpret_cast<const float4*>(addr);
  auto v4 = *s4;
  x[0] = v4.x;
  x[1] = v4.y;
  x[2] = v4.z;
  x[3] = v4.w;
}

DI void ldg(double& x, const double* addr) { x = *addr; }
DI void ldg(double (&x)[1], const double* addr) { x[0] = *addr; }
DI void ldg(double (&x)[2], const double* addr) {
  auto* s2 = reinterpret_cast<const double2*>(addr);
  auto v2 = *s2;
  x[0] = v2.x;
  x[1] = v2.y;
}
DI void ldg(double (&x)[4], const double* addr) {
  auto* s4 = reinterpret_cast<const double2*>(addr);
  auto v41 = *s4;
  auto v42 = *(s4 + 1);
  x[0] = v41.x;
  x[1] = v41.y;
  x[2] = v42.x;
  x[3] = v42.y;
}

template <int required_size>
HDI constexpr int best_max_smem_size() {
  if (required_size <= 0) return 0;
  if (required_size <= 8 * 1024) return 8 * 1024;
  if (required_size <= 16 * 1024) return 16 * 1024;
  if (required_size <= 32 * 1024) return 32 * 1024;
  if (required_size <= 64 * 1024) return 64 * 1024;
#if DEVICE_ARCH < 800
  constexpr int max_size = 96 * 1024;
  if (required_size <= max_size) return max_size;
  static_assert(required_size <= max_size, "requiring too much smem for <= Volta");
#elif DEVICE_ARCH == 800
  constexpr int max_size = 164 * 1024;
  if (required_size <= 100 * 1024) return 100 * 1024;
  if (required_size <= 132 * 1024) return 132 * 1024;
  if (required_size <= max_size) return max_size;
  static_assert(required_size <= max_size, "requiring too much smem for Ampere");
#endif
}
