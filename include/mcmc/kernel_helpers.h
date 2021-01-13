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

#include <cub/cub.cuh>
#include "utils_cuda.h"

template<typename T>
struct tanh_t {};

template<>
struct tanh_t<float> {
  DI float fwd(float in) {
    float expResult, denom, sigm;
    // basically 2 * sigmoid(2x) - 1 = tanh(x). For sigmoid(2x), we use ex2:
    // e^(-2x) = 2^(-2x * 1 / ln(2)) = 2^(x * -2.88539008177792681472)
    asm volatile("{ ex2.approx.f32.ftz %0, %1;}\n" : "=f"(expResult) : "f"(in * -2.88539008177792681472f));
    asm volatile("{ add.f32.ftz %0, %1, %2;}\n" : "=f"(denom) : "f"(1.f), "f"(expResult));
    asm volatile("{ rcp.approx.f32 %0, %1;}\n" : "=f"(sigm) : "f"(denom));
    return fabs(in) < 0.0048176045529544353485f ? in : sigm * 2.f - 1.f;
  }
  DI float bwd(float out, float d_grad) { return (1.f - out * out) * d_grad; }
};

// kernels to initialize matrixes / other helpers
template<typename T>
__global__ void fill_mat_det_kernel(T* d, int rows, int cols)
{
    for (int i = threadIdx.y + blockDim.y * blockIdx.y ; i < rows ; i += blockDim.y * gridDim.y) {
        for (int j = threadIdx.x + blockDim.x * blockIdx.x ; j < cols ; j += blockDim.x * gridDim.x) {
            T v = ((((i * cols + j) * 32749) & 0x7FFF) + .5f) / 32768.0f;
            d[j + i * cols] = j % 2 == 0 ? -v : v;
        }
    }
}

template <typename T>
void fill_mat(T* ptr, int rows, int cols) {
    fill_mat_det_kernel<<<
        dim3(ceildiv(rows, 16), ceildiv(cols, 16), 1), dim3(16,16,1)>>>(
            ptr, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename F, typename... Ps>
__global__ void eltWiseApplyKernel(F f, const int N, Ps*... ptrs) {
  auto tidx   = threadIdx.x + blockDim.x * blockIdx.x;
  auto stride = blockDim.x * gridDim.x;
  for (; tidx < N; tidx += stride) f(tidx, ptrs...);
}

template <int TPB = 256, typename F, typename... Ms>
void eltWiseApply(const cudaStream_t stream, F&& f, Ms*... mats) {
  std::array<std::size_t, sizeof...(Ms)> sizes{{mats->nel()...}};
  int N = *std::max_element(std::begin(sizes), std::end(sizes));
  using T
   = std::decay_t<decltype(*std::get<0>(std::make_tuple(mats...))->ptr())>;
  int n_blocks = ceildiv(N, TPB);
  dim3 grid(n_blocks, 1);
  dim3 block(TPB, 1);
  eltWiseApplyKernel<<<grid, block, 0, stream>>>(std::forward<F>(f), N,
                                                 mats->ptr()...);
}

template <typename T, int TPB = 256>
void fill_mat_fixed(const cudaStream_t stream, T* p, int N, T val) {
    dim3 grid(ceildiv(N, TPB));
    dim3 block(TPB, 1);
    eltWiseApplyKernel<<<grid, block, 0, stream>>>(
        [val] __device__(int i, T* ptr) { 
            ptr[i] = val;
        },
        N, p);
}

template <typename T, int N_PER_THREAD, int TPB, int N_LOSS, int N_MOM>
__global__ void log_accept_kernel(
        T* log_acpt_out, const T* prev_loss_values,
        const T* new_loss_values, T* prev_mom, const T* new_mom,
        const T* grad_cur, const T eps) {
    constexpr int MOM_PER_THREAD = ceildiv(N_MOM, TPB);
    static_assert(MOM_PER_THREAD <= N_PER_THREAD, "we assume that block size * elements per thread covers all momentum values");
    typedef cub::BlockReduce<T, TPB> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T t_acc[N_PER_THREAD];
    const int block_offset = blockIdx.x * TPB * N_PER_THREAD;
    const int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < N_PER_THREAD; ++i) {
        int offset = block_offset + i * TPB + tid;
        auto new_loss = offset < N_LOSS ? new_loss_values[offset] : lit<T>(0);
        auto prev_loss = offset < N_LOSS ? prev_loss_values[offset] : lit<T>(0);
        t_acc[i] = new_loss - prev_loss;
    }
    if (blockIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < MOM_PER_THREAD; ++i) {
            int offset = i * TPB + tid;
            auto prev_m = offset < N_MOM ? prev_mom[offset] : lit<T>(0);
            auto new_m = offset < N_MOM ? new_mom[offset] : lit<T>(0);
            // calculate final momentum by taking a half step
            auto grad = offset < N_MOM ? grad_cur[offset] : lit<T>(0);
            new_m = new_m + eps * grad * lit<T>(.5);
            t_acc[i] += lit<T>(.5) * (prev_m * prev_m - new_m * new_m);
        }
    }
    T aggregate = BlockReduce(temp_storage).Sum(t_acc);
    if (tid == 0) atomicAdd(log_acpt_out, aggregate);
}

// we want as many threads per block as possible (better numerically)
template <typename T, int N_LOSS, int N_MOM, int N_PER_THREAD=4, int TPB=1024>
T compute_log_acpt(
        T* log_acpt_out, const T* prev_loss_values,
        const T* new_loss_values, T* prev_mom, const T* new_mom,
        const T* grad_cur, const T eps, const cudaStream_t& stream) {
    dim3 grid(ceildiv(N_LOSS, TPB * N_PER_THREAD));
    dim3 block(TPB, 1);
    // set the accumulator to 0
    CUDA_CHECK(cudaMemsetAsync(log_acpt_out, 0, sizeof(T), stream));
    log_accept_kernel<T, N_PER_THREAD, TPB, N_LOSS, N_MOM>
        <<<grid, block, 0, stream>>>(
            log_acpt_out, prev_loss_values, new_loss_values, prev_mom, new_mom,
            grad_cur, eps);
    T output;
    CUDA_CHECK(cudaMemcpyAsync(
        &output, log_acpt_out, sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return output;
}

template <typename T, int N_LOSS, int N_MOM, int TPB=128>
T log_acpt_prec(
        T* log_acpt_out, const T* prev_loss_values,
        const T* new_loss_values, T* prev_mom_values, const T* new_mom_values,
        const T* grad_cur_ptr, T* loss_tmp, T*&tmp_storage, size_t& tmp_bytes,
        const T eps, const cudaStream_t& stream) {
    // first calculate the individual loss differences and add momentum
    int n_blocks = ceildiv(N_LOSS + N_MOM, TPB);
    dim3 grid(n_blocks, 1);
    dim3 block(TPB, 1);
    eltWiseApplyKernel<<<grid, block, 0, stream>>>(
        [eps] __device__(int i, const T* prev_l, const T* new_l, T* prev_mom, const T* new_mom, const T* grad_cur, T* out) {
            out[i] = new_l[i] - prev_l[i];
            if (i < N_MOM) {
                auto prev_m = prev_mom[i];
                auto new_m = new_mom[i];
                // calculate final momentum by taking a half step
                new_m = new_m + eps * grad_cur[i] * lit<T>(.5);
                out[N_LOSS + N_MOM + i] = lit<T>(.5) * (prev_m * prev_m - new_m * new_m);
            }
        },
        N_LOSS + N_MOM, prev_loss_values, new_loss_values, prev_mom_values,
        new_mom_values, grad_cur_ptr, loss_tmp);
    if (tmp_storage == nullptr) {
        cub::DeviceReduce::Sum(
            tmp_storage, tmp_bytes, loss_tmp, log_acpt_out, N_LOSS + N_MOM * 2, stream);
        CUDA_CHECK(cudaMalloc(&tmp_storage, tmp_bytes));
    }
    cub::DeviceReduce::Sum(
        tmp_storage, tmp_bytes, loss_tmp, log_acpt_out, N_LOSS + N_MOM * 2, stream);
    T final_output;
    CUDA_CHECK(cudaMemcpyAsync(
        &final_output, log_acpt_out, sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return final_output;
}

template <typename T>
void get_loss_values(T& loss1, T& loss2, const T* full_loss_v,
                     T* d_out_values, T*& tmp_storage,
                     size_t& tmp_storage_bytes, const cudaStream_t& stream) {
    if (tmp_storage == nullptr) {
        size_t tmp_bytes_l1, tmp_bytes_l2;
        cub::DeviceReduce::Sum(
            tmp_storage, tmp_bytes_l1, full_loss_v, d_out_values,
            N_SAMPLES, stream);
        cub::DeviceReduce::Sum(
            tmp_storage, tmp_bytes_l2, full_loss_v + N_SAMPLES, d_out_values + 1,
            NUM_PARAMS, stream);
        tmp_storage_bytes = std::max(tmp_bytes_l1, tmp_bytes_l2);
        CUDA_CHECK(cudaMalloc(&tmp_storage, tmp_storage_bytes));
    }
    cub::DeviceReduce::Sum(
        tmp_storage, tmp_storage_bytes, full_loss_v,
        d_out_values, N_SAMPLES, stream);
    cub::DeviceReduce::Sum(
        tmp_storage, tmp_storage_bytes, full_loss_v + N_SAMPLES,
        d_out_values + 1, NUM_PARAMS, stream);
    CUDA_CHECK(cudaMemcpyAsync(
        &loss1, d_out_values, sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &loss2, d_out_values + 1, sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
