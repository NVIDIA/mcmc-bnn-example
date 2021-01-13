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

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "global_params.h"
#include "kernel_helpers.h"
#include "utils_cuda.h"
#include "utils_matrix.h"

namespace mcmc {

static constexpr int W_LEN[] = {
  FEAT_DIM[0] * FEAT_DIM[1],
  FEAT_DIM[1] * FEAT_DIM[2]
};
static constexpr int P_OFFSETS[] = {
  W_LEN[0],
  W_LEN[0] + FEAT_DIM[1],
  W_LEN[0] + FEAT_DIM[1] + W_LEN[1],
  NUM_PARAMS
};

static constexpr int N_THREADS = 256;
static constexpr int N_WARPS = N_THREADS / WarpSize;
// number of input rows processed by each warp set by a macro
static constexpr int N_BLK_SAMPLES = N_WARP_SAMPLES * N_WARPS;
static constexpr int N_BLKS = ceildiv(N_SAMPLES, N_BLK_SAMPLES);
static constexpr int IVECLEN = 2;  // this is for vectorized loads/stores for input
static constexpr int N_P_ALIGNED = ceildiv(NUM_PARAMS, IVECLEN) * IVECLEN;

HDI constexpr int aligned_nel(int nel) { return ceildiv(nel, IVECLEN) * IVECLEN; }
template <typename T>
HDI constexpr int aligned_size(int nel) { return aligned_nel(nel) * sizeof(T); }
template <typename T>
HDI constexpr int warp_red_size() {
  typedef cub::WarpReduce<T> WarpReduce;
  int size = sizeof(typename WarpReduce::TempStorage);
  return ceildiv(size, IVECLEN) * IVECLEN;
}
template <typename T>
HDI constexpr int full_smem_size() {
  constexpr int SMEM_IN = aligned_size<T>(N_BLK_SAMPLES * FEAT_DIM[0]);
  constexpr int SMEM_OUT = aligned_size<T>(N_BLK_SAMPLES * FEAT_DIM[2]);
  constexpr int SMEM_W = aligned_size<T>(NUM_PARAMS);
  constexpr int SMEM_G = aligned_size<T>(NUM_PARAMS);
  return SMEM_IN + SMEM_OUT + SMEM_W + SMEM_G + warp_red_size<T>();
}
template <typename T>
HDI constexpr int leapfrog_max_smem_size() {
  return best_max_smem_size<full_smem_size<T>()>();
}

template <typename T, int IDIM>
DI void load_inputs(T *s_x, const T *x, int block_row_offset) {
  int offset = block_row_offset * IDIM;
  #pragma unroll
  for (int i = threadIdx.x * IVECLEN; i < N_BLK_SAMPLES * IDIM;
       i += N_THREADS * IVECLEN) {
    T val[IVECLEN];
    auto idx = offset + i;
    if (idx < N_SAMPLES * IDIM) {
      ldg(val, x + idx);
    } else {
      #pragma unroll
      for (int j = 0; j < IVECLEN; ++j) {
        val[j] = lit<T>(0);
      }
    }
    sts(s_x + i, val);
  }
}

template <typename T, int ODIM>
DI void load_outputs(T *s_y, const T *y, int block_row_offset) {
  int offset = block_row_offset * ODIM;
  #pragma unroll
  for (int i = threadIdx.x; i < N_BLK_SAMPLES * ODIM; i += N_THREADS) {
    auto idx = offset + i;
    s_y[i] = idx < N_SAMPLES * ODIM ? y[idx] : lit<T>(0);
  }
}

template<typename T, int MOM_CORRECTION, bool OUTPUT_LOSS>
DI T apply_leapfrog_step(
    T param, T& momentum, T grad_prev, T eps, T& loss_cur, T& grad_cur) {
  // first update the momentum
  momentum += eps * grad_prev / lit<T>(MOM_CORRECTION);
  // update the parameters
  param += eps * momentum;
  // pre-calculate the prior loss and gradient for the current gradient
  if (OUTPUT_LOSS) {
    auto loss_p = param * lit<T>(1. / PRIOR_SCALE);
    loss_cur = -(loss_p * loss_p) * lit<T>(.5) - lit<T>(PRIOR_LOSS_NORMALIZATION);
  }
  grad_cur = param * lit<T>(PRIOR_SCALE_NEG_RECP2);
  return param;
}

template <typename T, int MOM_CORRECTION, bool OUTPUT_LOSS>
DI void load_weights_bias(
    T* s_g, T* s_w, T* weights_prev, T* weights_next, T* mom_prev, T* mom_next,
    T* grad_prev, T* grad_next, T* loss_v, T eps) {
  // Philosophy behind update/gradient calculation:
  // 1. we load the weights from previous iteration
  // 2. we load the gradient calculated in previous iteration in grad/loss field 1
  // 3. we load the previous momentum values
  // 3. we update the momentum and weights (take the leap-frog step)
  // 4. we compute and write out part of the current gradient and loss (prior loss on weights)
  // 5. we write out the updated weights (to be used in the next iteration)
  // 6. we write out the updated momentum (to be used in the next iteration)
  // 6. we over-write the previous gradient field with 0 (to be used for accumulation in next it)
  // 7. we will use the current gradient field for accumulation in this iteration (previously zeroed-out)
  T loss_p = lit<T>(0);
  T grad_p = lit<T>(0);
  for (int p = threadIdx.x; p < NUM_PARAMS; p += N_THREADS) {
    auto new_m = mom_prev[p];
    auto new_p = apply_leapfrog_step<T, MOM_CORRECTION, OUTPUT_LOSS>(
      weights_prev[p], new_m, grad_prev[p], eps, loss_p, grad_p);
    s_w[p] = new_p;
    // reset the gradient accumulators for this and add grad_p immediately
    s_g[p] = blockIdx.x == 0 ? grad_p : lit<T>(0);
    if (blockIdx.x == 0) {
      weights_next[p] = new_p;
      mom_next[p] = new_m;
      grad_next[p] = lit<T>(0);
      if (OUTPUT_LOSS) loss_v[N_SAMPLES + p] = loss_p;
    }
  }
}

template <bool OUTPUT_LOSS, typename T, int MOM_CORRECTION>
__global__ void leapfrog_step_kernel(
  const T* x, const T* y, T* weights_prev, T* weights_next,
  T* mom_prev, T* mom_next, T* grad_prev, T* grad_cur, T* grad_next, T* loss_v,
  T eps) {
  // Philosophy:
  // All of inputs/outputs/weights/biases are all stored in smem
  // All of the intermediate activations are stored in registers, spread across a warp
  // Once the output is loaded to smem, it will be immediately copied over to every
  // thread's registers.

  constexpr auto OS0 = P_OFFSETS[0];
  constexpr auto OS1 = P_OFFSETS[1];
  constexpr auto OS2 = P_OFFSETS[2];
  constexpr auto DIM0 = FEAT_DIM[0];
  constexpr auto DIM1 = FEAT_DIM[1];
  constexpr auto DIM2 = FEAT_DIM[2];

  extern __shared__ T s_all[];

  // inputs and outputs
  auto s_x = s_all;
  auto s_y = s_all + aligned_nel(N_BLK_SAMPLES * DIM0);

  // parameters
  auto s_w_full = s_y + aligned_nel(N_BLK_SAMPLES * DIM2);
  auto s_weight0 = s_w_full;
  auto s_bias0 = s_w_full + OS0;
  auto s_weight1 = s_w_full + OS1;

  // parameter gradients
  // at first all threads do a shared mem atomic to update block-wide local gradient
  // then at the end, everyone cooperatively will do global atomic to compute the final
  // gradient for the whole batch
  // a block-wide reduction could also work, but typical block reductions
  // would perform the in-between warp additions linearly anyway
  // (see cub BlockReduce, both raking and warp reductions do this)
  auto s_g_full = s_w_full + aligned_nel(NUM_PARAMS);
  auto s_wgrad0 = s_g_full;
  auto s_bgrad0 = s_g_full + OS0;
  auto s_wgrad1 = s_g_full + OS1;

  // storage and types for cub warp reduction
  typedef cub::WarpReduce<T> WarpReduce;
  typename WarpReduce::TempStorage red_storage = *((typename WarpReduce::TempStorage*)s_g_full + aligned_nel(NUM_PARAMS));

  // load input and reference output
  int block_row_offset = blockIdx.x * N_BLK_SAMPLES;
  load_inputs<T, DIM0>(s_x, x, block_row_offset);
  load_outputs<T, DIM2>(s_y, y, block_row_offset);
  // load weights/biases for all layers
  load_weights_bias<T, MOM_CORRECTION, OUTPUT_LOSS>(
    s_g_full, s_w_full, weights_prev, weights_next, mom_prev, mom_next,
    grad_prev, grad_next, loss_v, eps);
  __syncthreads();
  __prof_trigger(15);

  auto wid = threadIdx.x / WarpSize;
  int row_offset = wid * N_WARP_SAMPLES;

  // load layer 1 weights
  // in order to hide the LDS latency under the fwd computations for layer 0
  auto lid = lane_id();
  constexpr int N_OUT0 = ceildiv(DIM1, WarpSize);  // output spread across all threads in a warp
  T wt1[N_OUT0];
  #pragma unroll
  for (int j = 0; j < N_OUT0; ++j) {
    auto colid = lid + j * WarpSize;
    wt1[j] = colid < DIM1 ? s_weight1[colid] : lit<T>(0);
  }

  // fwd of layer 0
  T y0[N_WARP_SAMPLES][N_OUT0];
  #pragma unroll
  for (int j = 0; j < N_OUT0; ++j) {
    auto colid = lid + j * WarpSize;
    auto b = colid < DIM1 ? s_bias0[colid] : lit<T>(0);
    #pragma unroll
    for (int i = 0; i < N_WARP_SAMPLES; ++i) {
      y0[i][j] = b;
    }
  }
  #pragma unroll
  for (int k = 0; k < DIM0; k += IVECLEN) {
    #pragma unroll
    for (int j = 0; j < N_OUT0; ++j) {
      auto colid = lid + j * WarpSize;
      T wt[IVECLEN];
      #pragma unroll
      for (int p = 0; p < IVECLEN; ++p)
        wt[p] = colid < DIM1 ? s_weight0[(k + p) * DIM1 + colid] : lit<T>(0);
      #pragma unroll
      for (int i = 0; i < N_WARP_SAMPLES; ++i) {
        T in[IVECLEN];
        lds(in, s_x + (row_offset + i) * DIM0 + k);
        #pragma unroll
        for (int p = 0; p < IVECLEN; ++p) {
          y0[i][j] += wt[p] * in[p];
        }
      }
    }
  }
  tanh_t<T> afunc0;
  #pragma unroll
  for (int i = 0; i < N_WARP_SAMPLES; ++i) {
    #pragma unroll
    for (int j = 0; j < N_OUT0; ++j) {
      y0[i][j] = afunc0.fwd(y0[i][j]);
    }
  }

  // fwd of layer 1
  T y1[N_WARP_SAMPLES];
  #pragma unroll
  for (int i = 0; i < N_WARP_SAMPLES; ++i) {
    y1[i] = lit<T>(0);
  }
  #pragma unroll
  for (int j = 0; j < N_OUT0; ++j) {
    #pragma unroll
    for (int i = 0; i < N_WARP_SAMPLES; ++i) {
      y1[i] += wt1[j] * y0[i][j];   // the partial gemv
    }
  }
  #pragma unroll
  for (int i = 0; i < N_WARP_SAMPLES; ++i) {
    y1[i] = WarpReduce(red_storage).Sum(y1[i]);  // full-and-final gemv
  }

  // evaluate loss and its gradient and bias-grad for layer 1
  if (lid == 0) {
    T db1 = lit<T>(0);
    T b1 = s_w_full[OS2];
    #pragma unroll
    for (int i = 0; i < N_WARP_SAMPLES; ++i) {
      constexpr T scale_recp = lit<T>(1. / STUDENT_T_SCALE);
      constexpr T df = lit<T>(STUDENT_T_DF);
      constexpr T df_recp = lit<T>(1. / STUDENT_T_DF);
      constexpr T loss_norm = lit<T>(STUDENT_T_LOSS_NORMALIZATION);
      auto diff = (s_y[row_offset + i] - y1[i] - b1) * scale_recp;
      auto diff2 = diff * diff;
      int full_row_offset = block_row_offset + row_offset + i;
      if (OUTPUT_LOSS && full_row_offset < N_SAMPLES) {
        constexpr T l_scale = -lit<T>(.5) * (df + lit<T>(1));
        loss_v[full_row_offset] = l_scale * log1p(diff2 * df_recp) - loss_norm;
      }
      constexpr T g_scale = (df + lit<T>(1)) * scale_recp;
      auto grad = full_row_offset < N_SAMPLES ? diff * g_scale / (diff2 + df) : lit<T>(0);

      y1[i] = grad;
      db1 += grad;
    }
    // doing atomic on smem first is (almost) negligible perf-wise,
    // but makes kernel cleaner since we do a full smem->global atomic at end
    // grad is the gradient for bias1 because last layer has no activation
    atomicAdd(s_g_full + OS2, db1);
  }
  #pragma unroll
  for (int i = 0; i < N_WARP_SAMPLES; ++i) {
    y1[i] = __shfl_sync(0xffffffff, y1[i], 0);
  }
  // layer 1 wgrad, layer 1 dgrad + layer 0 activation bwd + layer 0 bias-grad
  #pragma unroll
  for (int j = 0; j < N_OUT0; ++j) {
    auto dw1 = lit<T>(0), db0 = lit<T>(0);
    auto colid = lid + j * WarpSize;
    #pragma unroll
    for (int i = 0; i < N_WARP_SAMPLES; ++i) {
      dw1 += y1[i] * y0[i][j];
      auto grad = afunc0.bwd(y0[i][j], wt1[j] * y1[i]);
      // don't need to check for colid here because if colid >= DIM1
      // wt1[j] will be 0 making the whole grad 0
      y0[i][j] = grad;
      db0 += grad;
    }
    if (colid < DIM1) {
      atomicAdd(s_wgrad1 + colid, dw1);
      atomicAdd(s_bgrad0 + colid, db0);
    }
  }

  // layer 0 wgrad
  // work distribution pattern:
  //  each lane in the warp computes wgrad across the whole column of 22x40 weight matrix
  T dw0[DIM0][N_OUT0];
  #pragma unroll
  for (int i = 0; i < DIM0; ++i) {
    #pragma unroll
    for (int j = 0; j < N_OUT0; ++j) {
      dw0[i][j] = lit<T>(0);
    }
  }
  #pragma unroll
  for (int k = 0; k < N_WARP_SAMPLES; ++k) {
    #pragma unroll
    for (int i = 0; i < DIM0; i += IVECLEN) {
      T in[IVECLEN];
      lds(in, s_x + (row_offset + k) * DIM0 + i);
      #pragma unroll
      for (int p = 0; p < IVECLEN; ++p) {
        #pragma unroll
        for (int j = 0; j < N_OUT0; ++j) {
          dw0[i + p][j] += in[p] * y0[k][j];
        }
      }
    }
  }
  #pragma unroll
  for (int i = 0; i < DIM0; ++i) {
    #pragma unroll
    for (int j = 0; j < N_OUT0; ++j) {
      auto colid = lid + j * WarpSize;
      if (colid < DIM1) {
        atomicAdd(s_wgrad0 + i * DIM1 + colid, dw0[i][j]);
      }
    }
  }

  __syncthreads();
  __prof_trigger(15);
  // use the entire threadblock to atomically write-out the local grads
  #pragma unroll
  for (int i = threadIdx.x; i < NUM_PARAMS; i += N_THREADS) {
    atomicAdd(grad_cur + i, s_g_full[i]);
  }
}

// should be exactly the same as net.do_swap in main
template <typename T>
void net_do_swap(
  DeviceMatrix<T>& grad_prev, DeviceMatrix<T>& grad_cur,
  DeviceMatrix<T>& grad_next, DeviceMatrix<T>& w_prev,
  DeviceMatrix<T>& w_next, DeviceMatrix<T>& mom_prev,
  DeviceMatrix<T>& mom_next) {

  w_prev.swap(w_next);
  mom_prev.swap(mom_next);
  grad_cur.swap(grad_next);
  grad_prev.swap(grad_next);
}

template <typename T, int LP_STEPS=1>
void mcmc_step(
  const DeviceMatrix<T>& x, const DeviceMatrix<T>& y,
  DeviceMatrix<T>& weights_prev, DeviceMatrix<T>& weights_next,
  DeviceMatrix<T>& mom_prev, DeviceMatrix<T>& mom_next,
  DeviceMatrix<T>& grad_prev, DeviceMatrix<T>& grad_cur,
  DeviceMatrix<T>& grad_next, DeviceMatrix<T>& loss_values,
  DeviceMatrix<int>* grid_sync_workspace, T eps, const cudaStream_t& stream) {

  static_assert(NUM_LAYERS == 2, "expected 2-layer MLP model");
  static_assert(LP_STEPS > 0, "LP_STEPS must be >0");
  ASSERT(grid_sync_workspace == nullptr, "Expecting no grid sync workspace");

  leapfrog_step_kernel<LP_STEPS == 1, T, 2><<<N_BLKS, N_THREADS, full_smem_size<T>(), stream>>>(
    x.ptr(), y.ptr(), weights_prev.ptr(), weights_next.ptr(), mom_prev.ptr(),
    mom_next.ptr(), grad_prev.ptr(), grad_cur.ptr(), grad_next.ptr(),
    loss_values.ptr(), eps);
  CUDA_CHECK(cudaPeekAtLastError());
  for (int i = 0; i < LP_STEPS - 2; ++i) {
    net_do_swap(grad_prev, grad_cur, grad_next, weights_prev, weights_next, mom_prev, mom_next);
    leapfrog_step_kernel<false, T, 1><<<N_BLKS, N_THREADS, full_smem_size<T>(), stream>>>(
      x.ptr(), y.ptr(), weights_prev.ptr(), weights_next.ptr(), mom_prev.ptr(),
      mom_next.ptr(), grad_prev.ptr(), grad_cur.ptr(), grad_next.ptr(),
      loss_values.ptr(), eps);
  }
  if (LP_STEPS > 1) {
    net_do_swap(grad_prev, grad_cur, grad_next, weights_prev, weights_next, mom_prev, mom_next);
    leapfrog_step_kernel<true, T, 1><<<N_BLKS, N_THREADS, full_smem_size<T>(), stream>>>(
      x.ptr(), y.ptr(), weights_prev.ptr(), weights_next.ptr(), mom_prev.ptr(),
      mom_next.ptr(), grad_prev.ptr(), grad_cur.ptr(), grad_next.ptr(),
      loss_values.ptr(), eps);
  }
}

template <typename T>
void set_dynamic_smem_max_size() {
  int smem_max_size = leapfrog_max_smem_size<T>();
  CUDA_CHECK(cudaFuncSetAttribute(leapfrog_step_kernel<true, T, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));
  CUDA_CHECK(cudaFuncSetAttribute(leapfrog_step_kernel<true, T, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));
  CUDA_CHECK(cudaFuncSetAttribute(leapfrog_step_kernel<false, T, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));
  CUDA_CHECK(cudaFuncSetAttribute(leapfrog_step_kernel<false, T, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));
}

}  // namespace mcmc
