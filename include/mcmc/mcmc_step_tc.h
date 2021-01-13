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
#include <cooperative_groups.h>
#include <type_traits>
#include "global_params.h"
#include "kernel_helpers.h"
#include "utils_cuda.h"
#include "utils_matrix.h"

namespace cg = cooperative_groups;

namespace mcmc {

static constexpr int W_LEN[] = {
  FEAT_DIM[0] * FEAT_DIM[1],
  FEAT_DIM[1],
  FEAT_DIM[1] * FEAT_DIM[2],
  FEAT_DIM[2]
};
static constexpr int P_OFFSETS[] = {
  W_LEN[0],
  W_LEN[0] + W_LEN[1],
  W_LEN[0] + W_LEN[1] + W_LEN[2],
  NUM_PARAMS
};

static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 8;
static constexpr int MMA_REG_A = 4;
static constexpr int MMA_REG_B = 2;
static constexpr int MMA_REG_C = 4;
static constexpr int MMA_C_COLS = 2;
static constexpr int MMA_C_ROWS = 2;
static constexpr int MMA_C_GROUPS = 4;
DI constexpr int a_trow(int m_frag, int reg, int lid = 0) {
  return m_frag * MMA_M + lid / 4 + (reg % 2) * 8;
}
DI constexpr int a_tcol(int k_frag, int reg, int lid = 0) {
  return k_frag * MMA_K + lid % 4 + (reg / 2) * 4;
}
DI constexpr int b_trow(int k_frag, int reg, int lid = 0) {
  return k_frag * MMA_K + lid % 4 + reg * 4;
}
DI constexpr int b_tcol(int n_frag, int reg, int lid = 0) {
  return n_frag * MMA_N + lid / 4;
}
DI constexpr int c_trow(int m_frag, int reg, int lid = 0) {
  return m_frag * MMA_M + lid / 4 + (reg / 2) * 8;
}
DI constexpr int c_trow2flat_row(int m_frag, int trow, int lid = 0) {
  return m_frag * MMA_M + lid / 4 + trow * 8;
}
DI constexpr int c_tcol(int n_frag, int reg, int lid = 0) {
  return n_frag * MMA_N + (lid % 4) * 2 + reg % 2;
}
DI constexpr int c_tcol2flat_col(int n_frag, int tcol, int lid = 0) {
  return n_frag * MMA_N + (lid % 4) * 2 + tcol;
}
DI constexpr int c_reg2tcol(int reg) {
  return reg % 2;
}
DI constexpr int c_reg2trow(int reg) {
  return reg / 2;
}

template <typename T, int STRIDE = 1, int GROUP_SIZE = WarpSize>
DI T strided_group_warp_reduce(T val, uint32_t mask = 0xffffffffu) {
#pragma unroll
  for (int i = GROUP_SIZE / 2; i > STRIDE / 2; i >>= 1)
    val += __shfl_down_sync(mask, val, i, GROUP_SIZE);
  return val;
}

template <typename T, int STRIDE = 1>
DI T strided_warp_broadcast(T val) {
  #pragma unroll
  for (int i = STRIDE; i < WarpSize; i *= 2)
    val = __shfl_up_sync(0xffffffffu, val, i, WarpSize);
  return val;
}

template <typename T, int M, int N, int R>
DI void mma_c2a_frag_smem(T* s_tmp, T (&frags)[M][N][R], int lid, int wid) {
  // To turn the C fragment into the A fragment, we use groups of 4 threads.
  // Within each group, the following describes the exchange of information:
  // T0:c0 -> T0:a0 | T0:c1 -> T1:a0 | T0:c2 -> T0:a1 | T0:c3 -> T1:a1
  // T1:c0 -> T2:a0 | T1:c1 -> T3:a0 | T1:c2 -> T2:a1 | T1:c3 -> T3:a1
  // T2:c0 -> T0:a2 | T2:c1 -> T1:a2 | T2:c2 -> T0:a3 | T2:c3 -> T1:a3
  // T3:c0 -> T2:a2 | T3:c1 -> T3:a2 | T3:c2 -> T2:a3 | T3:c3 -> T3:a3
  // c0..3 are the registers before and a0..3 after the exchange

  // we exchange one fragment at a time (of size R), in each warp independently
  // right now, this is done through shared memory in a way that writes cause
  // bank conflicts and are not (necessarily) vectorized. The loading back to
  // the new format is coalesced and the compiler should be able to vectorize
  // as well since the register loop is compile-time constant.
  auto w_off = wid * R * WarpSize;
  auto tg_off = (lid / MMA_C_GROUPS) * MMA_C_GROUPS;
  auto t_rank = lid % MMA_C_GROUPS;
  #pragma unroll
  for (int m = 0; m < M; ++m) {
    #pragma unroll
    for (int n = 0; n < N; ++n) {
      #pragma unroll
      for (int r = 0; r < R; ++r) {
        auto r_new = (t_rank / 2) * 2 + r / 2;
        auto t_rank_new = (t_rank % 2) * 2 + r % 2;
        s_tmp[w_off + tg_off + r_new * WarpSize + t_rank_new] = frags[m][n][r];
      }
      __syncwarp();
      #pragma unroll
      for (int r = 0; r < R; ++r) {
        frags[m][n][r] = s_tmp[w_off + r * WarpSize + lid];
      }
      __syncwarp();
    }
  }
}

static constexpr int N_THREADS = 256;
static constexpr int N_WARPS = N_THREADS / WarpSize;
// number of input rows processed by each warp set by a macro
static constexpr int N_BLK_SAMPLES = N_WARP_SAMPLES * N_WARPS;
static constexpr int N_BLKS = ceildiv(N_SAMPLES, N_BLK_SAMPLES);
// this is Ampere specific (we basically want a persistent kernel/single wave)
static constexpr int N_SMS = 108;
static constexpr int MIN_BLKS_PER_SM = ceildiv(N_BLKS, N_SMS);
static constexpr int IVECLEN = 4;  // this is for vectorized loads/stores for input
static constexpr int N_G_RED = W_LEN[0] + (NUM_PARAMS - W_LEN[0]) * MMA_C_GROUPS;
static constexpr int N_P_ALIGNED = ceildiv(NUM_PARAMS, IVECLEN) * IVECLEN;

HDI constexpr int aligned_nel(int nel) { return ceildiv(nel, IVECLEN) * IVECLEN; }
template <typename T>
HDI constexpr int aligned_size(int nel) { return aligned_nel(nel) * sizeof(T); }
template <typename T>
HDI constexpr int full_smem_size() {
  constexpr int SMEM_IN = aligned_size<T>(N_BLK_SAMPLES * FEAT_DIM[0]);
  constexpr int SMEM_OUT = aligned_size<T>(N_BLK_SAMPLES * FEAT_DIM[2]);
  constexpr int SMEM_W = aligned_size<T>(NUM_PARAMS);
  constexpr int SMEM_M = aligned_size<T>(NUM_PARAMS);
  constexpr int SMEM_G_RED = aligned_size<T>(N_G_RED * N_WARPS);
  constexpr int SMEM_MMA_EXCH = aligned_size<T>(MMA_REG_C * N_THREADS);
  return SMEM_IN + SMEM_OUT + SMEM_W + SMEM_M + SMEM_G_RED + SMEM_MMA_EXCH;
}
template <typename T>
HDI constexpr int leapfrog_max_smem_size() {
  return best_max_smem_size<full_smem_size<T>()>();
}

template <typename T, int IDIM>
DI void load_inputs(T *s_x, const T *x, int block_sample_offset) {
  static_assert(IVECLEN < N_THREADS, "must have fewer vector elements than threads per block");
  int offset = block_sample_offset * IDIM;
  constexpr int CUT = ((N_BLK_SAMPLES * IDIM) / IVECLEN) * IVECLEN;
  #pragma unroll
  for (int i = threadIdx.x * IVECLEN; i < CUT; i += N_THREADS * IVECLEN) {
    T val[IVECLEN];
    auto idx = offset + i;
    ldg(val, x + idx);
    sts(s_x + i, val);
  }
  if (CUT < N_BLK_SAMPLES * IDIM && threadIdx.x < IVECLEN) {
    auto idx = offset + threadIdx.x + CUT;
    s_x[threadIdx.x + CUT] = idx < N_SAMPLES * IDIM ? x[idx] : lit<T>(0);
  }
}

template <typename T, int ODIM>
DI void load_outputs(T *s_y, const T *y, int block_sample_offset) {
  static_assert(IVECLEN < N_THREADS, "must have fewer vector elements than threads per block");
  int offset = block_sample_offset * ODIM;
  constexpr int CUT = ((N_BLK_SAMPLES * ODIM) / IVECLEN) * IVECLEN;
  #pragma unroll
  for (int i = threadIdx.x * IVECLEN; i < CUT; i += N_THREADS * IVECLEN) {
    T val[IVECLEN];
    auto idx = offset + i;
    ldg(val, y + idx);
    sts(s_y + i, val);
  }
  if (CUT < N_BLK_SAMPLES * ODIM && threadIdx.x < IVECLEN) {
    auto idx = offset + threadIdx.x + CUT;
    s_y[threadIdx.x + CUT] = idx < N_SAMPLES * ODIM ? y[idx] : lit<T>(0);
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

template <typename T, int MOM_CORRECTION, bool FINAL_OUTPUT>
DI void load_weights_bias(
    T* s_w, T* s_m, T* weights_prev, T* weights_next, T* mom_prev, T* mom_next,
    T* grad_prev, T* grad_cur, T* grad_next, T* loss_v, T eps) {
  static_assert(IVECLEN < N_THREADS, "must have fewer vector elements than threads per block");
  // Philosophy behind update/gradient calculation:
  // 1. we load the weights from previous iteration (can be from either global in initial leapfrog step or shared otherwise)
  // 2. we load the gradient calculated in previous iteration in grad/loss field 1 (always from global)
  // 3. we load the previous momentum values (similar to weights, from global or shared)
  // 3. we update the momentum and weights (take the leap-frog step)
  // 4. we compute and write out part of the current gradient and loss (prior loss on weights, always on global)
  // 5. we write out the updated weights (to be used in the next iteration, only if necessary on shared for next leapfrog step)
  // 6. we write out the updated momentum (to be used in the next iteration, only if necessary on shared for next leapfrog step)
  // 6. we over-write the previous gradient field with 0 (to be used for accumulation in next it, always on global)
  // 7. we will use the current gradient field for accumulation in this iteration (previously zeroed-out, always on global)
  constexpr int CUT = (NUM_PARAMS / IVECLEN) * IVECLEN;
  T loss_p[IVECLEN];
  T grad_p[IVECLEN];
  #pragma unroll
  for (int j = 0; j < IVECLEN; ++j) {
    loss_p[j] = lit<T>(0);
    grad_p[j] = lit<T>(0);
  }
  #pragma unroll
  for (int p = threadIdx.x * IVECLEN; p < CUT; p += N_THREADS * IVECLEN) {
    T new_m[IVECLEN];
    T w_p[IVECLEN];
    T g_p[IVECLEN];
    T new_p[IVECLEN];
    // mom_prev and weights_prev might be shared or global addresses
    ldg(new_m, mom_prev + p);
    ldg(w_p, weights_prev + p);
    ldg(g_p, grad_prev + p);
    #pragma unroll
    for (int j = 0; j < IVECLEN; ++j) {
      new_p[j] = apply_leapfrog_step<T, MOM_CORRECTION, FINAL_OUTPUT>(
        w_p[j], new_m[j], g_p[j], eps, loss_p[j], grad_p[j]);
    }
    sts(s_w + p, new_p);
    if (!FINAL_OUTPUT) {
      sts(s_m + p, new_m);
    }
    if (blockIdx.x == 0) {
      if (FINAL_OUTPUT) {
        // these are global addresses, sts works with that, too
        sts(weights_next + p, new_p);
        sts(mom_next + p, new_m);
      }
      T z[IVECLEN];
      #pragma unroll
      for (int j = 0; j < IVECLEN; ++j) {
        z[j] = lit<T>(0);
        if (FINAL_OUTPUT) loss_v[N_SAMPLES + p + j] = loss_p[j];
        atomicAdd(grad_cur + p + j, grad_p[j]);
      }
      sts(grad_next + p, z);
    }
  }
  if (CUT < NUM_PARAMS && threadIdx.x < IVECLEN) {
    auto p = CUT + threadIdx.x;
    auto new_m = p < NUM_PARAMS ? mom_prev[p] : lit<T>(0);
    auto w_p = p < NUM_PARAMS ? weights_prev[p] : lit<T>(0);
    auto g_p = p < NUM_PARAMS ? grad_prev[p] : lit<T>(0);
    auto new_p = apply_leapfrog_step<T, MOM_CORRECTION, FINAL_OUTPUT>(
      w_p, new_m, g_p, eps, loss_p[0], grad_p[0]);
    s_w[p] = new_p;
    if (!FINAL_OUTPUT) s_m[p] = new_m;
    if (blockIdx.x == 0) {
      if (FINAL_OUTPUT) {
        weights_next[p] = new_p;
        mom_next[p] = new_m;
        loss_v[N_SAMPLES + p] = loss_p[0];
      }
      atomicAdd(grad_cur + p, grad_p[0]);
      grad_next[p] = lit<T>(0);
    }
  }
}

template <typename T, bool OUTPUT_LOSS>
__device__ void leapfrog_step_single(
    T* s_x, T* s_y, T* s_weight0, T* s_bias0, T* s_weight1, T* s_bias1,
    T* loss_v, T* s_g_red, T* s_mma_exch, T* grad_cur, int block_sample_offset) { 
  // Philosophy:
  // We keep inputs/outputs/weights/biases and momentum in smem throughout all leapfrog steps
  // All of the intermediate activations are stored in registers, spread across a warp

  constexpr auto DIM0 = FEAT_DIM[0];
  constexpr auto DIM1 = FEAT_DIM[1];
  constexpr auto WLEN0 = W_LEN[0];
  constexpr auto G_OS_GROUPS = W_LEN[1] + W_LEN[2] + W_LEN[3];
  constexpr auto G_OS_B1 = WLEN0 + W_LEN[1] + W_LEN[2];
  constexpr auto G_OS_W1 = WLEN0 + W_LEN[1];
  constexpr auto G_OS_B0 = WLEN0;

  auto wid = threadIdx.x / WarpSize;
  int warp_sample_off = wid * N_WARP_SAMPLES;

  // load layer 1 weights
  // in order to hide the LDS latency under the fwd computations for layer 0
  auto lid = lane_id();
  // philosophy: in forward, we perform the following GEMM (per warp)
  // DIM1 * DIM0 x DIM0 * N_WARP_SAMPLES => DIM1 * N_WARP_SAMPLES
  // the C-rows thus refer to DIM1 and C-cols to N_WARP_SAMPLES
  // the M-dimension refers to DIM1, N to N_WARP_SAMPLES and K to DIM0
  constexpr int N_OUT0_FRAG = ceildiv(DIM1, MMA_M);
  constexpr int N_SAMPLE_FRAG = ceildiv(N_WARP_SAMPLES, MMA_N);
  constexpr int N_IN0_FRAG = ceildiv(DIM0, MMA_K);
  T wt1[N_OUT0_FRAG][MMA_C_ROWS];
  #pragma unroll
  for (int m = 0; m < N_OUT0_FRAG; ++m) {
    #pragma unroll
    for (int trow = 0; trow < MMA_C_ROWS; ++trow) {
      int c_row = c_trow2flat_row(m, trow, lid);
      wt1[m][trow] = c_row < DIM1 ? s_weight1[c_row] : lit<T>(0);
    }
  }

  // fwd of layer 0
  float y0_mma_c[N_OUT0_FRAG][N_SAMPLE_FRAG][MMA_REG_C];
  #pragma unroll
  for (int m = 0; m < N_OUT0_FRAG; ++m) {
    float bias[MMA_C_ROWS];
    #pragma unroll
    for (int trow = 0; trow < MMA_C_ROWS; ++trow) {
      int c_row = c_trow2flat_row(m, trow, lid);
      bias[trow] = c_row < DIM1 ? s_bias0[c_row] : 0.f;
    }
    #pragma unroll
    for (int n = 0; n < N_SAMPLE_FRAG; ++n) {
      #pragma unroll
      for (int r = 0; r < MMA_REG_C; ++r)
        y0_mma_c[m][n][r] = bias[c_reg2trow(r)];
    }
  }
  #pragma unroll
  for (int k = 0; k < N_IN0_FRAG; ++k) {
    #pragma unroll
    for (int m = 0; m < N_OUT0_FRAG; ++m) {
      uint32_t mma_a[MMA_REG_A];
      #pragma unroll
      for (int r = 0; r < MMA_REG_A; ++r) {
        int row = a_trow(m, r, lid);
        int col = a_tcol(k, r, lid);
        // we load weights in a transposed way/col-major here,
        // since for MMA we need DIM0 as leading dimension instead of DIM1
        float in = row < DIM1 && col < DIM0 ? s_weight0[col * DIM1 + row] : 0.f;
        asm volatile("mov.b32 %0, %1;" : "=r"(mma_a[r]) : "f"(in));
      }
      #pragma unroll
      for (int n = 0; n < N_SAMPLE_FRAG; ++n) {
        uint32_t mma_b[MMA_REG_B];
        #pragma unroll
        for (int r = 0; r < MMA_REG_B; ++r) {
          int row = b_trow(k, r, lid);
          int col = b_tcol(n, r, lid);
          // we need to load inputs transposed as well, since we need batch
          // as leading dimension instead of DIM0
          float in = row < DIM0 && col < N_WARP_SAMPLES ? s_x[(col + warp_sample_off) * DIM0 + row] : 0.f;
          asm volatile("mov.b32 %0, %1;" : "=r"(mma_b[r]) : "f"(in));
        }
        asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(y0_mma_c[m][n][0]), "+f"(y0_mma_c[m][n][1]), "+f"(y0_mma_c[m][n][2]), "+f"(y0_mma_c[m][n][3])
          : "r"(mma_a[0]), "r"(mma_a[1]), "r"(mma_a[2]), "r"(mma_a[3]),
            "r"(mma_b[0]), "r"(mma_b[1]));
      }
    }
  }
  tanh_t<T> afunc0;
  #pragma unroll
  for (int m = 0; m < N_OUT0_FRAG; ++m) {
    #pragma unroll
    for (int n = 0; n < N_SAMPLE_FRAG; ++n) {
      #pragma unroll
      for (int r = 0; r < MMA_REG_C; ++r)
        y0_mma_c[m][n][r] = afunc0.fwd(y0_mma_c[m][n][r]);
    }
  }

  // fwd of layer 1
  T y1[N_SAMPLE_FRAG][MMA_C_COLS];
  #pragma unroll
  for (int n = 0; n < N_SAMPLE_FRAG; ++n) {
    #pragma unroll
    for (int r = 0; r < MMA_C_COLS; ++r)
      y1[n][r] = lit<T>(0);
  }
  #pragma unroll
  for (int m = 0; m < N_OUT0_FRAG; ++m) {
    #pragma unroll
    for (int n = 0; n < N_SAMPLE_FRAG; ++n) {
      // the partial gemv
      #pragma unroll
      for (int r = 0; r < MMA_REG_C; ++r)
        y1[n][c_reg2tcol(r)] += wt1[m][c_reg2trow(r)] * y0_mma_c[m][n][r];
    }
  }
  #pragma unroll
  for (int n = 0; n < N_SAMPLE_FRAG; ++n) {
    #pragma unroll
    for (int r = 0; r < MMA_C_COLS; ++r) {
      y1[n][r] = strided_group_warp_reduce<T, MMA_C_GROUPS>(y1[n][r]);
    }
  }

  // evaluate loss and its gradient and bias-grad for layer 1
  auto dy1 = y1;
  if (lid < MMA_C_GROUPS) {
    T db1 = lit<T>(0);
    T b1 = *s_bias1;
    #pragma unroll
    for (int n = 0; n < N_SAMPLE_FRAG; ++n) {
      #pragma unroll
      for (int tcol = 0; tcol < MMA_C_COLS; ++tcol) {
        const int flat_col = c_tcol2flat_col(n, tcol, lid);
        constexpr T scale_recp = lit<T>(1. / STUDENT_T_SCALE);
        constexpr T df = lit<T>(STUDENT_T_DF);
        constexpr T df_recp = lit<T>(1. / STUDENT_T_DF);
        constexpr T loss_norm = lit<T>(STUDENT_T_LOSS_NORMALIZATION);
        T r_y = flat_col < N_WARP_SAMPLES ? s_y[flat_col + warp_sample_off] : lit<T>(0);
        auto diff = (r_y - y1[n][tcol] - b1) * scale_recp;
        auto diff2 = diff * diff;
        int full_sample_off = block_sample_offset + flat_col + warp_sample_off;
        if (OUTPUT_LOSS && flat_col < N_WARP_SAMPLES && full_sample_off < N_SAMPLES) {
          constexpr T l_scale = -lit<T>(.5) * (df + lit<T>(1));
          loss_v[full_sample_off] = l_scale * log1p(diff2 * df_recp) - loss_norm;
        }
        constexpr T g_scale = (df + lit<T>(1)) * scale_recp;
        auto grad = flat_col < N_WARP_SAMPLES ? diff * g_scale / (diff2 + df) : lit<T>(0);

        dy1[n][tcol] = grad;
        db1 += grad;
      }
    }
    // db1 = strided_group_warp_reduce<T, 1, MMA_C_GROUPS>(db1, 0xfu);
    // see comment in backward of bias0 / weight1
    s_g_red[wid * N_G_RED + G_OS_B1 + lid * G_OS_GROUPS] = db1;
  }
  // broadcast dy1 to all threads in same sample dimension on MMA_C
  #pragma unroll
  for (int n = 0; n < N_SAMPLE_FRAG; ++n) {
    #pragma unroll
    for (int r = 0; r < MMA_C_COLS; ++r) {
      dy1[n][r] = strided_warp_broadcast<T, MMA_C_GROUPS>(dy1[n][r]);
    }
  }

  // from here, we turn mma_c into mma_a for the backward matrix multiply
  // this assumes that MMA_REG_A == MMA_REG_C!
  auto& dy0_mma_a = y0_mma_c;
  #pragma unroll
  for (int m = 0; m < N_OUT0_FRAG; ++m) {
    T dw1[MMA_C_ROWS];
    T db0[MMA_C_ROWS];
    #pragma unroll
    for (int trow = 0; trow < MMA_C_ROWS; ++trow) {
      dw1[trow] = lit<T>(0);
      db0[trow] = lit<T>(0);
    }
    #pragma unroll
    for (int n = 0; n < N_SAMPLE_FRAG; ++n) {
      for (int r = 0; r < MMA_REG_C; ++r) {
        auto y0 = y0_mma_c[m][n][r];
        dw1[c_reg2trow(r)] += dy1[n][c_reg2tcol(r)] * y0;
        auto grad = afunc0.bwd(y0, wt1[m][c_reg2trow(r)] * dy1[n][c_reg2tcol(r)]);
        dy0_mma_a[m][n][r] = grad;
        db0[c_reg2trow(r)] += grad;
      }
    }
    #pragma unroll
    for (int trow = 0; trow < MMA_C_ROWS; ++trow) {
      // we write out everything into shared mem here then reduce at the very end
      auto flat_row = c_trow2flat_row(m, trow, lid);
      auto offset = flat_row + (lid % MMA_C_GROUPS) * G_OS_GROUPS;
      if (flat_row < DIM1) {
        s_g_red[wid * N_G_RED + G_OS_W1 + offset] = dw1[trow];
        s_g_red[wid * N_G_RED + G_OS_B0 + offset] = db0[trow];
      }
    }
  }

  // at this point, dy0_mma_a is still ordered like the C matrix (y0)
  // here, we turn it into the A matrix needed for backward
  // this happens in groups of 4
  // This is specific to Ampere and the TF32 16x8x8 mma instruction
  mma_c2a_frag_smem(s_mma_exch, dy0_mma_a, lid, wid);

  // layer 0 wgrad
  // work distribution pattern:
  // each lane in the warp computes wgrad across the whole column of 22x40 weight matrix
  T dw0[N_OUT0_FRAG][N_IN0_FRAG][MMA_REG_C];
  #pragma unroll
  for (int m = 0; m < N_OUT0_FRAG; ++m) {
    #pragma unroll
    for (int n = 0; n < N_IN0_FRAG; ++n) {
      #pragma unroll
      for (int r = 0; r < MMA_REG_C; ++r)
        dw0[m][n][r] = lit<T>(0);
    }
  }
  #pragma unroll
  for (int k = 0; k < N_SAMPLE_FRAG; ++k) {
    #pragma unroll
    for (int n = 0; n < N_IN0_FRAG; ++n) {
      uint32_t mma_b[MMA_REG_B];
      #pragma unroll
      for (int r = 0; r < MMA_REG_B; ++r) {
        int row = b_trow(k, r, lid);
        int col = b_tcol(n, r, lid);
        float in = row < N_WARP_SAMPLES && col < DIM0 ? s_x[(row + warp_sample_off) * DIM0 + col] : 0.f;
        asm volatile("mov.b32 %0, %1;" : "=r"(mma_b[r]) : "f"(in));
      }
      #pragma unroll
      for (int m = 0; m < N_OUT0_FRAG; ++m) {
        uint32_t mma_a[MMA_REG_A];
        #pragma unroll
        for (int r = 0; r < MMA_REG_A; ++r)
          asm volatile("mov.b32 %0, %1;" : "=r"(mma_a[r]) : "f"(dy0_mma_a[m][k][r]));
        asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(dw0[m][n][0]), "+f"(dw0[m][n][1]), "+f"(dw0[m][n][2]), "+f"(dw0[m][n][3])
          : "r"(mma_a[0]), "r"(mma_a[1]), "r"(mma_a[2]), "r"(mma_a[3]),
            "r"(mma_b[0]), "r"(mma_b[1]));
      }
    }
  }
  #pragma unroll
  for (int m = 0; m < N_OUT0_FRAG; ++m) {
    #pragma unroll
    for (int n = 0; n < N_IN0_FRAG; ++n) {
      #pragma unroll
      for (int r = 0; r < MMA_REG_C; ++r) {
        int row = c_trow(m, r, lid);
        int col = c_tcol(n, r, lid);
        if (row < DIM1 && col < DIM0) {
          s_g_red[wid * N_G_RED + col * DIM1 + row] = dw0[m][n][r];
        }
      }
    }
  }

  __syncthreads();

  // use the entire threadblock to reduce the values coming from all warps
  #pragma unroll
  for (int p = threadIdx.x; p < WLEN0; p += N_THREADS) {
    T g_out = s_g_red[p];
    #pragma unroll
    for (int w = 1; w < N_WARPS; ++w) {
      g_out += s_g_red[w * N_G_RED + p];
    }
    atomicAdd(grad_cur + p, g_out);
  }
  #pragma unroll
  for (int p = threadIdx.x + WLEN0; p < NUM_PARAMS; p += N_THREADS) {
    T g_out = lit<T>(0);
    #pragma unroll
    for (int w = 0; w < N_WARPS; ++w) {
      #pragma unroll
      for (int g = 0; g < MMA_C_GROUPS; ++g) {
        g_out += s_g_red[w * N_G_RED + g * G_OS_GROUPS + p];
      }
    }
    atomicAdd(grad_cur + p, g_out);
  }
}

template <typename T>
DI void sync_and_swap(int* workspace, T*& grad_prev, T*& grad_cur, T*& grad_next) {
  int delta = blockIdx.x == 0 ? int(gridDim.x) - 1 : -1;
  if (threadIdx.x == 0) {
    atomicAdd(workspace, delta);
    __threadfence();
    volatile int* addr = (volatile int*)workspace;
    while (*addr != 0);
  }
  __syncthreads();
  T* tmp = grad_prev;
  grad_prev = grad_cur;
  grad_cur = grad_next;
  grad_next = tmp;
}

template <typename T, int LP_STEPS>
__global__ __launch_bounds__(N_THREADS, MIN_BLKS_PER_SM) void mcmc_step_kernel(
  const T* x, const T* y, T* weights_prev, T* weights_next,
  T* mom_prev, T* mom_next, T* grad_prev, T* grad_cur, T* grad_next, T* loss_v,
  int* grid_sync_workspace, T eps) {
  constexpr auto OS0 = P_OFFSETS[0];
  constexpr auto OS1 = P_OFFSETS[1];
  constexpr auto OS2 = P_OFFSETS[2];
  constexpr auto DIM0 = FEAT_DIM[0];
  // constexpr auto DIM1 = FEAT_DIM[1];  // avoid unused warning
  constexpr auto DIM2 = FEAT_DIM[2];

  extern __shared__ T s_all[];

  // inputs and outputs
  auto s_x = s_all;
  auto s_y = s_all + aligned_nel(N_BLK_SAMPLES * DIM0);

  // parameters
  auto s_w_full = s_y + aligned_nel(N_BLK_SAMPLES * DIM2);
  auto s_mom = s_w_full + aligned_nel(NUM_PARAMS);

  // parameter gradients
  // Each warp updates its own section (size NUM_PARAMS) in this space.
  // Finally, we use the full block to perform the reduction across warps
  // and atomically reduce on global memory (w.r.t. other blocks)
  auto s_g_red = s_mom + aligned_nel(NUM_PARAMS);
  auto s_mma_exch = s_g_red + aligned_nel(N_G_RED * N_WARPS);

  // load input and reference output
  int block_sample_offset = blockIdx.x * N_BLK_SAMPLES;
  load_inputs<T, DIM0>(s_x, x, block_sample_offset);
  load_outputs<T, DIM2>(s_y, y, block_sample_offset);
  // load weights/biases for all layers
  load_weights_bias<T, 2, LP_STEPS == 1>(
    s_w_full, s_mom, weights_prev, weights_next, mom_prev, mom_next,
    grad_prev, grad_cur, grad_next, loss_v, eps);
  __syncthreads();

  static_assert(LP_STEPS > 0, "LP_STEPS must be >0");
  leapfrog_step_single<T, LP_STEPS == 1>(
    s_x, s_y, s_w_full, s_w_full + OS0, s_w_full + OS1, s_w_full + OS2,
    loss_v, s_g_red, s_mma_exch, grad_cur, block_sample_offset);
  for (int i = 0; i < LP_STEPS - 2; ++i) {
    sync_and_swap(grid_sync_workspace + i, grad_prev, grad_cur, grad_next);
    load_weights_bias<T, 1, false>(
      s_w_full, s_mom, s_w_full, weights_next, s_mom, mom_next,
      grad_prev, grad_cur, grad_next, loss_v, eps);
    __syncthreads();
    leapfrog_step_single<T, false>(
      s_x, s_y, s_w_full, s_w_full + OS0, s_w_full + OS1, s_w_full + OS2,
      loss_v, s_g_red, s_mma_exch, grad_cur, block_sample_offset);
  }
  if (LP_STEPS > 1) {
    sync_and_swap(grid_sync_workspace + LP_STEPS - 2, grad_prev, grad_cur, grad_next);
    load_weights_bias<T, 1, true>(
      s_w_full, s_mom, s_w_full, weights_next, s_mom, mom_next,
      grad_prev, grad_cur, grad_next, loss_v, eps);
    __syncthreads();
    leapfrog_step_single<T, true>(
      s_x, s_y, s_w_full, s_w_full + OS0, s_w_full + OS1, s_w_full + OS2,
      loss_v, s_g_red, s_mma_exch, grad_cur, block_sample_offset);
  }
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
  ASSERT(grid_sync_workspace != nullptr, "Expecting grid sync workspace");

  mcmc_step_kernel<T, LP_STEPS><<<N_BLKS, N_THREADS, full_smem_size<T>(), stream>>>(
    x.ptr(), y.ptr(), weights_prev.ptr(), weights_next.ptr(), mom_prev.ptr(),
    mom_next.ptr(), grad_prev.ptr(), grad_cur.ptr(), grad_next.ptr(),
    loss_values.ptr(), grid_sync_workspace->ptr(), eps);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
void set_dynamic_smem_max_size() {
  int smem_max_size = leapfrog_max_smem_size<T>();
  CUDA_CHECK(cudaFuncSetAttribute(mcmc_step_kernel<T, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));
  CUDA_CHECK(cudaFuncSetAttribute(mcmc_step_kernel<T, LEAPFROG_STEPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));
}

}  // namespace mcmc
