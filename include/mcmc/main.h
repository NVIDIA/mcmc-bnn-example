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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <thrust/random.h>
#include <thrust/random/normal_distribution.h>
#include "global_params.h"
#include "io.h"
#include "kernel_helpers.h"
#ifndef USE_TC
#include "mcmc_step_ffma.h"
#else
#include "mcmc_step_tc.h"
#endif
#include "utils_cuda.h"
#include "utils_matrix.h"

namespace mcmc {

template <typename T>
struct Network {
    Network(const cudaStream_t& init_stream) :
    // important to set the whole memory to 0 for proper initialization!
    block(7, N_P_ALIGNED, lit<T>(0), init_stream)
    , w_prev(1, N_P_ALIGNED, block.ptr())
    , mom_prev(1, N_P_ALIGNED, block.ptr() + N_P_ALIGNED * 1)
    , grad_prev(1, N_P_ALIGNED, block.ptr() + N_P_ALIGNED * 2)
    , w_next(1, N_P_ALIGNED, block.ptr() + N_P_ALIGNED * 3)
    , grad_cur(1, N_P_ALIGNED, block.ptr() + N_P_ALIGNED * 4)
    , grad_next(1, N_P_ALIGNED, block.ptr() + N_P_ALIGNED * 5)
    , mom_next(1, N_P_ALIGNED, block.ptr() + N_P_ALIGNED * 6)
    {}
    // this holds all params (such that they are in a single block)
    DeviceMatrix<T> block;
    // these are simply views on the block memory above
    DeviceMatrix<T> w_prev, w_next, mom_prev, mom_next, grad_prev, grad_cur, grad_next;

    void do_swap_g() {
        grad_cur.swap(grad_next);
        grad_prev.swap(grad_next);
    }

    void do_swap() {
        w_prev.swap(w_next);
        mom_prev.swap(mom_next);
        do_swap_g();
    }
};

template <typename T>
void output_final_loss(
        const DeviceMatrix<T>& data, const DeviceMatrix<T>& labels,
        Network<T>& net, DeviceMatrix<T>& final_samples,
        DeviceMatrix<T>& loss_values, DeviceMatrix<T>& loss_red_values,
        T* d_temp_storage, size_t tmp_bytes,
        DeviceMatrix<int>* grid_sync_workspace, const cudaStream_t& stream) {
    // evaluate final loss on saved samples
    double final_loss = 0., loss1 = 0., loss2 = 0.;
    for (int step = 0; step < SAMPLE_STEPS; ++step) {
        // set everything to 0, we're only interested in the loss
        CUDA_CHECK(cudaMemsetAsync(
            net.block.ptr(), 0, net.block.nbytes(), stream));
        // copy weights into prev weights. Those are used in leapfrog kernel
        CUDA_CHECK(cudaMemcpyAsync(
            net.w_prev.ptr(), final_samples.ptr() + step * NUM_PARAMS,
            NUM_PARAMS * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        // calculate loss using mcmc_step with single LP (we'll ignore gradients)
        mcmc_step<T, 1>(
            data, labels, net.w_prev, net.w_next, net.mom_prev, net.mom_next,
            net.grad_prev, net.grad_cur, net.grad_next, loss_values,
            grid_sync_workspace, lit<T>(0), stream);
        // get the final loss1 & loss2
        T step_l1, step_l2;
        get_loss_values(
            step_l1, step_l2, loss_values.ptr(), loss_red_values.ptr(),
            d_temp_storage, tmp_bytes, stream);
        // at this point, we have loss1 in step_l1 and loss2 in step_l2
        // we use a moving average to average over all steps
        double d_step = double(step);
        loss1 = (loss1 * d_step + double(step_l1)) / (d_step + 1);
        loss2 = (loss2 * d_step + double(step_l2)) / (d_step + 1);
        double d_loss = double(step_l1) + double(step_l2);
        final_loss = (final_loss * d_step + d_loss) / (d_step + 1);
    }
    std::cout << "Final loss1: " << loss1 << ", loss2: " << loss2
              << ", full loss: " << final_loss << std::endl;
}

template<typename T>
void main_loop() {
    // set max shared mem capacity according to our needs
    set_dynamic_smem_max_size<T>();
    std::cout << "Using " << N_WARP_SAMPLES << " rows/warp -> " << N_BLKS
              << " blocks with " << N_BLK_SAMPLES << " rows / block."
              << " Using " << leapfrog_max_smem_size<T>() << " max shared mem size "
              << "(fit the required " << full_smem_size<T>() << " smem)."
              << std::endl;

#ifdef USE_TC
    int max_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, mcmc_step_kernel<T, 1>, N_THREADS, full_smem_size<T>()));
    std::cout << "Expecting " << max_blocks_per_sm << " max blocks/SM with 1 LP_STEP" << std::endl;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, mcmc_step_kernel<T, LEAPFROG_STEPS>, N_THREADS, full_smem_size<T>()));
    std::cout << "Expecting " << max_blocks_per_sm << " max blocks/SM with " << LEAPFROG_STEPS << " LP_STEPS" << std::endl;
#endif

    // High-level description of the loop
    //  1. save the loss values for after all leapfrog steps
    //  2. initialize momentum (net.mom_prev) with randn and also save for after leapfrog steps
    //  3. init all weights (prev/next) to 0 or from previous step and
    //      save for after all leapfrog steps
    //  4. call mcmc_step which performs all leapfrog steps,
    //      switching between w_prev/w_next and round-robin between
    //      the gradient fields.
    //  5. after all leapfrog steps, calculate final momentum by taking half step
    //      (in TFP, they take a full step in the previous iteration, then a half step backwards)
    //  6. calculate log_accept_ratio based on initial and final values of
    //      momentum and loss (initial before leapfrog steps in same MCMC step)
    //  7. sample uniformly for the acceptance and update weights based on
    //      weights saved before leapfrog steps and new weights
    //  8. update step size (eps)
    //  9. save all new values for after next leapfrog steps,
    //      initialize momentum to randn and goto 4.
    cudaStream_t stream = cudaStreamPerThread;
    size_t rel_seed = 0;
    DeviceMatrix<T> data(N_SAMPLES, N_COLS);
    DeviceMatrix<T> labels(N_SAMPLES, FEAT_DIM[NUM_LAYERS]);
    read_data(data, labels);
    std::cout << "Data: " << data;
    std::cout << "Labels: " << labels;
    // normalize features
    eltWiseApply(stream, [] __device__(int i, T* d) {
        d[i] = d[i] * lit<T>(0.001);
    }, &data);

    Network<T> net(stream);
    DeviceMatrix<T> loss_values(1, N_SAMPLES + NUM_PARAMS);
    DeviceMatrix<T> loss_red_values(1, 3);
    DeviceMatrix<int>* grid_sync_workspace = nullptr;
#ifdef USE_TC
    DeviceMatrix<int> gsw(1, LEAPFROG_STEPS, 0, stream);
    grid_sync_workspace = &gsw;
#endif
    DeviceMatrix<T> loss_red_tmp(1, N_SAMPLES + NUM_PARAMS * 2);
    T* log_acpt_tmp_storage = nullptr;
    size_t log_acpt_tmp_bytes = 0;
    T* loss_red_tmp_storage = nullptr;
    size_t loss_red_tmp_bytes = 0;
    DeviceMatrix<T> prev_step_g(1, NUM_PARAMS);
    DeviceMatrix<T> prev_step_loss(1, N_SAMPLES + NUM_PARAMS);
    DeviceMatrix<T> prev_lp_mom(1, NUM_PARAMS);
    DeviceMatrix<T> prev_step_w(1, NUM_PARAMS, lit<T>(0), stream);
    DeviceMatrix<T> final_samples(SAMPLE_STEPS, NUM_PARAMS);
    // for the bootstrap results, everything should be 0:
    // we initialize the weights to 0 and we don't want to update yet

    CUDA_CHECK(cudaDeviceSynchronize());
    std::chrono::high_resolution_clock::time_point t_start, t_end;
    t_start = std::chrono::high_resolution_clock::now();

    T eps = lit<T>(STEP_SIZE_INIT);
    mcmc_step<T, 1>(
        data, labels, net.w_prev, net.w_next, net.mom_prev, net.mom_next,
        net.grad_prev, net.grad_cur, net.grad_next, loss_values,
        grid_sync_workspace, eps, stream);
    // save gradient & loss for later
    CUDA_CHECK(cudaMemcpyAsync(
        prev_step_g.ptr(), net.grad_cur.ptr(), prev_step_g.nbytes(),
        cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        prev_step_loss.ptr(), loss_values.ptr(), loss_values.nbytes(),
        cudaMemcpyDeviceToDevice, stream));

    for (int step = 0; step < BURN_IN_STEPS + SAMPLE_STEPS; step++) {
        // sample momentum from normal distribution and save for later
        // since this happens before swap, we write to mom_next
        eltWiseApply(stream, [rel_seed] __device__(int i, T* prev_mom, T* net_mom) {
            if (i >= NUM_PARAMS) return;
            thrust::default_random_engine rng;
            rng.discard(SEED_INIT + rel_seed + i);
            thrust::normal_distribution<T> dist_normal(lit<T>(0), lit<T>(1));
            auto v = dist_normal(rng);
            prev_mom[i] = v;
            net_mom[i] = v;
        }, &prev_lp_mom, &net.mom_next);
        rel_seed += prev_lp_mom.nel();

        net.do_swap();

        mcmc_step<T, LEAPFROG_STEPS>(
            data, labels, net.w_prev, net.w_next, net.mom_prev, net.mom_next,
            net.grad_prev, net.grad_cur, net.grad_next, loss_values,
            grid_sync_workspace, eps, stream);
#ifdef USE_TC
        // we need to simulate the swapping of grad pointers LEAPFROG_STEPS times
        for (int i = 0; i < (LEAPFROG_STEPS - 1) % 3; ++i) net.do_swap_g();
#endif
        auto h_log_acpt = log_acpt_prec<T, N_SAMPLES, NUM_PARAMS>(
            loss_red_values.ptr() + 2, prev_step_loss.ptr(), loss_values.ptr(),
            prev_lp_mom.ptr(), net.mom_next.ptr(), net.grad_cur.ptr(),
            loss_red_tmp.ptr(), log_acpt_tmp_storage, log_acpt_tmp_bytes,
            eps, stream);

        // determine if we are accepted/rejected and copy over the results
        std::default_random_engine rng;
        rng.seed(SEED_INIT + rel_seed);
        rel_seed++;
        std::uniform_real_distribution<T> dist_uniform(std::numeric_limits<T>::min(), lit<T>(1));
        T log_uniform = log(dist_uniform(rng));

        // copy the weights based on accept/reject decision
        if (log_uniform < h_log_acpt) {
            CUDA_CHECK(cudaMemcpyAsync(
                prev_step_w.ptr(), net.w_next.ptr(), prev_step_w.nbytes(),
                cudaMemcpyDeviceToDevice, stream));
            // gradients and loss are already "accepted" since they are in
            // grad_cur/loss_values. Copy over for end of next step
            CUDA_CHECK(cudaMemcpyAsync(
                prev_step_g.ptr(), net.grad_cur.ptr(),
                prev_step_g.nbytes(), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                prev_step_loss.ptr(), loss_values.ptr(),
                loss_values.nbytes(), cudaMemcpyDeviceToDevice, stream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(
                net.w_next.ptr(), prev_step_w.ptr(), prev_step_w.nbytes(),
                cudaMemcpyDeviceToDevice, stream));
            // we have to reject the new gradients and loss, too
            CUDA_CHECK(cudaMemcpyAsync(
                net.grad_cur.ptr(), prev_step_g.ptr(),
                prev_step_g.nbytes(), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                loss_values.ptr(), prev_step_loss.ptr(),
                loss_values.nbytes(), cudaMemcpyDeviceToDevice, stream));
        }
        // see make_simple_step_size_update_policy in tfp 0.8 hmc.py
        if (step < STEP_SIZE_ADAPT_STEPS) {
            // we omit log-sum-exp here because we only have a single value
            // this also means that in hmc.py, log_n == 0
            auto log_mean_acpt_r = min(h_log_acpt, lit<T>(0));
            auto dec_adj = -lit<T>(STEP_SIZE_DECREMENT) /
                (lit<T>(1) + lit<T>(STEP_SIZE_DECREMENT));
            auto inc_adj = lit<T>(STEP_SIZE_INCREMENT);
            auto adj = log_mean_acpt_r < lit<T>(log(STEP_SIZE_TARGET_RATE)) ? dec_adj : inc_adj;
            eps += eps * adj;
        }
        if (step >= BURN_IN_STEPS) {
            // copy the latest weights (state) into final samples
            int offset = (step - BURN_IN_STEPS) * NUM_PARAMS;
            CUDA_CHECK(cudaMemcpyAsync(
                final_samples.ptr() + offset, prev_step_w.ptr(),
                prev_step_w.nbytes(), cudaMemcpyDeviceToDevice, stream));
        }

        if ((step + 1) % OUTPUT_EVERY == 0) std::cout << "." << std::flush;

    }
    std::cout << std::endl;

    // no need to synchronize here, we have to sync the stream in order to
    // get the last log accept value on host, which means the last weights
    // are valid w.r.t. host
    t_end = std::chrono::high_resolution_clock::now();

    std::cout << "Final samples: " << final_samples;
    output_final_loss(
        data, labels, net, final_samples, loss_values, loss_red_values,
        loss_red_tmp_storage, loss_red_tmp_bytes, grid_sync_workspace, stream);

    auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_end - t_start).count();
    std::cout << "MCMC (" << BURN_IN_STEPS << " burn in, " << SAMPLE_STEPS
              << " sample steps, " << LEAPFROG_STEPS
              << " leap-frog steps) done in " 
              << t_ms << "ms. Expected time for 200 sequential passes: "
              << std::fixed << std::setprecision(3)
              << 200 * (double(t_ms) / 3600000) << "h. Good bye!" << std::endl;
}

}
