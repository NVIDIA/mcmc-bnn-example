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

// MCMC parameters
static constexpr size_t SEED_INIT = 0;
static constexpr int BURN_IN_STEPS = 100;
static constexpr int SAMPLE_STEPS = 800;
static constexpr int LEAPFROG_STEPS = 1000;
static constexpr int OUTPUT_EVERY = 10;
static constexpr double STEP_SIZE_INIT = 0.01;
static constexpr double STEP_SIZE_TARGET_RATE = 0.2;
static constexpr double STEP_SIZE_DECREMENT = 0.1;
static constexpr double STEP_SIZE_INCREMENT = 0.01;
static constexpr int STEP_SIZE_ADAPT_STEPS = BURN_IN_STEPS;

// loss parameters
static constexpr double STUDENT_T_SCALE = 100.0;
static constexpr double STUDENT_T_DF = 2.2;
// this equates to (log(abs(scale)) + 0.5 * log(df) + 0.5 * log(PI) +
//  lgamma(0.5 * df) - lgamma(0.5 * (df + 1)))
//  where lgamma is log(factorial(x - 1)),
//  which is valid here because x - 1 is always positive
static constexpr double STUDENT_T_LOSS_NORMALIZATION = 5.63448313353184;
// final student_t loss is [with x the estimate of return]
// -0.5 * (df + 1) * log(1 + ((return - x) / scale) ^ 2 / df) - student_t_norm
static constexpr double PRIOR_SCALE = 200.0;
// -1 / (prior_scale ** 2)
static constexpr double PRIOR_SCALE_NEG_RECP2 = -0.000025;
// this equates to 0.5 * log(2 * pi) + log(prior_scale)
static constexpr double PRIOR_LOSS_NORMALIZATION = 6.21725589975271;
// final prior loss is sum[p] -0.5 * (p / prior_scale) ^ 2 - prior_loss_norm

// model parameters
static constexpr int NUM_LAYERS = 2;  // wrt the claim, I'm fusing the last 2 layers!
static constexpr int FEAT_DIM[] = {22, 40, 1};
static constexpr int NUM_PARAMS = FEAT_DIM[0] * FEAT_DIM[1] + FEAT_DIM[1] + FEAT_DIM[1] * FEAT_DIM[2] + FEAT_DIM[2];

// data parameters
static constexpr int N_SAMPLES = 58368;  // this is also assumed to be the batchsize
static constexpr int N_COLS = FEAT_DIM[0];  // that we choose for training from the input dataset
static constexpr int TOTAL_COLS = 149;  // in the input dataset
static constexpr int SKIP_COLS = 2;  // that we skip in the input dataset
static constexpr const char* DATA_FNAME = "returns_and_features_for_mcmc.txt";
