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

#include <sstream>
#include <string>
#include "global_params.h"
#include "utils_cuda.h"
#include "utils_matrix.h"

namespace mcmc {
template<typename T>
struct scanf_format {};
template<>
struct scanf_format<float> {static constexpr auto value = "%f";};
template<>
struct scanf_format<double> {static constexpr auto value = "%lf";};

static constexpr int SPLIT = SKIP_COLS + FEAT_DIM[NUM_LAYERS];

template<typename T>
void read_data(DeviceMatrix<T> &x_out, DeviceMatrix<T> &y_out) {
    int x_rows = int(x_out.rows()), x_cols = int(x_out.cols());
    ASSERT(x_rows == N_SAMPLES && x_cols == N_COLS,
           "Expected data matrix of size %dx%d, got %dx%d",
           N_SAMPLES, N_COLS, x_rows, x_cols);
    int y_rows = int(y_out.rows()), y_cols = int(y_out.cols());
    ASSERT(y_rows == N_SAMPLES && y_cols == FEAT_DIM[NUM_LAYERS],
           "Expected label matrix of size %dx%d, got %dx%d",
           N_SAMPLES, FEAT_DIM[NUM_LAYERS], y_rows, y_cols);
    // read the dataset
    printf("Reading dataset...\n");
    std::vector<T> x(N_SAMPLES * N_COLS), y(N_SAMPLES * FEAT_DIM[NUM_LAYERS]);
    const char* scan_format = scanf_format<T>::value;
    char buff[256];
    FILE* fp = fopen(DATA_FNAME, "r");
    ASSERT(fp != nullptr, "Error: failed to read file '%s'!", DATA_FNAME);
    // ignore first line completely
    for (int j = 0; j < TOTAL_COLS; ++j) {
        if (fscanf(fp, "%s", buff) == EOF) break;
    }
    for (int i = 0; i < N_SAMPLES; ++i) {
        for (int j = 0; j < TOTAL_COLS; ++j) {
        // ignore first 2 columns and the rest of the columns after the first 25
        if (j < SKIP_COLS || j >= (N_COLS + SPLIT)) {
            if (fscanf(fp, "%s", buff) == EOF) break;
            continue;
        }
        // after skip columns, read labels
        if (j < SPLIT) {
            if (fscanf(
                fp, scan_format,
                &(y[i * FEAT_DIM[NUM_LAYERS] + (j - SKIP_COLS)])) == EOF) break;
            continue;
        }
        if (fscanf(fp, scan_format, &x[i * N_COLS + (j - SPLIT)]) == EOF) break;
        }
    }
    fclose(fp);
    printf("Dataset read successfully\n");
    CUDA_CHECK(cudaMemcpy(x_out.ptr(), x.data(), x_out.nbytes(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(y_out.ptr(), y.data(), y_out.nbytes(),
                          cudaMemcpyHostToDevice));
} // read_data

} // mcmc namespace
