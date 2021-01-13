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

#include <thrust/device_ptr.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "kernel_helpers.h"
#include "utils_cuda.h"

// forward declarations
template <typename T>
class HostMatrix;
template <typename T>
class DeviceMatrix;

template <typename T>
class Matrix {
 public:
  Matrix() : view(false), M(0), N(0), data(nullptr) {}
  Matrix(size_t M, size_t N) : view(false), M(M), N(N) {}
  Matrix(size_t M, size_t N, T* data) : view(true), M(M), N(N), data(data) {}
  Matrix(const Matrix& other)
   : view(true)
   , M(other.M)
   , N(other.N)
   , data(other.data)
   , display_thres(other.display_thres)
   , display_cut(other.display_cut) {}
  virtual ~Matrix() = 0;

 protected:
  bool view;
  size_t M, N;
  T* data;

 public:
  size_t display_thres = 0;
  size_t display_cut   = 0;

 public:
  virtual HostMatrix<T> toHost() const = 0;
  template <typename U>
  friend std::ostream& operator<<(std::ostream& os, const Matrix<U>& mat);
  HDI const T* ptr() const { return data; }
  HDI T* ptr() { return data; }
  HDI size_t nbytes() const { return nel() * sizeof(T); }
  HDI size_t nel() const { return M * N; }
  HDI size_t rows() const { return M; }
  HDI size_t cols() const { return N; }
  HDI bool is_view() const { return view; }

  void swap(Matrix<T>& other) noexcept {
    std::swap(view, other.view);
    std::swap(M, other.M);
    std::swap(N, other.N);
    std::swap(data, other.data);
    std::swap(display_thres, other.display_thres);
    std::swap(display_cut, other.display_cut);
  }
};

template <typename T>
Matrix<T>::~Matrix() {}

template <typename T>
class HostMatrix : public Matrix<T> {
 public:
  HDI T* begin() { return this->ptr(); }
  HDI T* end() { return this->ptr() + this->nel(); }
  HDI T const* begin() const { return this->ptr(); }
  HDI T const* end() const { return this->ptr() + this->nel(); }

  HostMatrix() : Matrix<T>() {}
  HostMatrix(size_t M, size_t N, const T* init_val = nullptr)
   : Matrix<T>(M, N) {
    this->data = (T*)malloc(this->nbytes());
    if (init_val != nullptr) {
      for (size_t i = 0; i < this->nel(); i++) this->data[i] = *init_val;
    }
  }
  HostMatrix(size_t M, size_t N, T* data) : Matrix<T>(M, N, data) {}
  HostMatrix(const DeviceMatrix<T>& other)
   : HostMatrix<T>(other.rows(), other.cols()) {
    this->display_thres = other.display_thres;
    this->display_cut   = other.display_cut;
    CUDA_CHECK(cudaMemcpy(this->data, other.ptr(), this->nbytes(),
                          cudaMemcpyDeviceToHost));
  }
  ~HostMatrix() {
    if (this->data != nullptr && !this->is_view()) free(this->data);
  }
  HostMatrix<T> toHost() const { return HostMatrix<T>(*this); }
};

template <typename T>
class DeviceMatrix : public Matrix<T> {
 public:
  using d_ptr  = thrust::device_ptr<T>;
  using d_cptr = thrust::device_ptr<T>;
  HDI d_ptr begin() { return d_ptr{this->ptr()}; }
  HDI d_ptr end() { return d_ptr{this->ptr() + this->nel()}; }
  HDI d_cptr begin() const { return d_cptr{this->ptr()}; }
  HDI d_cptr end() const { return d_cptr{this->ptr() + this->nel()}; }

  DeviceMatrix() : Matrix<T>() {}
  DeviceMatrix(size_t M, size_t N, bool fill=false) : Matrix<T>(M, N) {
    CUDA_CHECK(cudaMalloc(&(this->data), this->nbytes()));
    if (fill) fill_mat(this->data, int(M), int(N));
  }
  DeviceMatrix(size_t M, size_t N, T init_val, const cudaStream_t& init_stream)
   : DeviceMatrix<T>(M, N) {
     fill_mat_fixed(init_stream, this->ptr(), int(this->nel()), init_val);
  }
  DeviceMatrix(size_t M, size_t N, T* data) : Matrix<T>(M, N, data) {}
  DeviceMatrix(const HostMatrix<T>& other)
   : DeviceMatrix<T>(other.rows(), other.cols()) {
    this->display_thres = other.display_thres;
    this->display_cut   = other.display_cut;
    CUDA_CHECK(cudaMemcpy(this->data, other.ptr(), this->nbytes(),
                          cudaMemcpyHostToDevice));
  }
  ~DeviceMatrix() {
    if (this->data != nullptr && !this->is_view())
      CUDA_CHECK_NO_THROW(cudaFree(this->data));
  }
  HostMatrix<T> toHost() const { return HostMatrix<T>(*this); }
};

template <typename T>
void printData(std::ostream& os, const T* data, size_t n) {
  if (n <= 0) return;
  os << data[0];
  for (size_t i = 1; i < n; i++) os << " " << data[i];
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat) {
  if (mat.ptr() == nullptr || mat.rows() == 0 || mat.cols() == 0) {
    os << "Empty matrix" << std::endl;
    return os;
  }
  HostMatrix<T> matHost = mat.toHost();
  const T* data         = matHost.ptr();
  size_t n              = matHost.nel();
  T mean = T(0), min = T(INFINITY), max = -T(INFINITY);
  for (size_t i = 0; i < n; i++) {
    T v = data[i];
    mean += v / T(matHost.nel());
    if (v > max) max = v;
    if (v < min) min = v;
  }
  os << "Shape: " << matHost.rows() << " x " << matHost.cols();
  os << ", word: " << sizeof(T) << ", min: " << min << ", max: ";
  os << max << ", mean: " << mean << std::endl;
  if (matHost.display_thres <= 0) return os;

  if (n <= matHost.display_thres) {
    os << "All elements: ";
    printData(os, data, n);
    os << std::endl;
    return os;
  }
  ASSERT(matHost.display_cut <= matHost.display_thres * 2,
         "When displaying, cut must be smaller than threshold * 2!");
  n = matHost.display_cut;
  os << "First " << n / 2 << " elements: ";
  printData(os, data, n / 2);
  os << std::endl;
  os << "Last " << n / 2 << " elements: ";
  printData(os, data + matHost.nel() - n / 2, n / 2);
  os << std::endl;
  return os;
}

template <typename T>
void assert_close(
    Matrix<T>& in1, Matrix<T>& in2, const std::string& name,
    double atol = 1e-5, double rtol = 1e-4) {
  ASSERT(in1.rows() == in2.rows() && in1.cols() == in2.cols(),
         "checking two matrixes requires same size");
  HostMatrix<T> h1 = in1.toHost();
  HostMatrix<T> h2 = in2.toHost();
  size_t n_errors = 0, err_idx = 0, max_idx = 0;
  double max_error = T(-1), rel_error = T(-1);
  double max_v1 = T(NAN), max_v2 = T(NAN);
  for (size_t i = 0; i < h1.nel(); ++i) {
    double v1 = double(h1.ptr()[i]), v2 = double(h2.ptr()[i]);
    double err = std::abs(v1 - v2);
    double ref = std::max(std::abs(v1), std::abs(v2));
    if (err > atol + rtol * ref) {
      n_errors++;
      max_idx = i;
      if (err > max_error) {
        max_error = err;
        rel_error = err / ref;
        err_idx   = i;
        max_v1 = v1;
        max_v2 = v2;
      }
    }
    if ((std::isnan(v1) && !std::isnan(v2)) ||
        (!std::isnan(v1) && std::isnan(v2))) {
      n_errors++;
      max_idx = i;
      rel_error = double(NAN);
      err_idx   = i;
      // this makes all subsequent comparisons with max_error false
      max_error = double(NAN);
      max_v1 = v1;
      max_v2 = v2;
    }
  }
  ASSERT(n_errors == 0,
         "Found %ld errors (max error %lf / "
         "rel error %lf, v1/v2 %lf/%lf at %ld, %ld) in matrix '%s', %ld",
         n_errors, max_error, rel_error, max_v1, max_v2,
         err_idx / h1.cols(), err_idx % h1.cols(), name.c_str(), max_idx);
}
