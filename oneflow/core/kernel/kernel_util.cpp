/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/memory/memory_case.pb.h"

namespace oneflow {

namespace {

template<typename T>
void RngUniform(const int64_t elem_cnt, const T min, const T max, uint32_t random_seed, T* dptr) {
  CHECK_GE(elem_cnt, 0);
  CHECK(dptr);
  CHECK_LE(min, max);
  std::mt19937 generator(random_seed);
  std::uniform_real_distribution<T> random_distribution(min, std::nextafter(max, GetMaxVal<T>()));
  for (int64_t i = 0; i < elem_cnt; ++i) { dptr[i] = random_distribution(generator); }
}

template<typename T>
void RngIntUniform(const int64_t elem_cnt, const T min, const T max, uint32_t random_seed,
                   T* dptr) {
  CHECK_GE(elem_cnt, 0);
  CHECK(dptr);
  CHECK_LE(min, max);
  std::mt19937 generator(random_seed);
  std::uniform_int_distribution<T> random_distribution(min, std::nextafter(max, GetMaxVal<T>()));
  for (int64_t i = 0; i < elem_cnt; ++i) { dptr[i] = random_distribution(generator); }
}

template<typename T>
void RngNormal(const int64_t elem_cnt, const T mean, const T std, uint32_t random_seed, T* dptr) {
  CHECK_GE(elem_cnt, 0);
  CHECK(dptr);
  CHECK_GT(std, 0.0);
  std::mt19937 generator(random_seed);
  std::normal_distribution<T> random_distribution(mean, std);

  for (int64_t i = 0; i < elem_cnt; ++i) { dptr[i] = random_distribution(generator); }
}

template<typename T>
void RngTruncatedNormal(const int64_t elem_cnt, const T mean, const T std, uint32_t random_seed,
                        T* dptr) {
  CHECK_GE(elem_cnt, 0);
  CHECK(dptr);
  CHECK_GT(std, 0.0);
  T truncated_value = 2 * std;
  std::mt19937 generator(random_seed);
  std::normal_distribution<T> random_distribution(mean, std);
  int64_t index = 0;
  while (true) {
    T val = random_distribution(generator);
    if (std::abs(val - mean) < truncated_value) {
      dptr[index++] = val;
      if (index >= elem_cnt) { break; }
    }
  }
}

template<typename T>
void ConstantInitializer(const T& value, Blob* blob) {
  T* dptr = blob->mut_dptr<T>();
  const int64_t elem_cnt = blob->shape().elem_cnt();
  CHECK(elem_cnt);
  for (int64_t i = 0; i < elem_cnt; ++i) { dptr[i] = value; }
}

template<typename T>
void RandomUniformInitializer(const RandomUniformInitializerConf& initializer_conf,
                              uint32_t random_seed, Blob* blob) {
  CHECK(blob->shape().elem_cnt());
  RngUniform<T>(blob->shape().elem_cnt(), static_cast<T>(initializer_conf.min()),
                static_cast<T>(initializer_conf.max()), random_seed, blob->mut_dptr<T>());
}

template<typename T>
void RandomIntUniformInitializer(const RandomUniformIntInitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob) {
  CHECK(blob->shape().elem_cnt());
  RngIntUniform<T>(blob->shape().elem_cnt(), static_cast<T>(initializer_conf.min()),
                   static_cast<T>(initializer_conf.max()), random_seed, blob->mut_dptr<T>());
}

template<typename T>
void RandomNormalInitializer(const RandomNormalInitializerConf& initializer_conf,
                             uint32_t random_seed, Blob* blob) {
  CHECK(blob->shape().elem_cnt());
  RngNormal<T>(blob->shape().elem_cnt(), static_cast<T>(initializer_conf.mean()),
               static_cast<T>(initializer_conf.std()), random_seed, blob->mut_dptr<T>());
}

template<typename T>
void TruncatedNormalInitializer(const TruncatedNormalInitializerConf& initializer_conf,
                                uint32_t random_seed, Blob* blob) {
  CHECK(blob->shape().elem_cnt());
  RngTruncatedNormal<T>(blob->shape().elem_cnt(), static_cast<T>(initializer_conf.mean()),
                        static_cast<T>(initializer_conf.std()), random_seed, blob->mut_dptr<T>());
}

template<typename T>
T GenInitialFan(VarianceNorm variance_norm, Blob* blob, const std::string& data_format) {
  T fan = GetZeroVal<T>();
  T fan_in = static_cast<T>(blob->shape().Count(1));
  T fan_out = static_cast<T>(blob->shape().At(0));
  if (data_format == "channels_first") {
    fan_out *= static_cast<T>(blob->shape().Count(2));
  } else if (data_format == "channels_last") {
    fan_out *= static_cast<T>(blob->shape().Count(1, blob->shape().NumAxes() - 1));
  } else {
    CHECK_EQ(blob->shape().NumAxes(), 2);
    CHECK_EQ(data_format, "");
  }
  if (variance_norm == VarianceNorm::kAverage) {
    fan = (fan_in + fan_out) / static_cast<T>(2);
  } else if (variance_norm == VarianceNorm::kFanIn) {
    fan = fan_in;
  } else if (variance_norm == VarianceNorm::kFanOut) {
    fan = fan_out;
  } else {
    UNIMPLEMENTED();
  }
  return fan;
}

template<typename T>
void XavierInitializer(const XavierInitializerConf& initializer_conf, uint32_t random_seed,
                       Blob* blob) {
  CHECK(blob->shape().elem_cnt());
  VarianceNorm variance_norm = static_cast<VarianceNorm>(initializer_conf.variance_norm());
  T scale = std::sqrt(static_cast<T>(3)
                      / GenInitialFan<T>(variance_norm, blob, initializer_conf.data_format()));
  RngUniform<T>(blob->shape().elem_cnt(), static_cast<T>(-scale), static_cast<T>(scale),
                random_seed, blob->mut_dptr<T>());
}

template<typename T>
void MsraInitializer(const MsraInitializerConf& initializer_conf, uint32_t random_seed,
                     Blob* blob) {
  CHECK(blob->shape().elem_cnt());
  VarianceNorm variance_norm = static_cast<VarianceNorm>(initializer_conf.variance_norm());
  T std = std::sqrt(static_cast<T>(2)
                    / GenInitialFan<T>(variance_norm, blob, initializer_conf.data_format()));
  RngNormal<T>(blob->shape().elem_cnt(), GetZeroVal<T>(), static_cast<T>(std), random_seed,
               blob->mut_dptr<T>());
}

template<typename T>
void VarianceScalingInitializer(const VarianceScalingInitializerConf& initializer_conf,
                                uint32_t random_seed, Blob* blob) {
  CHECK(blob->shape().elem_cnt());
  VarianceNorm variance_norm = static_cast<VarianceNorm>(initializer_conf.variance_norm());
  T scale = static_cast<T>(initializer_conf.scale())
            / GenInitialFan<T>(variance_norm, blob, initializer_conf.data_format());
  RandomDistribution distribution =
      static_cast<RandomDistribution>(initializer_conf.distribution());
  if (kTruncatedNormal == distribution) {
    T stddev = std::sqrt(scale) / static_cast<T>(0.87962566103423978f);
    RngTruncatedNormal<T>(blob->shape().elem_cnt(), GetZeroVal<T>(), stddev, random_seed,
                          blob->mut_dptr<T>());
  } else if (kRandomNormal == distribution) {
    T stddev = std::sqrt(scale);
    RngNormal<T>(blob->shape().elem_cnt(), GetZeroVal<T>(), stddev, random_seed,
                 blob->mut_dptr<T>());
  } else {
    T limit = std::sqrt(static_cast<T>(3.0) * scale);
    RngUniform<T>(blob->shape().elem_cnt(), -limit, limit, random_seed, blob->mut_dptr<T>());
  }
}

template<typename T>
void RangeInitializer(int64_t outer_size, int64_t idx_dim_size, int64_t inner_size, T start,
                      T stride, T* out) {
  FOR_RANGE(int64_t, i, 0, outer_size) {
    FOR_RANGE(int64_t, j, 0, idx_dim_size) {
      FOR_RANGE(int64_t, k, 0, inner_size) {
        *(out + i * idx_dim_size * inner_size + j * inner_size + k) = start + j * stride;
      }
    }
  }
}

template<typename T, typename RangeInitializerConfT>
void RangeInitializer(const RangeInitializerConfT& initializer_conf, uint32_t random_seed,
                      Blob* blob) {
  CHECK_GT(blob->shape().NumAxes(), 0);
  const int64_t axis = initializer_conf.axis() < 0
                           ? blob->shape().NumAxes() + initializer_conf.axis()
                           : initializer_conf.axis();
  CHECK_GE(axis, 0);
  CHECK_LT(axis, blob->shape().NumAxes());
  RangeInitializer<T>(blob->shape().Count(0, axis), blob->shape().At(axis),
                      blob->shape().Count(axis + 1), static_cast<T>(initializer_conf.start()),
                      static_cast<T>(initializer_conf.stride()), blob->mut_dptr<T>());
}

template<typename T>
void RangeInitializer(const RangeInitializerConf& initializer_conf, uint32_t random_seed,
                      Blob* blob) {
  RangeInitializer<T, RangeInitializerConf>(initializer_conf, random_seed, blob);
}

template<typename T>
void IntSequenceInitializer(const IntRangeInitializerConf& initializer_conf, uint32_t random_seed,
                            Blob* blob) {
  RangeInitializer<T, IntRangeInitializerConf>(initializer_conf, random_seed, blob);
}

void ComputeOffset(const int32_t num_axes, const int64_t* shape, const int32_t* permutation,
                   DimVector& offset) {
  offset.resize(num_axes);
  DimVector buff(num_axes);
  int64_t cur_offset = 1;
  for (int32_t i = num_axes - 1; i >= 0; --i) {
    buff[i] = cur_offset;
    cur_offset *= shape[i];
  }
  for (int32_t i = 0; i < num_axes; ++i) { offset[permutation[i]] = buff[i]; }
}

void IncreaseIndex(const int64_t* shape, DimVector& index) {
  for (int32_t i = index.size() - 1; i >= 0; --i) {
    ++index[i];
    if (index[i] >= shape[i]) {
      index[i] -= shape[i];
    } else {
      break;
    }
  }
}

template<typename T, T (*reduce_core_func)(const T, const T)>
void MatrixRowReduce(const int64_t row_num, const int64_t col_num, const T* x, T* y) {
  FOR_RANGE(int64_t, i, 0, row_num) {
    y[i] = x[i * col_num];
    FOR_RANGE(int64_t, j, 1, col_num) { y[i] = reduce_core_func(y[i], x[i * col_num + j]); }
  }
}

}  // namespace

void AutoMemcpy(DeviceCtx* ctx, void* dst, const void* src, size_t sz,
                const MemoryCase& dst_mem_case, const MemoryCase& src_mem_case) {
  void (*func)(DeviceCtx*, void* dst, const void* src, size_t sz);
  if (src_mem_case.has_host_mem() && dst_mem_case.has_host_mem()) {
    func = &Memcpy<DeviceType::kCPU>;
  } else {
#ifdef WITH_CUDA
    func = &Memcpy<DeviceType::kGPU>;
#else
    UNIMPLEMENTED();
#endif  // WITH_CUDA
  }
  func(ctx, dst, src, sz);
}

void SyncAutoMemcpy(DeviceCtx* ctx, void* dst, const void* src, size_t sz,
                    const MemoryCase& dst_mem_case, const MemoryCase& src_mem_case) {
  AutoMemcpy(ctx, dst, src, sz, dst_mem_case, src_mem_case);
  if (src_mem_case.has_device_cuda_mem() || dst_mem_case.has_device_cuda_mem()) {
#ifdef WITH_CUDA
    OF_CUDA_CHECK(cudaStreamSynchronize(ctx->cuda_stream()));
#else
    UNIMPLEMENTED();
#endif  // WITH_CUDA
  }
}

#define KU_IF_METHOD                     \
  template<typename T, typename Derived> \
  void CpuKernelUtilIf<T, Derived>::

KU_IF_METHOD Axpy(DeviceCtx* ctx, const int n, const T* alpha, const T* x, const int incx, T* y,
                  const int incy) {
  Derived::Axpy(ctx, n, *alpha, x, incx, y, incy);
}
KU_IF_METHOD Max(DeviceCtx* ctx, const int64_t n, const T* x, T* max_ptr) {
  *max_ptr = *std::max_element(x, x + n);
}
KU_IF_METHOD Max(DeviceCtx* ctx, const int64_t n, const T* x, T* max_ptr, T* temp_storage,
                 size_t temp_storage_bytes) {
  Max(ctx, n, x, max_ptr);
}
KU_IF_METHOD Sum(DeviceCtx* ctx, const int64_t n, const T* x, T* sum_ptr) {
  *sum_ptr = 0;
  for (int64_t i = 0; i < n; ++i) { *sum_ptr += x[i]; }
}
KU_IF_METHOD Sum(DeviceCtx* ctx, const int64_t n, const T* x, T* sum_ptr, T* temp_storage,
                 size_t temp_storage_bytes) {
  Sum(ctx, n, x, sum_ptr);
}
KU_IF_METHOD CopyColsRegion(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num,
                            const T* x, const int64_t x_col_offset, const int64_t x_lda, T* y,
                            const int64_t y_col_offset, const int64_t y_lda) {
  for (int64_t i = 0; i < row_num; ++i) {
    for (int64_t j = 0; j < col_num; ++j) {
      y[i * y_lda + y_col_offset + j] = x[i * x_lda + x_col_offset + j];
    }
  }
}
KU_IF_METHOD RowMax(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const T* x,
                    T* y) {
  MatrixRowReduce<T, ReduceCoreMax>(row_num, col_num, x, y);
}
KU_IF_METHOD RowSum(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const T* x,
                    T* y) {
  MatrixRowReduce<T, ReduceCoreAdd>(row_num, col_num, x, y);
}
KU_IF_METHOD Transpose(DeviceCtx* ctx, const int32_t num_axis, const ShapeView& x_shape,
                       const ShapeView& y_shape, const PbRf<int32_t>& permutation,
                       const int64_t elem_cnt, const T* x, T* y) {
  int64_t block_size = 1;
  int32_t shared_idxs_num = 0;
  for (int32_t i = num_axis - 1; i >= 0 && permutation[i] == i; --i) {
    block_size *= y_shape.At(i);
    ++shared_idxs_num;
  }
  if (num_axis < 2 || shared_idxs_num == num_axis) {
    memcpy(y, x, elem_cnt * sizeof(T));
    return;
  }
  int32_t trans_axis = num_axis - shared_idxs_num;
  DimVector x_to_y_offset;
  ComputeOffset(trans_axis, y_shape.ptr(), permutation.data(), x_to_y_offset);
  DimVector x_index_digits(trans_axis, 0);
  int64_t num_blocks = elem_cnt / block_size;
  FOR_RANGE(int64_t, x_idx, 0, num_blocks) {
    int64_t y_idx = std::inner_product(x_to_y_offset.cbegin(), x_to_y_offset.cend(),
                                       x_index_digits.cbegin(), 0);
    if (block_size == 1) {
      y[y_idx] = x[x_idx];
    } else {
      memcpy(y + block_size * y_idx, x + block_size * x_idx, block_size * sizeof(T));
    }
    IncreaseIndex(x_shape.ptr(), x_index_digits);
  }
}
KU_IF_METHOD Set(DeviceCtx* ctx, const T value, T* addr) { *addr = value; }
KU_IF_METHOD Replicate(DeviceCtx* ctx, const int64_t n, T* y, const T* x) {
  for (int64_t i = 0; i < n; ++i) { y[i] = *x; }
}

#define KU_FLOATING_METHOD \
  template<typename T>     \
  void KernelUtil<DeviceType::kCPU, T, typename std::enable_if<IsFloating<T>::value>::type>::

KU_FLOATING_METHOD Dot(DeviceCtx* ctx, const int n, const T* x, const int incx, const T* y,
                       const int incy, T* result) {
  *result = cblas_dot<T>(n, x, incx, y, incy);
}
KU_FLOATING_METHOD Copy(DeviceCtx* ctx, const int n, const T* x, const int incx, T* y,
                        const int incy) {
  cblas_copy<T>(n, x, incx, y, incy);
}
KU_FLOATING_METHOD Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx,
                        T* y, const int incy) {
  cblas_axpy<T>(n, alpha, x, incx, y, incy);
}
KU_FLOATING_METHOD Scal(DeviceCtx* ctx, const int n, const T alpha, T* x, const int incx) {
  cblas_scal<T>(n, alpha, x, incx);
}
KU_FLOATING_METHOD Scal(DeviceCtx* ctx, const int n, const T* alpha, T* x, const int incx) {
  Scal(ctx, n, *alpha, x, incx);
}
KU_FLOATING_METHOD Gemv(DeviceCtx* ctx, const enum CBLAS_TRANSPOSE trans, int m, int n,
                        const T alpha, const T* a, int lda, const T* x, const int incx,
                        const T beta, T* y, const int incy) {
  // Set col major to keep it as the same with cublas
  cblas_gemv<T>(CBLAS_ORDER::CblasColMajor, trans, n, m, alpha, a, lda, x, incx, beta, y, incy);
}
KU_FLOATING_METHOD Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                        const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_TRANSPOSE trans_b,
                        const int m, const int n, const int k, const T alpha, const T* a,
                        const int lda, const T* b, const int ldb, const T beta, T* c,
                        const int ldc) {
  cblas_gemm<T>(order, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
KU_FLOATING_METHOD BatchedGemm(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                               const enum CBLAS_TRANSPOSE trans_a,
                               const enum CBLAS_TRANSPOSE trans_b, int batch_size, int m, int n,
                               int k, const T alpha, const T* a, const T* b, const T beta, T* c,
                               T** buf) {
  const int a_stride = m * k;
  const int b_stride = k * n;
  const int c_stride = m * n;
  FOR_RANGE(int32_t, i, 0, batch_size) {
    KernelUtil<DeviceType::kCPU, T>::OFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a + i * a_stride,
                                            b + i * b_stride, beta, c + i * c_stride);
  }
}

KU_FLOATING_METHOD Exp(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
  for (int64_t i = 0; i < n; ++i) { y[i] = std::exp(x[i]); }
}
KU_FLOATING_METHOD Div(DeviceCtx* ctx, const int64_t n, T* x, const T* alpha) {
  for (int64_t i = 0; i < n; ++i) { x[i] = x[i] / (*alpha); }
}
KU_FLOATING_METHOD Div(DeviceCtx* ctx, const int64_t n, T* x, const T alpha) {
  for (int64_t i = 0; i < n; ++i) { x[i] = x[i] / alpha; }
}
KU_FLOATING_METHOD Mul(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z) {
  for (int64_t i = 0; i < n; ++i) { z[i] = x[i] * y[i]; }
}
KU_FLOATING_METHOD Div(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z) {
  for (int64_t i = 0; i < n; ++i) { z[i] = x[i] / y[i]; }
}
KU_FLOATING_METHOD Square(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
  for (int64_t i = 0; i < n; ++i) { y[i] = x[i] * x[i]; }
}
KU_FLOATING_METHOD Sqrt(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
  for (int64_t i = 0; i < n; ++i) { y[i] = std::sqrt(x[i]); }
}
KU_FLOATING_METHOD Reciprocal(DeviceCtx* ctx, const int n, const T* x, T* y) {
  for (int64_t i = 0; i < n; ++i) { y[i] = static_cast<T>(1.0) / x[i]; }
}
KU_FLOATING_METHOD Rsqrt(DeviceCtx* ctx, const int64_t n, T* x, const float epsilon) {
  for (int64_t i = 0; i < n; ++i) { x[i] = 1.0 / std::sqrt(x[i] + epsilon); }
}
KU_FLOATING_METHOD Rsqrt(DeviceCtx* ctx, const int64_t n, const T* x, T* y, const float epsilon) {
  for (int64_t i = 0; i < n; ++i) { y[i] = 1.0 / std::sqrt(x[i] + epsilon); }
}
KU_FLOATING_METHOD Powx(DeviceCtx* ctx, const int64_t n, const T* x, const float power, T* y) {
  for (int64_t i = 0; i < n; ++i) { y[i] = std::pow(x[i], power); }
}

KU_FLOATING_METHOD Sigmoid(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
  T half = static_cast<T>(0.5);
  for (int64_t i = 0; i != n; ++i) { y[i] = half * std::tanh(half * x[i]) + half; }
}
KU_FLOATING_METHOD SigmoidBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y,
                                   const T* dy, T* dx) {
  for (int64_t i = 0; i != n; ++i) { dx[i] = y[i] * (1 - y[i]) * dy[i]; }
}
KU_FLOATING_METHOD Relu(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
  T zero = GetZeroVal<T>();
  for (int64_t i = 0; i != n; ++i) { y[i] = std::max(x[i], zero); }
}
KU_FLOATING_METHOD ReluBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y,
                                const T* dy, T* dx) {
  T zero = GetZeroVal<T>();
  for (int64_t i = 0; i != n; ++i) { dx[i] = (y[i] > zero) * dy[i]; }
}
KU_FLOATING_METHOD Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0) {
  for (int64_t i = 0; i != n; ++i) { out[i] = in_0[i]; }
}
KU_FLOATING_METHOD Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1) {
  for (int64_t i = 0; i != n; ++i) { out[i] = in_0[i] + in_1[i]; }
}
KU_FLOATING_METHOD Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                            const T* in_2) {
  for (int64_t i = 0; i != n; ++i) { out[i] = in_0[i] + in_1[i] + in_2[i]; }
}
KU_FLOATING_METHOD Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                            const T* in_2, const T* in_3) {
  for (int64_t i = 0; i != n; ++i) { out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i]; }
}
KU_FLOATING_METHOD Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                            const T* in_2, const T* in_3, const T* in_4) {
  for (int64_t i = 0; i != n; ++i) { out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i]; }
}
KU_FLOATING_METHOD Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                            const T* in_2, const T* in_3, const T* in_4, const T* in_5) {
  for (int64_t i = 0; i != n; ++i) {
    out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i] + in_5[i];
  }
}
KU_FLOATING_METHOD Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                            const T* in_2, const T* in_3, const T* in_4, const T* in_5,
                            const T* in_6) {
  for (int64_t i = 0; i != n; ++i) {
    out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i] + in_5[i] + in_6[i];
  }
}
KU_FLOATING_METHOD Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                            const T* in_2, const T* in_3, const T* in_4, const T* in_5,
                            const T* in_6, const T* in_7) {
  for (int64_t i = 0; i != n; ++i) {
    out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i] + in_5[i] + in_6[i] + in_7[i];
  }
}
KU_FLOATING_METHOD Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                            const T* in_2, const T* in_3, const T* in_4, const T* in_5,
                            const T* in_6, const T* in_7, const T* in_8) {
  for (int64_t i = 0; i != n; ++i) {
    out[i] =
        in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i] + in_5[i] + in_6[i] + in_7[i] + in_8[i];
  }
}

KU_FLOATING_METHOD InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                      uint32_t random_seed, Blob* blob) {
  if (initializer_conf.has_constant_conf()) {
    ConstantInitializer<T>(static_cast<T>(initializer_conf.constant_conf().value()), blob);
  } else if (initializer_conf.has_constant_int_conf()) {
    ConstantInitializer<T>(initializer_conf.constant_int_conf().value(), blob);
  } else if (initializer_conf.has_random_uniform_conf()) {
    RandomUniformInitializer<T>(initializer_conf.random_uniform_conf(), random_seed, blob);
  } else if (initializer_conf.has_random_normal_conf()) {
    RandomNormalInitializer<T>(initializer_conf.random_normal_conf(), random_seed, blob);
  } else if (initializer_conf.has_truncated_normal_conf()) {
    TruncatedNormalInitializer<T>(initializer_conf.truncated_normal_conf(), random_seed, blob);
  } else if (initializer_conf.has_xavier_conf()) {
    XavierInitializer<T>(initializer_conf.xavier_conf(), random_seed, blob);
  } else if (initializer_conf.has_msra_conf()) {
    MsraInitializer<T>(initializer_conf.msra_conf(), random_seed, blob);
  } else if (initializer_conf.has_range_conf()) {
    RangeInitializer<T>(initializer_conf.range_conf(), random_seed, blob);
  } else if (initializer_conf.has_variance_scaling_conf()) {
    VarianceScalingInitializer<T>(initializer_conf.variance_scaling_conf(), random_seed, blob);
  } else {
    UNIMPLEMENTED();
  }
}

#define KU_INTEGRAL_METHOD \
  template<typename T>     \
  void KernelUtil<DeviceType::kCPU, T, typename std::enable_if<IsIntegral<T>::value>::type>::

KU_INTEGRAL_METHOD Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx,
                        T* y, const int incy) {
  FOR_RANGE(int, i, 0, n) {
    *y += alpha * *x;
    x += incx;
    y += incy;
  }
}
KU_INTEGRAL_METHOD InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                      uint32_t random_seed, Blob* blob) {
  if (initializer_conf.has_constant_int_conf()) {
    ConstantInitializer<T>(static_cast<T>(initializer_conf.constant_int_conf().value()), blob);
  } else if (initializer_conf.has_random_uniform_int_conf()) {
    RandomIntUniformInitializer<T>(initializer_conf.random_uniform_int_conf(), random_seed, blob);
  } else if (initializer_conf.has_int_range_conf()) {
    IntSequenceInitializer<T>(initializer_conf.int_range_conf(), random_seed, blob);
  } else {
    UNIMPLEMENTED();
  }
}

KU_INTEGRAL_METHOD Mul(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z) {
  for (int64_t i = 0; i < n; ++i) { z[i] = x[i] * y[i]; }
}

#define INSTANTIATE_KERNEL_UTIL(type_cpp, type_proto)                                \
  template struct CpuKernelUtilIf<type_cpp, KernelUtil<DeviceType::kCPU, type_cpp>>; \
  template struct KernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ);

}  //  namespace oneflow
