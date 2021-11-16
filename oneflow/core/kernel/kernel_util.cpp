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
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/ep/include/primitive/memset.h"
#include "oneflow/core/stream/include/stream_context_adapter.h"

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
void EmptyInitializer() {}

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

}  // namespace

void AutoMemcpy(DeviceCtx* ctx, void* dst, const void* src, size_t sz,
                const MemoryCase& dst_mem_case, const MemoryCase& src_mem_case) {
  std::unique_ptr<StreamContext> stream_ctx(NewStreamContextAdapter(ctx));
  AutoMemcpy(stream_ctx.get(), dst, src, sz, dst_mem_case, src_mem_case);
}

void AutoMemcpy(DeviceCtx* ctx, Blob* dst, const Blob* src) {
  std::unique_ptr<StreamContext> stream_ctx(NewStreamContextAdapter(ctx));
  AutoMemcpy(stream_ctx.get(), dst, src);
}

void AutoMemcpy(StreamContext* stream_ctx, void* dst, const void* src, size_t sz,
                const MemoryCase& dst_mem_case, const MemoryCase& src_mem_case) {
  ep::primitive::MemcpyKind kind{};
  if (stream_ctx->device_type() == DeviceType::kCPU) {
    CHECK(src_mem_case.has_host_mem());
    CHECK(dst_mem_case.has_host_mem());
    kind = ep::primitive::MemcpyKind::kDtoD;
  } else {
    if (src_mem_case.has_host_mem()) {
      CHECK(!dst_mem_case.has_host_mem());
      kind = ep::primitive::MemcpyKind::kHtoD;
    } else if (dst_mem_case.has_host_mem()) {
      CHECK(!src_mem_case.has_host_mem());
      kind = ep::primitive::MemcpyKind::kDtoH;
    } else {
      kind = ep::primitive::MemcpyKind::kDtoD;
    }
  }
  std::unique_ptr<ep::primitive::Memcpy> primitive =
      ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(stream_ctx->device_type(), kind);
  CHECK(primitive);
  primitive->Launch(stream_ctx->stream(), dst, src, sz);
}

void AutoMemcpy(StreamContext* stream_ctx, Blob* dst, const Blob* src) {
  const size_t body_bytes = src->ByteSizeOfBlobBody();
  CHECK_EQ(dst->ByteSizeOfBlobBody(), body_bytes);
  AutoMemcpy(stream_ctx, dst->mut_dptr(), src->dptr(), body_bytes, dst->mem_case(),
             src->mem_case());
}

void SyncAutoMemcpy(DeviceCtx* ctx, void* dst, const void* src, size_t sz,
                    const MemoryCase& dst_mem_case, const MemoryCase& src_mem_case) {
  AutoMemcpy(ctx, dst, src, sz, dst_mem_case, src_mem_case);
  ctx->SyncDevice();
}

void AutoMemset(DeviceCtx* ctx, void* dst, const char value, size_t sz,
                const MemoryCase& dst_mem_case) {
  std::unique_ptr<StreamContext> stream_ctx(NewStreamContextAdapter(ctx));
  AutoMemset(stream_ctx.get(), dst, value, sz, dst_mem_case);
}

void AutoMemset(StreamContext* stream_ctx, void* dst, const char value, size_t sz,
                const MemoryCase& /*dst_mem_case*/) {
  std::unique_ptr<ep::primitive::Memset> primitive =
      ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(stream_ctx->device_type());
  primitive->Launch(stream_ctx->stream(), dst, value, sz);
}

#define KU_IF_METHOD                     \
  template<typename T, typename Derived> \
  void CpuKernelUtilIf<T, Derived>::

#define KU_FLOATING_METHOD \
  template<typename T>     \
  void KernelUtil<DeviceType::kCPU, T, typename std::enable_if<IsFloating<T>::value>::type>::

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
  } else if (initializer_conf.has_empty_conf()) {
    EmptyInitializer<T>();
  } else {
    UNIMPLEMENTED();
  }
}

#define KU_INTEGRAL_METHOD \
  template<typename T>     \
  void KernelUtil<DeviceType::kCPU, T, typename std::enable_if<IsIntegral<T>::value>::type>::

KU_INTEGRAL_METHOD InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                      uint32_t random_seed, Blob* blob) {
  if (initializer_conf.has_constant_int_conf()) {
    ConstantInitializer<T>(static_cast<T>(initializer_conf.constant_int_conf().value()), blob);
  } else if (initializer_conf.has_random_uniform_int_conf()) {
    RandomIntUniformInitializer<T>(initializer_conf.random_uniform_int_conf(), random_seed, blob);
  } else if (initializer_conf.has_int_range_conf()) {
    IntSequenceInitializer<T>(initializer_conf.int_range_conf(), random_seed, blob);
  } else if (initializer_conf.has_empty_conf()) {
    EmptyInitializer<T>();
  } else {
    UNIMPLEMENTED();
  }
}

#define INSTANTIATE_KERNEL_UTIL(type_cpp, type_proto)                                \
  template struct CpuKernelUtilIf<type_cpp, KernelUtil<DeviceType::kCPU, type_cpp>>; \
  template struct KernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ);

}  //  namespace oneflow
