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

#ifdef WITH_CUTLASS

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/job/lazy_mode.h"
#include <cutlass/library/handle.h>
#include <cutlass/library/library.h>
#include <cutlass/library/singleton.h>

namespace oneflow {

namespace {

struct Conv2dOperationCacheKey {
  cutlass::library::ConvFunctionalKey functional_key;
  cutlass::library::Conv2dConfiguration configuraion;
  size_t alignment;
  Conv2dOperationCacheKey(cutlass::library::ConvFunctionalKey functional_key,
                          cutlass::library::Conv2dConfiguration configuraion,
                          cutlass::library::ConvArguments arguments)
      : functional_key(functional_key), configuraion(configuraion) {
    const auto IsStrideAligned = [&](const std::vector<int64_t>& stride, size_t n) {
      return std::all_of(stride.cbegin(), stride.cend(),
                         [&](const int64_t& s) { return s % n == 0; });
    };
    const auto IsAligned = [&](size_t n) {
      return reinterpret_cast<uintptr_t>(arguments.A) % n == 0
             && reinterpret_cast<uintptr_t>(arguments.B) % n == 0
             && reinterpret_cast<uintptr_t>(arguments.C) % n == 0
             && reinterpret_cast<uintptr_t>(arguments.D) % n == 0
             && IsStrideAligned(configuraion.stride_a, n)
             && IsStrideAligned(configuraion.stride_b, n)
             && IsStrideAligned(configuraion.stride_c, n);
    };
    if (IsAligned(8)) {
      alignment = 8;
    } else if (IsAligned(4)) {
      alignment = 4;
    } else if (IsAligned(2)) {
      alignment = 2;
    } else {
      alignment = 1;
    }
  }
};

struct Conv2dProblemSizeHasher {
  size_t operator()(const cutlass::conv::Conv2dProblemSize& problem_size) const {
    size_t hash = 0;
    hash = HashCombine(hash, std::hash<int>()(problem_size.N));
    hash = HashCombine(hash, std::hash<int>()(problem_size.H));
    hash = HashCombine(hash, std::hash<int>()(problem_size.W));
    hash = HashCombine(hash, std::hash<int>()(problem_size.C));
    hash = HashCombine(hash, std::hash<int>()(problem_size.P));
    hash = HashCombine(hash, std::hash<int>()(problem_size.Q));
    hash = HashCombine(hash, std::hash<int>()(problem_size.K));
    hash = HashCombine(hash, std::hash<int>()(problem_size.R));
    hash = HashCombine(hash, std::hash<int>()(problem_size.S));
    hash = HashCombine(hash, std::hash<int>()(problem_size.pad_h));
    hash = HashCombine(hash, std::hash<int>()(problem_size.pad_w));
    hash = HashCombine(hash, std::hash<int>()(problem_size.stride_h));
    hash = HashCombine(hash, std::hash<int>()(problem_size.stride_w));
    hash = HashCombine(hash, std::hash<int>()(problem_size.dilation_h));
    hash = HashCombine(hash, std::hash<int>()(problem_size.dilation_w));
    hash = HashCombine(hash, std::hash<int>()(static_cast<int>(problem_size.mode)));
    hash = HashCombine(hash, std::hash<int>()(problem_size.split_k_slices));
    hash = HashCombine(hash, std::hash<int>()(problem_size.groups));
    return hash;
  }
};

struct Conv2dConfigurationHasher {
  size_t operator()(const cutlass::library::Conv2dConfiguration& configuraion) const {
    size_t hash = std::hash<int>()(static_cast<int>(configuraion.split_k_mode));
    hash = HashCombine(hash, Conv2dProblemSizeHasher()(configuraion.problem_size));
    for (const int64_t v : configuraion.stride_a) {
      hash = HashCombine(hash, std::hash<int64_t>()(v));
    }
    for (const int64_t v : configuraion.stride_b) {
      hash = HashCombine(hash, std::hash<int64_t>()(v));
    }
    for (const int64_t v : configuraion.stride_c) {
      hash = HashCombine(hash, std::hash<int64_t>()(v));
    }
    return hash;
  }
};

struct Conv2dOperationCacheKeyHasher {
  size_t operator()(const Conv2dOperationCacheKey& key) const {
    size_t hash = cutlass::library::ConvFunctionalKeyHasher()(key.functional_key);
    hash = HashCombine(hash, Conv2dConfigurationHasher()(key.configuraion));
    hash = HashCombine(hash, std::hash<size_t>()(key.alignment));
    return hash;
  }
};

inline bool operator==(const cutlass::library::Conv2dConfiguration& lhs,
                       const cutlass::library::Conv2dConfiguration& rhs) {
  return lhs.split_k_mode == rhs.split_k_mode && lhs.problem_size == rhs.problem_size
         && lhs.stride_a == rhs.stride_a && lhs.stride_b == rhs.stride_b
         && lhs.stride_c == rhs.stride_c;
}

inline bool operator==(const Conv2dOperationCacheKey& lhs, const Conv2dOperationCacheKey& rhs) {
  return lhs.functional_key == rhs.functional_key && lhs.configuraion == rhs.configuraion
         && lhs.alignment == rhs.alignment;
}

const cutlass::library::Operation* FindConv2dOperation(
    ep::CudaStream* stream, cutlass::library::ConvFunctionalKey functional_key,
    const cutlass::library::Conv2dConfiguration& configuraion,
    const cutlass::library::ConvArguments& arguments, void* workspace, size_t workspace_size) {
  Conv2dOperationCacheKey cache_key(functional_key, configuraion, arguments);
  using CacheMap = std::unordered_map<Conv2dOperationCacheKey, const cutlass::library::Operation*,
                                      Conv2dOperationCacheKeyHasher>;
  static CacheMap cache;
  static std::mutex cache_mutex;
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(cache_key);
    if (it != cache.end()) { return it->second; }
  }

  constexpr int turing_warmup_iters = 3;
  constexpr int turing_iters = 7;
  cudaEvent_t start{};
  cudaEvent_t end{};
  OF_CUDA_CHECK(cudaEventCreate(&start));
  OF_CUDA_CHECK(cudaEventCreate(&end));
  const cutlass::library::Operation* fastest_operation = nullptr;
  float fastest_time = 0;
  const auto& operations_map_it =
      cutlass::library::Singleton::get().operation_table.conv2d_operations.find(functional_key);
  CHECK(operations_map_it
        != cutlass::library::Singleton::get().operation_table.conv2d_operations.cend());
  const cutlass::library::ConvOperationVectorMap& operations_map = operations_map_it->second;
  for (const auto& pair : operations_map) {
    for (auto operation : pair.second) {
      if (operation->description().tile_description.minimum_compute_capability * 10
              > stream->cuda_arch()
          || operation->description().tile_description.maximum_compute_capability * 10
                 < stream->cuda_arch()) {
        continue;
      }
      auto status = operation->can_implement(&configuraion, &arguments);
      const auto* conv_description =
          static_cast<const cutlass::library::ConvDescription*>(&operation->description());
      if (status != cutlass::Status::kSuccess) { continue; }
      const size_t host_workspace_size = operation->get_host_workspace_size(&configuraion);
      const size_t device_workspace_size = operation->get_device_workspace_size(&configuraion);
      if (device_workspace_size > workspace_size) { continue; }
      std::vector<uint8_t> host_workspace(host_workspace_size, 0);
      if (operation->initialize(&configuraion, host_workspace.data(), workspace,
                                stream->cuda_stream())
          != cutlass::Status::kSuccess) {
        continue;
      }

      const auto Run = [&]() {
        auto init_status = operation->initialize(&configuraion, host_workspace.data(), workspace,
                                                 stream->cuda_stream());
        CHECK(init_status == cutlass::Status::kSuccess);
        auto run_status =
            operation->run(&arguments, host_workspace.data(), workspace, stream->cuda_stream());
        CHECK(run_status == cutlass::Status::kSuccess);
      };
      OF_CUDA_CHECK(cudaStreamSynchronize(stream->cuda_stream()));
      for (int i = 0; i < turing_warmup_iters; ++i) { Run(); }
      OF_CUDA_CHECK(cudaEventRecord(start, stream->cuda_stream()));
      for (int i = 0; i < turing_iters; ++i) { Run(); }
      OF_CUDA_CHECK(cudaEventRecord(end, stream->cuda_stream()));
      OF_CUDA_CHECK(cudaEventSynchronize(end));
      float time = 0;
      OF_CUDA_CHECK(cudaEventElapsedTime(&time, start, end));
      VLOG(3) << operation->description().name << " " << time;
      if (fastest_operation == nullptr || time < fastest_time) {
        fastest_operation = operation;
        fastest_time = time;
      }
    }
  }
  OF_CUDA_CHECK(cudaEventDestroy(start));
  OF_CUDA_CHECK(cudaEventDestroy(end));
  CHECK(fastest_operation != nullptr);
  VLOG(3) << "Fastest: " << fastest_operation->description().name << " " << fastest_time;
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    cache[cache_key] = fastest_operation;
  }
  return fastest_operation;
}

class Conv2dCutlassKernel final : public user_op::OpKernel {
 public:
  Conv2dCutlassKernel() = default;
  ~Conv2dCutlassKernel() override = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
    const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
    CHECK(add_to_output == nullptr);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const auto& padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
    auto dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    auto strides = ctx->Attr<std::vector<int32_t>>("strides");

    const int n = in->shape_view().At(0);
    const int h = in->shape_view().At(1);
    const int w = in->shape_view().At(2);
    const int c = in->shape_view().At(3);

    const int k = weight->shape_view().At(0);
    const int r = weight->shape_view().At(1);
    const int s = weight->shape_view().At(2);
    CHECK_EQ(weight->shape_view().At(3), c);

    const int p = out->shape_view().At(1);
    const int q = out->shape_view().At(2);

    auto* stream = ctx->stream()->As<ep::CudaStream>();

    cutlass::library::ConvFunctionalKey key(
        cutlass::library::Provider::kCUTLASS, cutlass::library::ConvKind::kFprop,
        cutlass::library::NumericTypeID::kF16, cutlass::library::LayoutTypeID::kTensorNHWC,
        cutlass::library::NumericTypeID::kF16, cutlass::library::LayoutTypeID::kTensorNHWC,
        cutlass::library::NumericTypeID::kF16, cutlass::library::LayoutTypeID::kTensorNHWC,
        cutlass::library::NumericTypeID::kF32, cutlass::library::NumericTypeID::kF32);

    const static bool allow_half_accumulation =
        ParseBooleanFromEnv("ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION", false);

    if (allow_half_accumulation) {
      key.element_accumulator = cutlass::library::NumericTypeID::kF16;
      key.element_compute = cutlass::library::NumericTypeID::kF16;
    }

    cutlass::conv::Conv2dProblemSize problem_size(
        n, h, w, c, k, r, s, p, q, padding_before.at(0), padding_before.at(1), strides.at(0),
        strides.at(1), dilation_rate.at(0), dilation_rate.at(1),
        cutlass::conv::Mode::kCrossCorrelation);
    cutlass::library::Conv2dConfiguration configuraion;
    configuraion.split_k_mode = cutlass::conv::SplitKMode::kSerial;
    configuraion.problem_size = problem_size;
    configuraion.stride_a = {c, w * c, h * w * c};
    configuraion.stride_b = {c, s * c, r * s * c};
    configuraion.stride_c = {0, 0, 0};

    cutlass::library::ConvArguments arguments;
    arguments.A = in->dptr();
    arguments.B = weight->dptr();
    arguments.reordered_B = nullptr;
    if (bias == nullptr) {
      arguments.C = nullptr;
    } else {
      arguments.C = bias->dptr();
    }
    arguments.D = out->mut_dptr();

    union SP {
      float f;
      half h;
    };

    SP alpha;
    SP beta;

    if (allow_half_accumulation) {
      alpha.h = static_cast<half>(1.0F);
      if (bias == nullptr) {
        beta.h = static_cast<half>(0.0F);
      } else {
        beta.h = static_cast<half>(1.0F);
      }
    } else {
      alpha.f = 1.0F;
      if (bias == nullptr) {
        beta.f = 0.0F;
      } else {
        beta.f = 1.0F;
      }
    }
    arguments.alpha = &alpha;
    arguments.beta = &beta;
    arguments.pointer_mode = cutlass::library::ScalarPointerMode::kHost;

    const cutlass::library::Operation* operation =
        FindConv2dOperation(stream, key, configuraion, arguments, tmp_buffer->mut_dptr(),
                            tmp_buffer->shape_view().elem_cnt());

    const size_t host_workspace_size = operation->get_host_workspace_size(&configuraion);
    std::vector<uint8_t> host_workspace(host_workspace_size, 0);
    auto init_status = operation->initialize(&configuraion, host_workspace.data(),
                                             tmp_buffer->mut_dptr(), stream->cuda_stream());
    CHECK(init_status == cutlass::Status::kSuccess);
    auto run_status = operation->run(&arguments, host_workspace.data(), tmp_buffer->mut_dptr(),
                                     stream->cuda_stream());
    CHECK(run_status == cutlass::Status::kSuccess);
  }
};

REGISTER_USER_KERNEL("conv2d")
    .SetCreateFn<Conv2dCutlassKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     && (user_op::HobAttr<std::string>("data_format") == "channels_last")
                     && (user_op::HobAttr<int32_t>("groups") == 1)
                     && (user_op::HobDataType("in", 0) == DataType::kFloat16)
                     && (user_op::HobTrue()
                         == ParseBooleanFromEnv("ONEFLOW_KERENL_CONV_ENABLE_CUTLASS_IMPL", false)))
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {
      // use static workspace size
      return 128 * 1024 * 1024;
    })
    .SetPriority(user_op::kKernelPriorityOptimized);

}  // namespace

}  // namespace oneflow

#endif  // WITH_CUTLASS
