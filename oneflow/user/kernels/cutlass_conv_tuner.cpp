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

#include "oneflow/user/kernels/cutlass_conv_tuner.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/job/lazy_mode.h"
#include <cutlass/library/handle.h>
#include <cutlass/library/library.h>
#include <cutlass/library/singleton.h>

namespace oneflow {

namespace {

bool IsWeakerAlginOperation(const cutlass::library::Operation* lhs,
                            const cutlass::library::Operation* rhs) {
  const char* lhs_name = lhs->description().name;
  const char* rhs_name = rhs->description().name;
  const size_t len = std::strlen(lhs_name);
  const size_t suffix_len = std::strlen("align8");
  if (std::strlen(rhs_name) != len) { return false; }
  if (len < suffix_len) { return false; }
  const size_t prefix_len = len - suffix_len;
  if (std::strncmp(lhs_name, rhs_name, prefix_len) != 0) { return false; }
  const auto& HasLegalSuffix = [&](const char* str) {
    if (std::strncmp(str + prefix_len, "align", std::strlen("align")) != 0) { return false; }
    const char align = str[len - 1];
    return align == '8' || align == '4' || align == '2' || align == '1';
  };
  if ((!HasLegalSuffix(lhs_name)) || (!HasLegalSuffix(rhs_name))) { return false; }
  return lhs_name[len - 1] < rhs_name[len - 1];
}

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

};  // namespace

using CacheMap = std::unordered_map<Conv2dOperationCacheKey, const cutlass::library::Operation*,
                                    Conv2dOperationCacheKeyHasher>;
struct CutlassConvTuner::Impl {
  std::mutex mutex;
  std::unordered_map<int, CacheMap> cache;

  const cutlass::library::Operation* FindConv2dOperation(
      ep::CudaStream* stream, cutlass::library::ConvFunctionalKey functional_key,
      const cutlass::library::Conv2dConfiguration& configuraion,
      const cutlass::library::ConvArguments& arguments, void* workspace, size_t workspace_size);

  const cutlass::library::Operation* GetConv2dOperation(
      const std::string& name, ep::CudaStream* stream,
      cutlass::library::ConvFunctionalKey functional_key,
      const cutlass::library::Conv2dConfiguration& configuraion,
      const cutlass::library::ConvArguments& arguments, void* workspace, size_t workspace_size);
};

const cutlass::library::Operation* CutlassConvTuner::Impl::FindConv2dOperation(
    ep::CudaStream* stream, cutlass::library::ConvFunctionalKey functional_key,
    const cutlass::library::Conv2dConfiguration& configuraion,
    const cutlass::library::ConvArguments& arguments, void* workspace, size_t workspace_size) {
  int dev = 0;
  OF_CUDA_CHECK(cudaGetDevice(&dev));
  Conv2dOperationCacheKey cache_key(functional_key, configuraion, arguments);
  {
    std::lock_guard<std::mutex> lock(mutex);
    const auto& device_cache = cache[dev];
    const auto& it = device_cache.find(cache_key);
    if (it != device_cache.end()) { return it->second; }
  }

  constexpr int turing_warmup_iters = 2;
  constexpr int turing_iters = 5;
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
    std::map<std::string, const cutlass::library::Operation*, std::greater<std::string>> operations;
    for (auto operation : pair.second) {
      operations.emplace(operation->description().name, operation);
    }
    const cutlass::library::Operation* prev_operation = nullptr;
    for (const auto& name_operation : operations) {
      const cutlass::library::Operation* operation = name_operation.second;
      if (prev_operation != nullptr && IsWeakerAlginOperation(operation, prev_operation)) {
        continue;
      }
      if (operation->description().tile_description.minimum_compute_capability * 10
              > stream->cuda_arch()
          || operation->description().tile_description.maximum_compute_capability * 10
                 < stream->cuda_arch()) {
        continue;
      }
      auto status = operation->can_implement(&configuraion, &arguments);
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
      prev_operation = operation;
      if (fastest_operation == nullptr || time < fastest_time) {
        fastest_operation = operation;
        fastest_time = time;
      }
    }
  }
  OF_CUDA_CHECK(cudaEventDestroy(start));
  OF_CUDA_CHECK(cudaEventDestroy(end));
  if (fastest_operation != nullptr) {
    VLOG(3) << "Fastest: " << fastest_operation->description().name << " " << fastest_time;
    {
      std::lock_guard<std::mutex> lock(mutex);
      cache[dev][cache_key] = fastest_operation;
    }
  }
  return fastest_operation;
}

const cutlass::library::Operation* CutlassConvTuner::Impl::GetConv2dOperation(
    const std::string& name, ep::CudaStream* stream,
    cutlass::library::ConvFunctionalKey functional_key,
    const cutlass::library::Conv2dConfiguration& configuraion,
    const cutlass::library::ConvArguments& arguments, void* workspace, size_t workspace_size) {
  int dev = 0;
  OF_CUDA_CHECK(cudaGetDevice(&dev));
  const auto& operations_map_it =
      cutlass::library::Singleton::get().operation_table.conv2d_operations.find(functional_key);
  if (operations_map_it
      == cutlass::library::Singleton::get().operation_table.conv2d_operations.cend()) {
    return nullptr;
  }
  const cutlass::library::ConvOperationVectorMap& operations_map = operations_map_it->second;
  for (const auto& pair : operations_map) {
    for (auto operation : pair.second) {
      if (name != operation->description().name) { continue; }
      if (operation->description().tile_description.minimum_compute_capability * 10
              > stream->cuda_arch()
          || operation->description().tile_description.maximum_compute_capability * 10
                 < stream->cuda_arch()) {
        continue;
      }
      auto status = operation->can_implement(&configuraion, &arguments);
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
      return operation;
    }
  }
  return nullptr;
}

CutlassConvTuner::CutlassConvTuner() { impl_.reset(new Impl()); }

const CutlassConvTuner& CutlassConvTuner::Get() {
  static CutlassConvTuner instance;
  return instance;
}

const cutlass::library::Operation* CutlassConvTuner::FindConv2dOperation(
    ep::CudaStream* stream, cutlass::library::ConvFunctionalKey functional_key,
    const cutlass::library::Conv2dConfiguration& configuraion,
    const cutlass::library::ConvArguments& arguments, void* workspace,
    size_t workspace_size) const {
  return impl_->FindConv2dOperation(stream, functional_key, configuraion, arguments, workspace,
                                    workspace_size);
}

const cutlass::library::Operation* CutlassConvTuner::GetConv2dOperation(
    const std::string& name, ep::CudaStream* stream,
    cutlass::library::ConvFunctionalKey functional_key,
    const cutlass::library::Conv2dConfiguration& configuraion,
    const cutlass::library::ConvArguments& arguments, void* workspace,
    size_t workspace_size) const {
  return impl_->GetConv2dOperation(name, stream, functional_key, configuraion, arguments, workspace,
                                   workspace_size);
}

}  // namespace oneflow

#endif  // WITH_CUTLASS
