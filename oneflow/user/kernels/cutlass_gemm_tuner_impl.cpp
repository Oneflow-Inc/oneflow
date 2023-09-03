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

#include "oneflow/user/kernels/cutlass_gemm_tuner_impl.h"

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include <cutlass/library/library.h>
#include <cutlass/library/singleton.h>

#include "oneflow/user/kernels/cutlass_gemm_operation_cache_key.h"
#ifdef WITH_CUTLASS_EXTENSION
#include <cutlass/library/cutlass_extension_library.h>
#include <cutlass/library/cutlass_extension_singleton.h>
#endif  // WITH_CUTLASS_EXTENSION

namespace oneflow {

namespace {

bool IsWeakerAlginOperation(const cutlass::library::Operation* lhs,
                            const cutlass::library::Operation* rhs) {
  const std::string lhs_name = lhs->description().name;
  const std::string rhs_name = rhs->description().name;
  size_t lhs_pos = lhs_name.rfind("align");
  if (lhs_pos == std::string::npos) { return false; }
  size_t rhs_pos = rhs_name.rfind("align");
  if (rhs_pos == std::string::npos) { return false; }
  if (lhs_name.substr(0, lhs_pos) != rhs_name.substr(0, rhs_pos)) { return false; }
  size_t align_len = std::strlen("align");
  int lhs_alignment = std::atoi(lhs_name.substr(lhs_pos + align_len).c_str());
  int rhs_alignment = std::atoi(rhs_name.substr(rhs_pos + align_len).c_str());
  return lhs_alignment < rhs_alignment;
}

size_t GetTensorSize(cutlass::library::NumericTypeID element, cutlass::library::LayoutTypeID layout,
                     const int row, const int col, const int ldc) {
  const size_t element_size = cutlass::library::sizeof_bits(element) / 8;
  size_t capacity = 0;
  if (layout == cutlass::library::LayoutTypeID::kRowMajor) {
    capacity = row * ldc;
  } else if (layout == cutlass::library::LayoutTypeID::kColumnMajor) {
    capacity = ldc * col;
  } else {
    UNIMPLEMENTED();
  }
  return capacity * element_size;
}

template<typename Singleton, typename Configuration, typename Arguments>
const cutlass::library::Operation* FindFastestOperation(
    const Singleton* singleton, const cutlass::library::GemmFunctionalKey& functional_key,
    const Configuration& configuraion, const Arguments& arguments, void* workspace,
    size_t workspace_size, cudaStream_t stream, int cuda_arch) {
  constexpr int turing_warmup_iters = 2;
  constexpr int turing_iters = 5;
  cudaEvent_t start{};
  cudaEvent_t end{};
  OF_CUDA_CHECK(cudaEventCreate(&start));
  OF_CUDA_CHECK(cudaEventCreate(&end));
  const cutlass::library::Operation* fastest_operation = nullptr;
  float fastest_time = 0;
  const auto& operations_map = [&]() {
    const auto& it = singleton->operation_table.gemm_operations.find(functional_key);
    CHECK(it != singleton->operation_table.gemm_operations.cend());
    return it->second;
  }();

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
      if (operation->description().tile_description.minimum_compute_capability * 10 > cuda_arch
          || operation->description().tile_description.maximum_compute_capability * 10
                 < cuda_arch) {
        continue;
      }
      auto status = operation->can_implement(&configuraion, &arguments);
      if (status != cutlass::Status::kSuccess) { continue; }
      const size_t host_workspace_size = operation->get_host_workspace_size(&configuraion);
      const size_t device_workspace_size = operation->get_device_workspace_size(&configuraion);
      if (device_workspace_size > workspace_size) { continue; }
      std::vector<uint8_t> host_workspace(host_workspace_size, 0);
      if (operation->initialize(&configuraion, host_workspace.data(), workspace, stream)
          != cutlass::Status::kSuccess) {
        continue;
      }

      const auto Run = [&]() {
        auto init_status =
            operation->initialize(&configuraion, host_workspace.data(), workspace, stream);
        CHECK(init_status == cutlass::Status::kSuccess);
        auto run_status = operation->run(&arguments, host_workspace.data(), workspace, stream);
        CHECK(run_status == cutlass::Status::kSuccess);
      };
      OF_CUDA_CHECK(cudaStreamSynchronize(stream));
      for (int i = 0; i < turing_warmup_iters; ++i) { Run(); }
      OF_CUDA_CHECK(cudaEventRecord(start, stream));
      for (int i = 0; i < turing_iters; ++i) { Run(); }
      OF_CUDA_CHECK(cudaEventRecord(end, stream));
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
  VLOG(3) << "Fastest: " << fastest_operation->description().name << " " << fastest_time;
  return fastest_operation;
}

template<typename Singleton, typename Configuration, typename Arguments>
const cutlass::library::Operation* GetOperation(
    const Singleton* singleton, const std::string& name,
    const cutlass::library::GemmFunctionalKey& functional_key, const Configuration& configuraion,
    const Arguments& arguments, void* workspace, size_t workspace_size, cudaStream_t stream,
    int cuda_arch) {
  const auto& it = singleton->operation_table.gemm_operations.find(functional_key);
  if (it == singleton->operation_table.gemm_operations.cend()) { return nullptr; }
  const cutlass::library::GemmOperationVectorMap& operations_map = it->second;
  for (const auto& pair : operations_map) {
    for (auto operation : pair.second) {
      if (name != operation->description().name) { continue; }
      if (operation->description().tile_description.minimum_compute_capability * 10 > cuda_arch
          || operation->description().tile_description.maximum_compute_capability * 10
                 < cuda_arch) {
        continue;
      }
      auto status = operation->can_implement(&configuraion, &arguments);
      if (status != cutlass::Status::kSuccess) { continue; }
      const size_t host_workspace_size = operation->get_host_workspace_size(&configuraion);
      const size_t device_workspace_size = operation->get_device_workspace_size(&configuraion);
      if (device_workspace_size > workspace_size) { continue; }
      std::vector<uint8_t> host_workspace(host_workspace_size, 0);
      if (operation->initialize(&configuraion, host_workspace.data(), workspace, stream)
          != cutlass::Status::kSuccess) {
        continue;
      }
      return operation;
    }
  }
  return nullptr;
}

}  // namespace

#ifdef WITH_CUTLASS_EXTENSION
template<>
class CutlassGemmTunerImpl<cutlass::library::GemmScaleBiasFusionConfiguration,
                           cutlass::library::GemmScaleBiasFusionArguments> {
 public:
  using CacheMap = std::unordered_map<GemmOperationCacheKey, const cutlass::library::Operation*,
                                      GemmOperationCacheKeyHasher>;

  CutlassGemmTunerImpl() {
    singleton = &cutlass::library::CutlassExtensionSingleton::get(
        cutlass::library::SingletonKind::kGemmScaleBiasFusion);
    residual_singleton = &cutlass::library::CutlassExtensionSingleton::get(
        cutlass::library::SingletonKind::kGemmScaleBiasResidualFusion);
  }

  const cutlass::library::Operation* Find(
      ep::CudaStream* stream, cutlass::library::GemmFunctionalKey functional_key,
      const cutlass::library::GemmScaleBiasFusionConfiguration& configuraion,
      const cutlass::library::GemmScaleBiasFusionArguments& arguments, void* workspace,
      size_t workspace_size);

  const cutlass::library::Operation* Get(
      const std::string& name, ep::CudaStream* stream,
      cutlass::library::GemmFunctionalKey functional_key,
      const cutlass::library::GemmScaleBiasFusionConfiguration& configuraion,
      const cutlass::library::GemmScaleBiasFusionArguments& arguments, void* workspace,
      size_t workspace_size);

 private:
  std::mutex mutex;
  std::unordered_map<int, CacheMap> cache;
  const cutlass::library::CutlassExtensionSingleton* singleton;
  const cutlass::library::CutlassExtensionSingleton* residual_singleton;
};

const cutlass::library::Operation*
CutlassGemmTunerImpl<cutlass::library::GemmScaleBiasFusionConfiguration,
                     cutlass::library::GemmScaleBiasFusionArguments>::
    Find(ep::CudaStream* stream, cutlass::library::GemmFunctionalKey functional_key,
         const cutlass::library::GemmScaleBiasFusionConfiguration& configuraion,
         const cutlass::library::GemmScaleBiasFusionArguments& arguments, void* workspace,
         size_t workspace_size) {
  int dev = 0;
  OF_CUDA_CHECK(cudaGetDevice(&dev));
  GemmOperationCacheKey cache_key(functional_key, configuraion, arguments);
  {
    std::lock_guard<std::mutex> lock(mutex);
    const auto& device_cache = cache[dev];
    const auto& it = device_cache.find(cache_key);
    if (it != device_cache.end()) { return it->second; }
  }
  cutlass::library::GemmScaleBiasFusionArguments benchmark_arguments = arguments;
  void* benchmark_workspace = workspace;
  cudaStream_t benchmark_stream = stream->cuda_stream();
#ifdef WITH_CUDA_GRAPHS
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  if (stream->IsGraphCapturing()) {
    OF_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
    OF_CUDA_CHECK(cudaStreamCreate(&benchmark_stream));
    OF_CUDA_CHECK(cudaMalloc(&benchmark_workspace, workspace_size));
    const size_t a_size = GetTensorSize(functional_key.element_A, functional_key.layout_A,
                                        configuraion.problem_size.m(),
                                        configuraion.problem_size.k(), configuraion.lda);
    OF_CUDA_CHECK(cudaMalloc(&benchmark_arguments.A, a_size));
    const size_t b_size = GetTensorSize(functional_key.element_B, functional_key.layout_B,
                                        configuraion.problem_size.k(),
                                        configuraion.problem_size.m(), configuraion.ldb);
    OF_CUDA_CHECK(cudaMalloc(&benchmark_arguments.B, b_size));

    if (benchmark_arguments.Scale != nullptr) {
      const size_t scale_size = configuraion.problem_size.n()
                                * cutlass::library::sizeof_bits(functional_key.element_D) / 8;
      OF_CUDA_CHECK(cudaMalloc(&benchmark_arguments.Scale, scale_size));
    }
    if (benchmark_arguments.Bias != nullptr) {
      const size_t bias_size = configuraion.problem_size.n()
                               * cutlass::library::sizeof_bits(functional_key.element_D) / 8;
      OF_CUDA_CHECK(cudaMalloc(&benchmark_arguments.Bias, bias_size));
    }
    if (benchmark_arguments.Residual != nullptr) {
      const size_t residual_size = GetTensorSize(functional_key.element_D, functional_key.layout_D,
                                                 configuraion.problem_size.m(),
                                                 configuraion.problem_size.n(), configuraion.ldr);
      OF_CUDA_CHECK(cudaMalloc(&benchmark_arguments.Residual, residual_size));
    }
    const size_t d_size = GetTensorSize(functional_key.element_D, functional_key.layout_D,
                                        configuraion.problem_size.m(),
                                        configuraion.problem_size.n(), configuraion.ldd);
    OF_CUDA_CHECK(cudaMalloc(&benchmark_arguments.D, d_size));
  }
#endif  // WITH_CUDA_GRAPHS

  const cutlass::library::Operation* fastest_operation =
      FindFastestOperation((benchmark_arguments.Residual ? residual_singleton : singleton),
                           functional_key, configuraion, benchmark_arguments, benchmark_workspace,
                           workspace_size, benchmark_stream, stream->cuda_arch());

#ifdef WITH_CUDA_GRAPHS
  if (stream->IsGraphCapturing()) {
    OF_CUDA_CHECK(cudaStreamSynchronize(benchmark_stream));
    OF_CUDA_CHECK(cudaStreamDestroy(benchmark_stream));
    OF_CUDA_CHECK(cudaFree(const_cast<void*>(benchmark_arguments.A)));
    OF_CUDA_CHECK(cudaFree(const_cast<void*>(benchmark_arguments.B)));
    OF_CUDA_CHECK(cudaFree(const_cast<void*>(benchmark_arguments.Scale)));
    OF_CUDA_CHECK(cudaFree(const_cast<void*>(benchmark_arguments.Bias)));
    OF_CUDA_CHECK(cudaFree(const_cast<void*>(benchmark_arguments.Residual)));
    OF_CUDA_CHECK(cudaFree(benchmark_arguments.D));
    OF_CUDA_CHECK(cudaFree(benchmark_workspace));
    OF_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  }
#endif  // WITH_CUDA_GRAPHS
  if (fastest_operation != nullptr) {
    std::lock_guard<std::mutex> lock(mutex);
    cache[dev][cache_key] = fastest_operation;
  }
  return fastest_operation;
}

const cutlass::library::Operation*
CutlassGemmTunerImpl<cutlass::library::GemmScaleBiasFusionConfiguration,
                     cutlass::library::GemmScaleBiasFusionArguments>::
    Get(const std::string& name, ep::CudaStream* stream,
        cutlass::library::GemmFunctionalKey functional_key,
        const cutlass::library::GemmScaleBiasFusionConfiguration& configuraion,
        const cutlass::library::GemmScaleBiasFusionArguments& arguments, void* workspace,
        size_t workspace_size) {
  int dev = 0;
  OF_CUDA_CHECK(cudaGetDevice(&dev));
  return GetOperation((arguments.Residual ? residual_singleton : singleton), name, functional_key,
                      configuraion, arguments, workspace, workspace_size, stream->cuda_stream(),
                      stream->cuda_arch());
}
#endif  // WITH_CUTLASS_EXTENSION

template<typename Configuration, typename Arguments>
CutlassGemmTunerImpl<Configuration, Arguments>* GetCutlassGemmTunerImpl() {
  static CutlassGemmTunerImpl<Configuration, Arguments> impl;
  return &impl;
}

#ifdef WITH_CUTLASS_EXTENSION
template CutlassGemmTunerImpl<cutlass::library::GemmScaleBiasFusionConfiguration,
                              cutlass::library::GemmScaleBiasFusionArguments>*
GetCutlassGemmTunerImpl<cutlass::library::GemmScaleBiasFusionConfiguration,
                        cutlass::library::GemmScaleBiasFusionArguments>();
#endif  // WITH_CUTLASS_EXTENSION

}  // namespace oneflow

#endif  // WITH_CUTLASS
