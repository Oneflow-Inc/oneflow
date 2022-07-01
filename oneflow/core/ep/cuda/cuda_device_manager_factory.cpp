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
#include "oneflow/core/ep/include/device_manager_factory.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/ep/cuda/cuda_device_manager.h"

#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cudnn.h>
#include <nccl.h>

namespace oneflow {

namespace ep {

namespace {

std::string GetCudaVersionString(int version) {
  return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
}

bool GetCudnnVersion(libraryPropertyType type, int* version) {
  cudnnStatus_t status = cudnnGetProperty(type, version);
  if (status == CUDNN_STATUS_SUCCESS) {
    return true;
  } else {
    LOG(ERROR) << "Failed to get cuDNN version: " << cudnnGetErrorString(status);
    return false;
  }
}

bool GetCudnnVersionString(std::string* version) {
  int version_major = 0;
  int version_minor = 0;
  int version_patch = 0;
  if (!GetCudnnVersion(libraryPropertyType::MAJOR_VERSION, &version_major)) { return false; }
  if (!GetCudnnVersion(libraryPropertyType::MINOR_VERSION, &version_minor)) { return false; }
  if (!GetCudnnVersion(libraryPropertyType::PATCH_LEVEL, &version_patch)) { return false; }
  *version = std::to_string(version_major) + "." + std::to_string(version_minor) + "."
             + std::to_string(version_patch);
  return true;
}

void CudaDumpVersionInfo() {
  {
    int cuda_runtime_version = 0;
    cudaError_t err = cudaRuntimeGetVersion(&cuda_runtime_version);
    if (err == cudaSuccess) {
      LOG(INFO) << "CUDA runtime version: " << GetCudaVersionString(cuda_runtime_version);
    } else {
      LOG(ERROR) << "Failed to get cuda runtime version: " << cudaGetErrorString(err);
    }
  }

  {
    std::string cudnn_version_string;
    if (GetCudnnVersionString(&cudnn_version_string)) {
      LOG(INFO) << "cuDNN version: " << cudnn_version_string;
    }
  }

  {
    int nccl_version = 0;
    ncclResult_t result = ncclGetVersion(&nccl_version);
    if (result == ncclSuccess) {
      int nccl_version_major =
          (nccl_version >= 20900) ? (nccl_version / 10000) : (nccl_version / 1000);
      int nccl_version_minor =
          (nccl_version >= 20900) ? (nccl_version % 10000) / 100 : (nccl_version % 1000) / 100;
      int nccl_version_patch = (nccl_version % 100);
      LOG(INFO) << "NCCL version: " << nccl_version_major << "." << nccl_version_minor << "."
                << nccl_version_patch;
    } else {
      LOG(ERROR) << "Failed to get NCCL version: " << ncclGetErrorString(result);
    }
  }
}

class CudaDeviceManagerFactory : public DeviceManagerFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaDeviceManagerFactory);
  CudaDeviceManagerFactory() = default;
  ~CudaDeviceManagerFactory() override = default;

  std::unique_ptr<DeviceManager> NewDeviceManager(DeviceManagerRegistry* registry) override {
    return std::make_unique<CudaDeviceManager>(registry);
  }

  DeviceType device_type() const override { return DeviceType::kCUDA; }

  std::string device_type_name() const override { return "cuda"; }

  void DumpVersionInfo() const override { CudaDumpVersionInfo(); }
};

COMMAND(DeviceManagerRegistry::RegisterDeviceManagerFactory(
    std::make_unique<CudaDeviceManagerFactory>()))

}  // namespace

}  // namespace ep

}  // namespace oneflow

#endif  // WITH_CUDA
