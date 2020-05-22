#include "oneflow/core/job/version.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <cudnn.h>
#include <nccl.h>
#endif  // WITH_CUDA

namespace oneflow {

namespace {

#ifdef WITH_CUDA

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
  int version_major;
  int version_minor;
  int version_patch;
  if (!GetCudnnVersion(libraryPropertyType::MAJOR_VERSION, &version_major)) { return false; }
  if (!GetCudnnVersion(libraryPropertyType::MINOR_VERSION, &version_minor)) { return false; }
  if (!GetCudnnVersion(libraryPropertyType::PATCH_LEVEL, &version_patch)) { return false; }
  *version = std::to_string(version_major) + "." + std::to_string(version_minor) + "."
             + std::to_string(version_patch);
  return true;
}

#endif  // WITH_CUDA

}  // namespace

void DumpVersionInfo() {
#ifdef WITH_GIT_VERSION
  LOG(INFO) << "OneFlow git version: " << GetOneFlowGitVersion();
#endif  // WITH_GIT_VERSION

#ifdef WITH_CUDA
  {
    int cuda_driver_version;
    cudaError_t err = cudaDriverGetVersion(&cuda_driver_version);
    if (err == cudaSuccess) {
      LOG(INFO) << "CUDA driver version: " << GetCudaVersionString(cuda_driver_version);
    } else {
      LOG(ERROR) << "Failed to get cuda driver version: " << cudaGetErrorString(err);
    }
  }

  {
    int cuda_runtime_version;
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
    int nccl_version;
    ncclResult_t result = ncclGetVersion(&nccl_version);
    if (result == ncclSuccess) {
      int nccl_version_major = nccl_version / 1000;
      int nccl_version_minor = (nccl_version % 1000) / 100;
      int nccl_version_patch = (nccl_version % 100);
      LOG(INFO) << "NCCL version: " << nccl_version_major << "." << nccl_version_minor << "."
                << nccl_version_patch;
    } else {
      LOG(ERROR) << "Failed to get NCCL version: " << ncclGetErrorString(result);
    }
  }
#endif  // WITH_CUDA
}

}  // namespace oneflow
