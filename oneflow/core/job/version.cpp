#include "oneflow/core/job/version.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <cudnn.h>
#include <nccl.h>
#endif  // WITH_CUDA

namespace oneflow {

namespace {

std::string GetCudaVersionString(int version) {
  return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
}

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

  do {
    int cudnn_version_major;
    int cudnn_version_minor;
    int cudnn_version_path;
    {
      cudnnStatus_t status =
          cudnnGetProperty(libraryPropertyType::MAJOR_VERSION, &cudnn_version_major);
      if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "Failed to get cuDNN version: " << cudnnGetErrorString(status);
        break;
      }
    }
    {
      cudnnStatus_t status =
          cudnnGetProperty(libraryPropertyType::MINOR_VERSION, &cudnn_version_minor);
      if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "Failed to get cuDNN version: " << cudnnGetErrorString(status);
        break;
      }
    }
    {
      cudnnStatus_t status =
          cudnnGetProperty(libraryPropertyType::PATCH_LEVEL, &cudnn_version_path);
      if (status != CUDNN_STATUS_SUCCESS) {
        LOG(ERROR) << "Failed to get cuDNN version: " << cudnnGetErrorString(status);
        break;
      }
    }
    LOG(INFO) << "cuDNN version: " << cudnn_version_major << "." << cudnn_version_minor << "."
              << cudnn_version_path;
  } while (false);

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
