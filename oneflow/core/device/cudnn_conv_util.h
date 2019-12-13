#ifndef ONEFLOW_CORE_DEVICE_CUDNN_CONV_UTIL_H_
#define ONEFLOW_CORE_DEVICE_CUDNN_CONV_UTIL_H_

#ifdef WITH_CUDA

#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/operator/conv_op.h"

namespace oneflow {

struct CudnnConvParams {
  static constexpr size_t max_dim = 3;

  cudnnDataType_t data_type;
  int x_dim[2 + max_dim];
  int x_stride[2 + max_dim];
  int weight_dim[2 + max_dim];
  int padding[max_dim];
  int stride[max_dim];
  int dilation[max_dim];
  bool deterministic;
  bool heuristic;
};

bool operator==(const CudnnConvParams& a, const CudnnConvParams& b);

struct CudnnConvArgs final {
  cudnnHandle_t handle;
  CudnnConvParams params;
  CudnnTensorDesc xdesc;
  CudnnTensorDesc ydesc;
  CudnnFilterDesc wdesc;
  CudnnConvDesc cdesc;
  int x_ndims;
  int y_ndims;
  int w_ndims;
  void* x_dptr;
  void* y_dptr;
  void* w_dptr;
  void* work_space;
  size_t ws_size;
  bool need_destroy_handle;
  bool need_free_memory;

  OF_DISALLOW_COPY_AND_MOVE(CudnnConvArgs);
  CudnnConvArgs(const PbMessage& conf, const BlobDesc* x, const BlobDesc* y, const BlobDesc* w,
                size_t max_ws_size, bool deterministic, bool heuristic,
                const bool enable_true_half);
  CudnnConvArgs(const PbMessage& conf, cudnnHandle_t handle, const Blob* x, const Blob* y,
                const Blob* w, Blob* buf, bool deterministic, bool heuristic,
                const bool enable_true_half);
  ~CudnnConvArgs();
};

template<typename perf_t>
struct CudnnConvAlgorithmSearch;

cudnnStatus_t GetConvWorkspaceSize(const CudnnConvArgs& args, cudnnConvolutionFwdAlgo_t algo,
                                   size_t* sz);
cudnnStatus_t GetConvWorkspaceSize(const CudnnConvArgs& args, cudnnConvolutionBwdDataAlgo_t algo,
                                   size_t* sz);
cudnnStatus_t GetConvWorkspaceSize(const CudnnConvArgs& args, cudnnConvolutionBwdFilterAlgo_t algo,
                                   size_t* sz);

template<typename perf_t>
std::shared_ptr<perf_t> FindCudnnConvAlgorithm(const CudnnConvArgs& args);

}  // namespace oneflow

namespace std {

// Hashing machinery for Params
// see https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
template<>
struct hash<oneflow::CudnnConvParams> final {
  // Params must be a POD because we read out its memory
  // contenst as char* when hashing
  static_assert(std::is_pod<oneflow::CudnnConvParams>::value, "CudnnConvParams is not POD");

  size_t operator()(const oneflow::CudnnConvParams& params) const {
    const auto* ptr = reinterpret_cast<const uint8_t*>(&params);
    uint32_t value = 0x811C9DC5;
    for (int i = 0; i < (int)sizeof(oneflow::CudnnConvParams); ++i) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return (size_t)value;
  }
};

}  // namespace std

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_DEVICE_CUDNN_CONV_UTIL_H_
