#ifndef ONEFLOW_CORE_DEVICE_CUDNN_CONV_UTIL_H_
#define ONEFLOW_CORE_DEVICE_CUDNN_CONV_UTIL_H_

#ifdef WITH_CUDA

#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job.pb.h"

namespace oneflow {

class CudnnConvDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnConvDesc);
  CudnnConvDesc() = delete;
  ~CudnnConvDesc();

  CudnnConvDesc(const DataType& data_type, const ShapeView& in_blob_shape,
                const PbMessage& conv_conf);

  const cudnnConvolutionDescriptor_t& Get() const { return val_; }

 private:
  cudnnConvolutionDescriptor_t val_;
};

struct CudnnConvParams {
  static constexpr size_t kTensorMaxDims = 5;
  static constexpr size_t kConvMaxDims = 3;

  cudnnDataType_t x_data_type;
  cudnnDataType_t w_data_type;
  cudnnDataType_t y_data_type;
  cudnnDataType_t data_type;
  cudnnTensorFormat_t w_format;
  int x_ndim;
  int w_ndim;
  int y_ndim;
  int x_dims[kTensorMaxDims];
  int x_strides[kTensorMaxDims];
  int y_dims[kTensorMaxDims];
  int y_strides[kTensorMaxDims];
  int w_dims[kTensorMaxDims];
  int padding[kConvMaxDims];
  int stride[kConvMaxDims];
  int dilation[kConvMaxDims];
  size_t max_ws_size;
};

bool operator==(const CudnnConvParams& a, const CudnnConvParams& b);

struct CudnnConvArgs final {
  CudnnConvParams params;
  CudnnTensorDesc xdesc;
  CudnnTensorDesc ydesc;
  CudnnFilterDesc wdesc;
  CudnnConvDesc cdesc;
  cudnnHandle_t handle;
  void* x_dptr;
  void* y_dptr;
  void* w_dptr;
  void* ws_dptr;
  size_t max_ws_size;
  bool need_create_handle;
  bool need_free_memory;
  const JobConfigProto& conf;

  OF_DISALLOW_COPY_AND_MOVE(CudnnConvArgs);
  CudnnConvArgs(const JobConfigProto& job_conf, const PbMessage& conv_conf, const BlobDesc* x,
                const BlobDesc* y, const BlobDesc* w, size_t max_ws_size);
  CudnnConvArgs(const JobConfigProto& job_conf, const PbMessage& conv_conf, cudnnHandle_t handle,
                const Blob* x, const Blob* y, const Blob* w, Blob* buf);
  ~CudnnConvArgs();
  void AllocateIfNeed();
};

DataType GetConvDescDataType(DataType data_type, bool pseudo_half);
size_t GetByteSizeOfCudnnDataType(cudnnDataType_t data_type);

template<typename perf_t>
struct CudnnConvAlgorithmSearch;

cudnnStatus_t GetConvWorkspaceSize(const CudnnConvArgs& args, cudnnConvolutionFwdAlgo_t algo,
                                   size_t* sz);
cudnnStatus_t GetConvWorkspaceSize(const CudnnConvArgs& args, cudnnConvolutionBwdDataAlgo_t algo,
                                   size_t* sz);
cudnnStatus_t GetConvWorkspaceSize(const CudnnConvArgs& args, cudnnConvolutionBwdFilterAlgo_t algo,
                                   size_t* sz);

template<typename perf_t>
perf_t FindCudnnConvAlgorithm(CudnnConvArgs* args);

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
