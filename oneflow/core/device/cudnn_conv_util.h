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
#ifndef ONEFLOW_CORE_DEVICE_CUDNN_CONV_UTIL_H_
#define ONEFLOW_CORE_DEVICE_CUDNN_CONV_UTIL_H_

#ifdef WITH_CUDA

#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

namespace user_op {

class UserOpConfWrapper;

}  // namespace user_op

class CudnnConvDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnConvDesc);
  CudnnConvDesc() = delete;
  ~CudnnConvDesc();

  CudnnConvDesc(const DataType compute_type, const DataType data_type,
                const ShapeView& in_blob_shape, const user_op::UserOpConfWrapper& conv_conf);

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
  int groups;
};

struct CudnnConvArgs final {
  CudnnConvParams params;
  CudnnTensorDesc xdesc;
  CudnnTensorDesc ydesc;
  CudnnFilterDesc wdesc;
  CudnnConvDesc cdesc;
  bool heuristic;
  bool deterministic;

  OF_DISALLOW_COPY_AND_MOVE(CudnnConvArgs);
  CudnnConvArgs(const user_op::UserOpConfWrapper& conv_conf, DataType x_data_type,
                const ShapeView& x_shape, DataType w_data_type, const ShapeView& w_shape,
                DataType y_data_type, const ShapeView& y_shape, const std::string& data_format,
                size_t max_workspace_size, bool heuristic_search, bool use_deterministic_algo_only,
                bool enable_pseudo_half);
};

class CudnnConvResource {
 public:
  CudnnConvResource() = default;
  virtual ~CudnnConvResource() = default;
  virtual cudnnHandle_t cudnn_handle() = 0;
  virtual void* w_mut_dptr() = 0;
  virtual void* x_mut_dptr() = 0;
  virtual void* y_mut_dptr() = 0;
  virtual const void* w_const_dptr() const = 0;
  virtual const void* x_const_dptr() const = 0;
  virtual const void* y_const_dptr() const = 0;
  virtual void* ws_dptr() = 0;
};

class AllocatedCudnnConvResource final : public CudnnConvResource {
 public:
  AllocatedCudnnConvResource(cudnnHandle_t handle, void* x_dptr, void* w_dptr, void* y_dptr,
                             void* ws_dptr)
      : handle_(handle), x_dptr_(x_dptr), w_dptr_(w_dptr), y_dptr_(y_dptr), ws_dptr_(ws_dptr) {}
  ~AllocatedCudnnConvResource() = default;
  cudnnHandle_t cudnn_handle() override { return handle_; }
  const void* x_const_dptr() const override { return x_dptr_; }
  const void* w_const_dptr() const override { return w_dptr_; }
  const void* y_const_dptr() const override { return y_dptr_; }
  void* x_mut_dptr() override { return x_dptr_; }
  void* w_mut_dptr() override { return w_dptr_; }
  void* y_mut_dptr() override { return y_dptr_; }
  void* ws_dptr() override { return ws_dptr_; }

 private:
  cudnnHandle_t handle_;
  void* x_dptr_;
  void* w_dptr_;
  void* y_dptr_;
  void* ws_dptr_;
};

class ManagedCudnnConvResource final : public CudnnConvResource {
 public:
  ManagedCudnnConvResource(const CudnnConvArgs& args);
  ~ManagedCudnnConvResource() override;
  cudnnHandle_t cudnn_handle() override;
  void* x_mut_dptr() override;
  void* w_mut_dptr() override;
  void* y_mut_dptr() override;
  const void* x_const_dptr() const override;
  const void* w_const_dptr() const override;
  const void* y_const_dptr() const override;
  void* ws_dptr() override;

 private:
  cudnnHandle_t handle_;
  void* x_dptr_;
  void* w_dptr_;
  void* y_dptr_;
  void* ws_dptr_;
  size_t x_byte_size_;
  size_t w_byte_size_;
  size_t y_byte_size_;
  size_t ws_byte_size_;
};

bool operator==(const CudnnConvParams& a, const CudnnConvParams& b);
DataType GetConvDescDataType(DataType data_type, bool pseudo_half);

template<typename perf_t>
struct CudnnConvAlgorithmSearch;

cudnnStatus_t GetCudnnConvWorkspaceSize(const CudnnConvArgs& args, CudnnConvResource* res,
                                        cudnnConvolutionFwdAlgo_t algo, size_t* sz);
cudnnStatus_t GetCudnnConvWorkspaceSize(const CudnnConvArgs& args, CudnnConvResource* res,
                                        cudnnConvolutionBwdDataAlgo_t algo, size_t* sz);
cudnnStatus_t GetCudnnConvWorkspaceSize(const CudnnConvArgs& args, CudnnConvResource* res,
                                        cudnnConvolutionBwdFilterAlgo_t algo, size_t* sz);

template<typename perf_t>
perf_t FindCudnnConvAlgorithm(CudnnConvArgs* args);

template<typename perf_t>
perf_t FindCudnnConvAlgorithmWithResource(CudnnConvArgs* args, CudnnConvResource* res);

template<typename perf_t, typename algo_t>
perf_t GetCudnnConvAlgorithmPerference(CudnnConvArgs* args, algo_t algo);

template<typename perf_t, typename algo_t>
perf_t GetCudnnConvAlgorithmPerferenceWithResource(CudnnConvArgs* args, CudnnConvResource* res,
                                                   algo_t algo);

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

namespace oneflow {

class CudnnConvAlgoCache final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnConvAlgoCache);
  CudnnConvAlgoCache() = default;
  ~CudnnConvAlgoCache() = default;

  template<typename perf_t>
  using WorkspaceSizeAndPerfT = std::pair<size_t, perf_t>;
  template<typename perf_t>
  using Store = HashMap<CudnnConvParams, std::list<WorkspaceSizeAndPerfT<perf_t>>>;

  template<typename perf_t>
  perf_t Remember(const CudnnConvParams& params,
                  const std::function<perf_t(const CudnnConvParams& param)>& InferFn);

 private:
  Store<cudnnConvolutionFwdAlgoPerf_t> fwd_algo_store_;
  std::mutex fwd_algo_store_mutex_;
  Store<cudnnConvolutionBwdDataAlgoPerf_t> bwd_data_algo_store_;
  std::mutex bwd_data_algo_store_mutex_;
  Store<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filter_algo_store_;
  std::mutex bwd_filter_algo_cache_mutex_;
};

}  // namespace oneflow

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_DEVICE_CUDNN_CONV_UTIL_H_
