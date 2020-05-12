#ifdef WITH_CUDA
#include "oneflow/core/device/cudnn_conv_util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/cached_caller.h"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

namespace {

template<typename algo_t>
algo_t GetDefaultAlgo();

template<>
cudnnConvolutionFwdAlgo_t GetDefaultAlgo<cudnnConvolutionFwdAlgo_t>() {
  return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
}

template<>
cudnnConvolutionBwdDataAlgo_t GetDefaultAlgo<cudnnConvolutionBwdDataAlgo_t>() {
  return CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
}

template<>
cudnnConvolutionBwdFilterAlgo_t GetDefaultAlgo<cudnnConvolutionBwdFilterAlgo_t>() {
  return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
}

size_t ByteSize4Tensor(const int* dims, int ndim, cudnnDataType_t data_type) {
  size_t byte_size = GetCudnnDataTypeByteSize(data_type);
  FOR_RANGE(int, i, 0, ndim) { byte_size *= dims[i]; }
  return byte_size;
}

template<typename perf_t, typename algo_t>
void SetAlgo4Perf(const CudnnConvArgs& args, CudnnConvResource* res, perf_t* algo_perf,
                  algo_t algo) {
  algo_perf->algo = algo;
  if (args.params.data_type == CUDNN_DATA_HALF) {
    algo_perf->mathType = CUDNN_TENSOR_OP_MATH;
  } else {
    algo_perf->mathType = CUDNN_DEFAULT_MATH;
  }
  CudaCheck(GetCudnnConvWorkspaceSize(args, res, algo_perf->algo, &(algo_perf->memory)));
  algo_perf->status = CUDNN_STATUS_SUCCESS;
}

template<typename perf_t>
perf_t GetBestAlgorithm(const CudnnConvArgs& args, CudnnConvResource* res,
                        const std::vector<perf_t>& perf_vec) {
  using algo_t = decltype(std::declval<perf_t>().algo);
  if (perf_vec.size() == 0) {
    LOG(WARNING) << "There is no result with "
                 << (args.heuristic ? "heuristic searching way." : "exhaustive searching way.")
                 << " (max_workspace_size=" << args.params.max_ws_size << ")"
                 << " Use default algo(" << GetDefaultAlgo<algo_t>() << ") instead.";
    perf_t perf;
    SetAlgo4Perf(args, res, &perf, GetDefaultAlgo<algo_t>());
    return perf;
  }

  int found_algo_idx = -1;
  FOR_RANGE(size_t, i, 0, perf_vec.size()) {
    // Note: Shouldn't all returned results be successful?
    CHECK_EQ(perf_vec[i].status, CUDNN_STATUS_SUCCESS);
    if (perf_vec[i].memory > args.params.max_ws_size) { continue; }
    if (args.deterministic && perf_vec[i].determinism == CUDNN_NON_DETERMINISTIC) { continue; }
    found_algo_idx = i;
    break;
  }

  if (found_algo_idx == -1) {
    LOG(WARNING) << "Cannot find any algorithm meets requirements (max_workspace_size="
                 << args.params.max_ws_size << ", determinism=" << args.deterministic << ") using "
                 << (args.heuristic ? "heuristic searching way." : "exhaustive searching way.")
                 << " Using default algo(" << GetDefaultAlgo<algo_t>() << ") instead.";
    perf_t algo_perf;
    SetAlgo4Perf(args, res, &algo_perf, GetDefaultAlgo<algo_t>());
    return algo_perf;
  }

  if (found_algo_idx != 0) {
    LOG(WARNING) << "Currently available alogrithm (algo=" << perf_vec[found_algo_idx].algo
                 << ", require memory=" << perf_vec[found_algo_idx].memory
                 << ", idx=" << found_algo_idx
                 << ") meeting requirments (max_workspace_size=" << args.params.max_ws_size
                 << ", determinism=" << args.deterministic
                 << ") is not fastest. Fastest algorithm (" << perf_vec[0].algo
                 << ") requires memory " << perf_vec[0].memory;
  }

#if CUDNN_VERSION < 7500
  // google [blacklist fft algorithms for strided dgrad]
  if (std::is_same<decltype(perf_vec[found_algo_idx].algo), cudnnConvolutionBwdDataAlgo_t>::value) {
    int stride_dim = args.params.x_ndim - 2;
    bool blacklist =
        std::any_of(std::begin(args.params.stride), std::begin(args.params.stride) + stride_dim,
                    [](int n) { return n != 1; });
    if (blacklist
        && (static_cast<cudnnConvolutionBwdDataAlgo_t>(perf_vec[found_algo_idx].algo)
                == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING
            || static_cast<cudnnConvolutionBwdDataAlgo_t>(perf_vec[found_algo_idx].algo)
                   == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)) {
      perf_t algo_perf;
      SetAlgo4Perf(args, res, &algo_perf, GetDefaultAlgo<algo_t>());
      return algo_perf;
    }
  }
#endif

  return perf_vec.at(found_algo_idx);
}

}  // namespace

CudnnConvDesc::~CudnnConvDesc() { CudaCheck(cudnnDestroyConvolutionDescriptor(val_)); }

CudnnConvDesc::CudnnConvDesc(const DataType& data_type, const ShapeView& in_blob_shape,
                             const PbMessage& conv_conf) {
  int32_t opkernel_dim = in_blob_shape.NumAxes() - 2;
  CudaCheck(cudnnCreateConvolutionDescriptor(&val_));
  std::vector<int32_t> pad_large_side;
  GetConvOutAndPad(in_blob_shape, conv_conf, nullptr, nullptr, &pad_large_side);
  const PbRf<int32_t>& strides = GetPbRfFromPbMessage<int32_t>(conv_conf, "strides");
  const PbRf<int32_t>& dilation_rate = GetPbRfFromPbMessage<int32_t>(conv_conf, "dilation_rate");
  if (opkernel_dim == 2) {
    CudaCheck(cudnnSetConvolution2dDescriptor(val_, pad_large_side[0], pad_large_side[1],
                                              strides.Get(0), strides.Get(1), dilation_rate.Get(0),
                                              dilation_rate.Get(1), CUDNN_CROSS_CORRELATION,
                                              GetCudnnDataType(data_type)));
  } else if (opkernel_dim == 1) {
    CudaCheck(cudnnSetConvolution2dDescriptor(val_, pad_large_side[0], 0, strides.Get(0), 1,
                                              dilation_rate.Get(0), 1, CUDNN_CROSS_CORRELATION,
                                              GetCudnnDataType(data_type)));
  } else {
    CudaCheck(cudnnSetConvolutionNdDescriptor(
        val_, opkernel_dim, pad_large_side.data(), strides.data(), dilation_rate.data(),
        CUDNN_CROSS_CORRELATION, GetCudnnDataType(data_type)));
  }
  const int32_t groups = GetValFromPbMessage<int32_t>(conv_conf, "groups");
  if (groups != 1) { CudaCheck(cudnnSetConvolutionGroupCount(val_, groups)); }
}

CudnnConvDesc::CudnnConvDesc(const DataType& data_type, const ShapeView& in_blob_shape,
                             const user_op::UserOpConfWrapper& conv_conf) {
  int32_t opkernel_dim = in_blob_shape.NumAxes() - 2;
  CudaCheck(cudnnCreateConvolutionDescriptor(&val_));
  std::vector<int32_t> pad_large_side;
  GetConvOutAndPad(in_blob_shape, conv_conf, nullptr, nullptr, &pad_large_side);
  const auto& strides = conv_conf.attr<std::vector<int32_t>>("strides");
  const auto& dilation_rate = conv_conf.attr<std::vector<int32_t>>("dilation_rate");
  if (opkernel_dim == 2) {
    CudaCheck(cudnnSetConvolution2dDescriptor(val_, pad_large_side[0], pad_large_side[1],
                                              strides.at(0), strides.at(1), dilation_rate.at(0),
                                              dilation_rate.at(1), CUDNN_CROSS_CORRELATION,
                                              GetCudnnDataType(data_type)));
  } else if (opkernel_dim == 1) {
    CudaCheck(cudnnSetConvolution2dDescriptor(val_, pad_large_side[0], 0, strides.at(0), 1,
                                              dilation_rate.at(0), 1, CUDNN_CROSS_CORRELATION,
                                              GetCudnnDataType(data_type)));
  } else {
    CudaCheck(cudnnSetConvolutionNdDescriptor(
        val_, opkernel_dim, pad_large_side.data(), strides.data(), dilation_rate.data(),
        CUDNN_CROSS_CORRELATION, GetCudnnDataType(data_type)));
  }
  const int32_t groups = conv_conf.attr<int32_t>("groups");
  if (groups != 1) { CudaCheck(cudnnSetConvolutionGroupCount(val_, groups)); }
}

CudnnConvArgs::CudnnConvArgs(const PbMessage& conv_conf, DataType x_data_type,
                             const ShapeView& x_shape, DataType w_data_type,
                             const ShapeView& w_shape, DataType y_data_type,
                             const ShapeView& y_shape, const std::string& data_format,
                             size_t max_workspace_size, bool heuristic_search,
                             bool use_deterministic_algo_only, bool enable_pseudo_half)
    : xdesc(x_data_type, x_shape, data_format),
      ydesc(y_data_type, y_shape, data_format),
      wdesc(w_data_type, w_shape, data_format),
      cdesc(GetConvDescDataType(x_data_type, enable_pseudo_half), x_shape, conv_conf),
      heuristic(heuristic_search),
      deterministic(use_deterministic_algo_only) {
  std::memset(&params, 0, sizeof(CudnnConvParams));
  CudaCheck(cudnnGetTensorNdDescriptor(xdesc.Get(), CudnnConvParams::kTensorMaxDims,
                                       &params.x_data_type, &params.x_ndim, params.x_dims,
                                       params.x_strides));
  CudaCheck(cudnnGetTensorNdDescriptor(ydesc.Get(), CudnnConvParams::kTensorMaxDims,
                                       &params.y_data_type, &params.y_ndim, params.y_dims,
                                       params.y_strides));
  CudaCheck(cudnnGetFilterNdDescriptor(wdesc.Get(), CudnnConvParams::kTensorMaxDims,
                                       &params.w_data_type, &params.w_format, &params.w_ndim,
                                       params.w_dims));
  cudnnConvolutionMode_t mode;
  int conv_dim_size = 0;
  CudaCheck(cudnnGetConvolutionNdDescriptor(cdesc.Get(), CudnnConvParams::kConvMaxDims,
                                            &conv_dim_size, params.padding, params.stride,
                                            params.dilation, &mode, &params.data_type));
  CHECK_EQ(params.x_data_type, params.w_data_type);
  CHECK_EQ(params.x_ndim, params.w_ndim);
  CHECK_EQ(conv_dim_size + 2, params.x_ndim);
  CudaCheck(cudnnGetConvolutionGroupCount(cdesc.Get(), &params.groups));
  params.max_ws_size = max_workspace_size;
}

CudnnConvArgs::CudnnConvArgs(const user_op::UserOpConfWrapper& conv_conf, DataType x_data_type,
                             const ShapeView& x_shape, DataType w_data_type,
                             const ShapeView& w_shape, DataType y_data_type,
                             const ShapeView& y_shape, const std::string& data_format,
                             size_t max_workspace_size, bool heuristic_search,
                             bool use_deterministic_algo_only, bool enable_pseudo_half)
    : xdesc(x_data_type, x_shape, data_format),
      ydesc(y_data_type, y_shape, data_format),
      wdesc(w_data_type, w_shape, data_format),
      cdesc(GetConvDescDataType(x_data_type, enable_pseudo_half), x_shape, conv_conf),
      heuristic(heuristic_search),
      deterministic(use_deterministic_algo_only) {
  std::memset(&params, 0, sizeof(CudnnConvParams));
  CudaCheck(cudnnGetTensorNdDescriptor(xdesc.Get(), CudnnConvParams::kTensorMaxDims,
                                       &params.x_data_type, &params.x_ndim, params.x_dims,
                                       params.x_strides));
  CudaCheck(cudnnGetTensorNdDescriptor(ydesc.Get(), CudnnConvParams::kTensorMaxDims,
                                       &params.y_data_type, &params.y_ndim, params.y_dims,
                                       params.y_strides));
  CudaCheck(cudnnGetFilterNdDescriptor(wdesc.Get(), CudnnConvParams::kTensorMaxDims,
                                       &params.w_data_type, &params.w_format, &params.w_ndim,
                                       params.w_dims));
  cudnnConvolutionMode_t mode;
  int conv_dim_size = 0;
  CudaCheck(cudnnGetConvolutionNdDescriptor(cdesc.Get(), CudnnConvParams::kConvMaxDims,
                                            &conv_dim_size, params.padding, params.stride,
                                            params.dilation, &mode, &params.data_type));
  CHECK_EQ(params.x_data_type, params.w_data_type);
  CHECK_EQ(params.x_ndim, params.w_ndim);
  CHECK_EQ(conv_dim_size + 2, params.x_ndim);
  CudaCheck(cudnnGetConvolutionGroupCount(cdesc.Get(), &params.groups));
  params.max_ws_size = max_workspace_size;
}

ManagedCudnnConvResource::ManagedCudnnConvResource(const CudnnConvArgs& args)
    : handle_(nullptr), x_dptr_(nullptr), w_dptr_(nullptr), y_dptr_(nullptr), ws_dptr_(nullptr) {
  x_byte_size_ = ByteSize4Tensor(args.params.x_dims, args.params.x_ndim, args.params.x_data_type);
  w_byte_size_ = ByteSize4Tensor(args.params.w_dims, args.params.w_ndim, args.params.w_data_type);
  y_byte_size_ = ByteSize4Tensor(args.params.y_dims, args.params.y_ndim, args.params.y_data_type);
  ws_byte_size_ = args.params.max_ws_size;
}

ManagedCudnnConvResource::~ManagedCudnnConvResource() {
  if (handle_ != nullptr) { CudaCheck(cudnnDestroy(handle_)); }
  if (x_dptr_ != nullptr) { CudaCheck(cudaFree(x_dptr_)); }
  if (w_dptr_ != nullptr) { CudaCheck(cudaFree(w_dptr_)); }
  if (y_dptr_ != nullptr) { CudaCheck(cudaFree(y_dptr_)); }
  if (ws_dptr_ != nullptr) { CudaCheck(cudaFree(ws_dptr_)); }
}

cudnnHandle_t ManagedCudnnConvResource::cudnn_handle() {
  if (handle_ == nullptr) { CudaCheck(cudnnCreate(&handle_)); }
  return handle_;
}

void* ManagedCudnnConvResource::x_mut_dptr() {
  if (x_dptr_ == nullptr) { CudaCheck(cudaMalloc(&x_dptr_, x_byte_size_)); }
  return x_dptr_;
}

void* ManagedCudnnConvResource::w_mut_dptr() {
  if (w_dptr_ == nullptr) { CudaCheck(cudaMalloc(&w_dptr_, w_byte_size_)); }
  return w_dptr_;
}

void* ManagedCudnnConvResource::y_mut_dptr() {
  if (y_dptr_ == nullptr) { CudaCheck(cudaMalloc(&y_dptr_, y_byte_size_)); }
  return y_dptr_;
}

const void* ManagedCudnnConvResource::x_const_dptr() const {
  return const_cast<ManagedCudnnConvResource*>(this)->x_mut_dptr();
}

const void* ManagedCudnnConvResource::w_const_dptr() const {
  return const_cast<ManagedCudnnConvResource*>(this)->w_mut_dptr();
}

const void* ManagedCudnnConvResource::y_const_dptr() const {
  return const_cast<ManagedCudnnConvResource*>(this)->y_mut_dptr();
}

void* ManagedCudnnConvResource::ws_dptr() {
  if (ws_dptr_ == nullptr) { CudaCheck(cudaMalloc(&ws_dptr_, ws_byte_size_)); }
  return ws_dptr_;
}

bool operator==(const CudnnConvParams& a, const CudnnConvParams& b) {
  auto ptr1 = reinterpret_cast<const uint8_t*>(&a);
  auto ptr2 = reinterpret_cast<const uint8_t*>(&b);
  return memcmp(ptr1, ptr2, sizeof(CudnnConvParams)) == 0;
}

DataType GetConvDescDataType(DataType data_type, bool pseudo_half) {
  return (data_type == DataType::kFloat16 && pseudo_half) ? DataType::kFloat : data_type;
}

cudnnStatus_t GetCudnnConvWorkspaceSize(const CudnnConvArgs& args, CudnnConvResource* res,
                                        cudnnConvolutionFwdAlgo_t algo, size_t* sz) {
  return cudnnGetConvolutionForwardWorkspaceSize(res->cudnn_handle(), args.xdesc.Get(),
                                                 args.wdesc.Get(), args.cdesc.Get(),
                                                 args.ydesc.Get(), algo, sz);
}

cudnnStatus_t GetCudnnConvWorkspaceSize(const CudnnConvArgs& args, CudnnConvResource* res,
                                        cudnnConvolutionBwdDataAlgo_t algo, size_t* sz) {
  return cudnnGetConvolutionBackwardDataWorkspaceSize(res->cudnn_handle(), args.wdesc.Get(),
                                                      args.ydesc.Get(), args.cdesc.Get(),
                                                      args.xdesc.Get(), algo, sz);
}

cudnnStatus_t GetCudnnConvWorkspaceSize(const CudnnConvArgs& args, CudnnConvResource* res,
                                        cudnnConvolutionBwdFilterAlgo_t algo, size_t* sz) {
  return cudnnGetConvolutionBackwardFilterWorkspaceSize(res->cudnn_handle(), args.xdesc.Get(),
                                                        args.ydesc.Get(), args.cdesc.Get(),
                                                        args.wdesc.Get(), algo, sz);
}

template<>
struct CudnnConvAlgorithmSearch<cudnnConvolutionFwdAlgoPerf_t> {
  using perf_t = cudnnConvolutionFwdAlgoPerf_t;

  static int GetAlgoMaxCount(CudnnConvResource* res) {
    int max_algo_cnt = 0;
    CudaCheck(cudnnGetConvolutionForwardAlgorithmMaxCount(res->cudnn_handle(), &max_algo_cnt));
    return max_algo_cnt;
  }

  static void HeuristicSearch(const CudnnConvArgs& args, CudnnConvResource* res,
                              std::vector<perf_t>* perf_vec) {
    int found_algo_cnt = 0;
    perf_vec->resize(GetAlgoMaxCount(res));
    CudaCheck(cudnnGetConvolutionForwardAlgorithm_v7(
        res->cudnn_handle(), args.xdesc.Get(), args.wdesc.Get(), args.cdesc.Get(), args.ydesc.Get(),
        perf_vec->capacity(), &found_algo_cnt, perf_vec->data()));
    // vector::resize does not affect the first found_algo_cnt elements.
    perf_vec->resize(found_algo_cnt);
  }

  static void ExhaustiveSearch(const CudnnConvArgs& args, CudnnConvResource* res,
                               std::vector<perf_t>* perf_vec) {
    int found_algo_cnt = 0;
    perf_vec->resize(GetAlgoMaxCount(res));
    CudaCheck(cudnnFindConvolutionForwardAlgorithmEx(
        res->cudnn_handle(), args.xdesc.Get(), res->x_const_dptr(), args.wdesc.Get(),
        res->w_const_dptr(), args.cdesc.Get(), args.ydesc.Get(), res->y_mut_dptr(),
        perf_vec->capacity(), &found_algo_cnt, perf_vec->data(), res->ws_dptr(),
        args.params.max_ws_size));
    // vector::resize does not affect the first found_algo_cnt elements.
    perf_vec->resize(found_algo_cnt);
  }
};

template<>
struct CudnnConvAlgorithmSearch<cudnnConvolutionBwdDataAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdDataAlgoPerf_t;

  static int GetAlgoMaxCount(CudnnConvResource* res) {
    int max_algo_cnt = 0;
    CudaCheck(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(res->cudnn_handle(), &max_algo_cnt));
    return max_algo_cnt;
  }

  static void HeuristicSearch(const CudnnConvArgs& args, CudnnConvResource* res,
                              std::vector<perf_t>* perf_vec) {
    int found_algo_cnt = 0;
    perf_vec->resize(GetAlgoMaxCount(res));
    CudaCheck(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        res->cudnn_handle(), args.wdesc.Get(), args.ydesc.Get(), args.cdesc.Get(), args.xdesc.Get(),
        perf_vec->capacity(), &found_algo_cnt, perf_vec->data()));
    // vector::resize does not affect the first found_algo_cnt elements.
    perf_vec->resize(found_algo_cnt);
  }

  static void ExhaustiveSearch(const CudnnConvArgs& args, CudnnConvResource* res,
                               std::vector<perf_t>* perf_vec) {
    int found_algo_cnt = 0;
    perf_vec->resize(GetAlgoMaxCount(res));
    CudaCheck(cudnnFindConvolutionBackwardDataAlgorithmEx(
        res->cudnn_handle(), args.wdesc.Get(), res->w_const_dptr(), args.ydesc.Get(),
        res->y_const_dptr(), args.cdesc.Get(), args.xdesc.Get(), res->x_mut_dptr(),
        perf_vec->capacity(), &found_algo_cnt, perf_vec->data(), res->ws_dptr(),
        args.params.max_ws_size));
    // vector::resize does not affect the first found_algo_cnt elements.
    perf_vec->resize(found_algo_cnt);
  }
};

template<>
struct CudnnConvAlgorithmSearch<cudnnConvolutionBwdFilterAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdFilterAlgoPerf_t;

  static int GetAlgoMaxCount(CudnnConvResource* res) {
    int max_algo_cnt = 0;
    CudaCheck(
        cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(res->cudnn_handle(), &max_algo_cnt));
    return max_algo_cnt;
  }

  static void HeuristicSearch(const CudnnConvArgs& args, CudnnConvResource* res,
                              std::vector<perf_t>* perf_vec) {
    int found_algo_cnt = 0;
    perf_vec->resize(GetAlgoMaxCount(res));
    CudaCheck(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        res->cudnn_handle(), args.xdesc.Get(), args.ydesc.Get(), args.cdesc.Get(), args.wdesc.Get(),
        perf_vec->capacity(), &found_algo_cnt, perf_vec->data()));
    // vector::resize does not affect the first found_algo_cnt elements.
    perf_vec->resize(found_algo_cnt);
  }

  static void ExhaustiveSearch(const CudnnConvArgs& args, CudnnConvResource* res,
                               std::vector<perf_t>* perf_vec) {
    int found_algo_cnt = 0;
    perf_vec->resize(GetAlgoMaxCount(res));
    CudaCheck(cudnnFindConvolutionBackwardFilterAlgorithmEx(
        res->cudnn_handle(), args.xdesc.Get(), res->x_const_dptr(), args.ydesc.Get(),
        res->y_const_dptr(), args.cdesc.Get(), args.wdesc.Get(), res->w_mut_dptr(),
        perf_vec->capacity(), &found_algo_cnt, perf_vec->data(), res->ws_dptr(),
        args.params.max_ws_size));
    // vector::resize does not affect the first found_algo_cnt elements.
    perf_vec->resize(found_algo_cnt);
  }
};

template<typename perf_t>
perf_t FindCudnnConvAlgorithm(CudnnConvArgs* args) {
  ManagedCudnnConvResource res(*args);
  return FindCudnnConvAlgorithmWithResource<perf_t>(args, &res);
}

template<typename perf_t>
perf_t FindCudnnConvAlgorithmWithResource(CudnnConvArgs* args, CudnnConvResource* res) {
  auto Infer = [args, res](const CudnnConvParams& params) {
    std::vector<perf_t> perf_vec;
    if (args->heuristic) {
      CudnnConvAlgorithmSearch<perf_t>::HeuristicSearch(*args, res, &perf_vec);
    } else {
      CudnnConvAlgorithmSearch<perf_t>::ExhaustiveSearch(*args, res, &perf_vec);
    }
    return GetBestAlgorithm<perf_t>(*args, res, perf_vec);
  };
  size_t cache_size = Global<ResourceDesc>::Get()->thread_local_cache_max_size();
  return ThreadLocalCachedCall(cache_size, Infer, args->params);
}

template<typename perf_t, typename algo_t>
perf_t GetCudnnConvAlgorithmPerference(CudnnConvArgs* args, algo_t algo) {
  ManagedCudnnConvResource res(*args);
  return GetCudnnConvAlgorithmPerferenceWithResource<perf_t>(args, &res, algo);
}

template<typename perf_t, typename algo_t>
perf_t GetCudnnConvAlgorithmPerferenceWithResource(CudnnConvArgs* args, CudnnConvResource* res,
                                                   algo_t algo) {
  perf_t perf;
  SetAlgo4Perf(*args, res, &perf, algo);
  return perf;
}

#define EXPLICIT_INSTANTIAT_CUDNN_CONV_ALGORITHM_INTERFACE(perf_t)                        \
  template perf_t FindCudnnConvAlgorithm(CudnnConvArgs*);                                 \
  template perf_t FindCudnnConvAlgorithmWithResource(CudnnConvArgs*, CudnnConvResource*); \
  template perf_t GetCudnnConvAlgorithmPerference(CudnnConvArgs*,                         \
                                                  decltype(std::declval<perf_t>().algo)); \
  template perf_t GetCudnnConvAlgorithmPerferenceWithResource(                            \
      CudnnConvArgs*, CudnnConvResource*, decltype(std::declval<perf_t>().algo));

EXPLICIT_INSTANTIAT_CUDNN_CONV_ALGORITHM_INTERFACE(cudnnConvolutionFwdAlgoPerf_t)
EXPLICIT_INSTANTIAT_CUDNN_CONV_ALGORITHM_INTERFACE(cudnnConvolutionBwdDataAlgoPerf_t)
EXPLICIT_INSTANTIAT_CUDNN_CONV_ALGORITHM_INTERFACE(cudnnConvolutionBwdFilterAlgoPerf_t)

}  // namespace oneflow

#endif  // WITH_CUDA
