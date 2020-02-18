#ifdef WITH_CUDA
#include "oneflow/core/device/cudnn_conv_util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/cached_caller.h"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/core/job/resource_desc.h"

namespace oneflow {

namespace {

template<typename perf_t, typename algo_t>
void SetAlgo4Perf(const CudnnConvArgs& args, perf_t* algo_perf, algo_t algo) {
  algo_perf->algo = algo;
  if (args.params.data_type == CUDNN_DATA_HALF) {
    algo_perf->mathType = CUDNN_TENSOR_OP_MATH;
  } else {
    algo_perf->mathType = CUDNN_DEFAULT_MATH;
  }
  CudaCheck(GetConvWorkspaceSize(args, algo_perf->algo, &(algo_perf->memory)));
}

template<typename perf_t>
perf_t GetBestAlgorithm(const CudnnConvArgs& args, const std::vector<perf_t>& perf_vec) {
  int found_algo_idx = -1;
  bool find_deterministic = args.conf.cudnn_conv_use_deterministic_algo_only();
  FOR_RANGE(size_t, i, 0, perf_vec.size()) {
    // Note: Shouldn't all returned results be successful?
    CHECK_EQ(perf_vec[i].status, CUDNN_STATUS_SUCCESS);
    if (perf_vec[i].memory > args.params.max_ws_size) { continue; }
    if (find_deterministic && perf_vec[i].determinism == CUDNN_NON_DETERMINISTIC) { continue; }
    found_algo_idx = i;
    break;
  }

#if CUDNN_VERSION < 7500
  // google [blacklist fft algorithms for strided dgrad]
  if (std::is_same<decltype(perf_vec[found_algo_idx].algo), cudnnConvolutionBwdDataAlgo_t>::value) {
    int stride_dim = args.params.x_ndim - 2;
    bool blacklist =
        std::any_of(std::begin(args.params.stride), std::begin(args.params.stride) + stride_dim,
                    [](int n) { return n != 1; });
    if (blacklist
        && (perf_vec[found_algo_idx].algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING
            || perf_vec[found_algo_idx].algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)) {
      perf_t algo_perf;
      SetAlgo4Perf(args, &algo_perf, CudnnConvAlgorithmSearch<perf_t>::DEFAULT_ALGO);
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
}

CudnnConvArgs::CudnnConvArgs(const JobConfigProto& job_conf, const PbMessage& conv_conf,
                             const BlobDesc* x, const BlobDesc* y, const BlobDesc* w,
                             size_t max_ws_size)
    : xdesc(x->data_type(), x->shape(), GetValFromPbMessage<std::string>(conv_conf, "data_format")),
      ydesc(y->data_type(), y->shape(), GetValFromPbMessage<std::string>(conv_conf, "data_format")),
      wdesc(w->data_type(), w->shape(), GetValFromPbMessage<std::string>(conv_conf, "data_format")),
      cdesc(GetConvDescDataType(x->data_type(), job_conf.cudnn_conv_use_pseudo_half()), x->shape(),
            conv_conf),
      x_dptr(nullptr),
      y_dptr(nullptr),
      w_dptr(nullptr),
      ws_dptr(nullptr),
      need_create_handle(true),
      need_free_memory(false),
      conf(job_conf) {
  CudaCheck(cudnnCreate(&handle));
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
  params.max_ws_size = max_ws_size;
}

CudnnConvArgs::CudnnConvArgs(const JobConfigProto& job_conf, const PbMessage& conv_conf,
                             cudnnHandle_t handle, const Blob* x, const Blob* y, const Blob* w,
                             Blob* buf)
    : xdesc(x->data_type(), x->shape(), GetValFromPbMessage<std::string>(conv_conf, "data_format")),
      ydesc(y->data_type(), y->shape(), GetValFromPbMessage<std::string>(conv_conf, "data_format")),
      wdesc(w->data_type(), w->shape(), GetValFromPbMessage<std::string>(conv_conf, "data_format")),
      cdesc(GetConvDescDataType(x->data_type(), job_conf.cudnn_conv_use_pseudo_half()), x->shape(),
            conv_conf),
      handle(handle),
      x_dptr(const_cast<void*>(x->dptr())),
      y_dptr(const_cast<void*>(y->dptr())),
      w_dptr(const_cast<void*>(w->dptr())),
      ws_dptr(buf ? buf->mut_dptr() : nullptr),
      need_create_handle(false),
      need_free_memory(false),
      conf(job_conf) {
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
  params.max_ws_size = buf ? buf->ByteSizeOfBlobBody() : 0;
}

CudnnConvArgs::~CudnnConvArgs() {
  if (need_free_memory) {
    CudaCheck(cudaFree(x_dptr));
    CudaCheck(cudaFree(w_dptr));
    CudaCheck(cudaFree(y_dptr));
    CudaCheck(cudaFree(ws_dptr));
  }
  if (need_create_handle) { CudaCheck(cudnnDestroy(handle)); }
}

void CudnnConvArgs::AllocateIfNeed() {
  if (!conf.cudnn_conv_heuristic_search_algo() && x_dptr == nullptr && y_dptr == nullptr
      && w_dptr == nullptr && ws_dptr == nullptr) {
    size_t x_byte_size = GetByteSizeOfCudnnDataType(params.x_data_type);
    FOR_RANGE(int, i, 0, params.x_ndim) { x_byte_size *= params.x_dims[i]; }
    CudaCheck(cudaMalloc(&x_dptr, RoundUp(x_byte_size, kCudaMemAllocAlignSize)));
    size_t w_byte_size = GetByteSizeOfCudnnDataType(params.w_data_type);
    FOR_RANGE(int, i, 0, params.w_ndim) { w_byte_size *= params.w_dims[i]; }
    CudaCheck(cudaMalloc(&w_dptr, RoundUp(w_byte_size, kCudaMemAllocAlignSize)));
    size_t y_byte_size = GetByteSizeOfCudnnDataType(params.y_data_type);
    FOR_RANGE(int, i, 0, params.y_ndim) { y_byte_size *= params.y_dims[i]; }
    CudaCheck(cudaMalloc(&y_dptr, RoundUp(y_byte_size, kCudaMemAllocAlignSize)));
    CudaCheck(cudaMalloc(&ws_dptr, params.max_ws_size));
    need_free_memory = true;
  }
}

bool operator==(const CudnnConvParams& a, const CudnnConvParams& b) {
  auto ptr1 = reinterpret_cast<const uint8_t*>(&a);
  auto ptr2 = reinterpret_cast<const uint8_t*>(&b);
  return memcmp(ptr1, ptr2, sizeof(CudnnConvParams)) == 0;
}

DataType GetConvDescDataType(DataType data_type, bool pseudo_half) {
  return (data_type == DataType::kFloat16 && pseudo_half) ? DataType::kFloat : data_type;
}

size_t GetByteSizeOfCudnnDataType(cudnnDataType_t data_type) {
  size_t byte_size = 0;
  switch (data_type) {
    case CUDNN_DATA_FLOAT:
    case CUDNN_DATA_INT32:
    case CUDNN_DATA_INT8x4:
    case CUDNN_DATA_UINT8x4: {
      byte_size = 4;
    }
    case CUDNN_DATA_DOUBLE: {
      byte_size = 8;
    }
    case CUDNN_DATA_HALF: {
      byte_size = 2;
    }
    case CUDNN_DATA_INT8:
    case CUDNN_DATA_UINT8: {
      byte_size = 1;
    }
    case CUDNN_DATA_INT8x32: {
      byte_size = 32;
    }
    default: { UNIMPLEMENTED(); }
  }
  return byte_size;
}

cudnnStatus_t GetConvWorkspaceSize(const CudnnConvArgs& args, cudnnConvolutionFwdAlgo_t algo,
                                   size_t* sz) {
  return cudnnGetConvolutionForwardWorkspaceSize(args.handle, args.xdesc.Get(), args.wdesc.Get(),
                                                 args.cdesc.Get(), args.ydesc.Get(), algo, sz);
}

cudnnStatus_t GetConvWorkspaceSize(const CudnnConvArgs& args, cudnnConvolutionBwdDataAlgo_t algo,
                                   size_t* sz) {
  return cudnnGetConvolutionBackwardDataWorkspaceSize(args.handle, args.wdesc.Get(),
                                                      args.ydesc.Get(), args.cdesc.Get(),
                                                      args.xdesc.Get(), algo, sz);
}

cudnnStatus_t GetConvWorkspaceSize(const CudnnConvArgs& args, cudnnConvolutionBwdFilterAlgo_t algo,
                                   size_t* sz) {
  return cudnnGetConvolutionBackwardFilterWorkspaceSize(args.handle, args.xdesc.Get(),
                                                        args.ydesc.Get(), args.cdesc.Get(),
                                                        args.wdesc.Get(), algo, sz);
}

template<>
struct CudnnConvAlgorithmSearch<cudnnConvolutionFwdAlgoPerf_t> {
  using perf_t = cudnnConvolutionFwdAlgoPerf_t;
  using algo_t = cudnnConvolutionFwdAlgo_t;
  static constexpr algo_t DEFAULT_ALGO = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

  static void FindAlgorithm(const CudnnConvArgs& args, perf_t* algo_perf) {
    if (args.conf.has_cudnn_conv_force_fwd_algo()) {
      SetAlgo4Perf(args, algo_perf, static_cast<algo_t>(args.conf.cudnn_conv_force_fwd_algo()));
    } else {
      int max_algo_cnt = 0;
      int found_algo_cnt = 0;
      CudaCheck(cudnnGetConvolutionForwardAlgorithmMaxCount(args.handle, &max_algo_cnt));
      std::vector<perf_t> perf_vec(max_algo_cnt);
      if (args.conf.cudnn_conv_heuristic_search_algo()) {
        CudaCheck(cudnnGetConvolutionForwardAlgorithm_v7(
            args.handle, args.xdesc.Get(), args.wdesc.Get(), args.cdesc.Get(), args.ydesc.Get(),
            max_algo_cnt, &found_algo_cnt, perf_vec.data()));
      } else {
        CudaCheck(cudnnFindConvolutionForwardAlgorithmEx(
            args.handle, args.xdesc.Get(), args.x_dptr, args.wdesc.Get(), args.w_dptr,
            args.cdesc.Get(), args.ydesc.Get(), args.y_dptr, max_algo_cnt, &found_algo_cnt,
            perf_vec.data(), args.ws_dptr, args.params.max_ws_size));
      }
      perf_vec.resize(found_algo_cnt);
      if (perf_vec.size() > 0) {
        *algo_perf = GetBestAlgorithm<perf_t>(args, perf_vec);
      } else {
        SetAlgo4Perf(args, algo_perf, DEFAULT_ALGO);
      }
    }
  }
};

template<>
struct CudnnConvAlgorithmSearch<cudnnConvolutionBwdDataAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdDataAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdDataAlgo_t;
  static constexpr algo_t DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

  static void FindAlgorithm(const CudnnConvArgs& args, perf_t* algo_perf) {
    if (args.conf.has_cudnn_conv_force_bwd_data_algo()) {
      SetAlgo4Perf(args, algo_perf,
                   static_cast<algo_t>(args.conf.cudnn_conv_force_bwd_data_algo()));
    } else {
      int max_algo_cnt = 0;
      int found_algo_cnt = 0;
      CudaCheck(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(args.handle, &max_algo_cnt));
      std::vector<perf_t> perf_vec(max_algo_cnt);
      if (args.conf.cudnn_conv_heuristic_search_algo()) {
        CudaCheck(cudnnGetConvolutionBackwardDataAlgorithm_v7(
            args.handle, args.wdesc.Get(), args.ydesc.Get(), args.cdesc.Get(), args.xdesc.Get(),
            max_algo_cnt, &found_algo_cnt, perf_vec.data()));
      } else {
        CudaCheck(cudnnFindConvolutionBackwardDataAlgorithmEx(
            args.handle, args.wdesc.Get(), args.w_dptr, args.ydesc.Get(), args.y_dptr,
            args.cdesc.Get(), args.xdesc.Get(), args.x_dptr, max_algo_cnt, &found_algo_cnt,
            perf_vec.data(), args.ws_dptr, args.params.max_ws_size));
      }
      perf_vec.resize(found_algo_cnt);
      if (perf_vec.size() > 0) {
        *algo_perf = GetBestAlgorithm<perf_t>(args, perf_vec);
      } else {
        SetAlgo4Perf(args, algo_perf, DEFAULT_ALGO);
      }
    }
  }
};

template<>
struct CudnnConvAlgorithmSearch<cudnnConvolutionBwdFilterAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdFilterAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdFilterAlgo_t;
  static constexpr algo_t DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

  static void FindAlgorithm(const CudnnConvArgs& args, perf_t* algo_perf) {
    if (args.conf.has_cudnn_conv_force_bwd_filter_algo()) {
      SetAlgo4Perf(args, algo_perf,
                   static_cast<algo_t>(args.conf.cudnn_conv_force_bwd_filter_algo()));
    } else {
      int max_algo_cnt = 0;
      int found_algo_cnt = 0;
      CudaCheck(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(args.handle, &max_algo_cnt));
      std::vector<perf_t> perf_vec(max_algo_cnt);
      if (args.conf.cudnn_conv_heuristic_search_algo()) {
        CudaCheck(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
            args.handle, args.xdesc.Get(), args.ydesc.Get(), args.cdesc.Get(), args.wdesc.Get(),
            max_algo_cnt, &found_algo_cnt, perf_vec.data()));
      } else {
        CudaCheck(cudnnFindConvolutionBackwardFilterAlgorithmEx(
            args.handle, args.xdesc.Get(), args.x_dptr, args.ydesc.Get(), args.y_dptr,
            args.cdesc.Get(), args.wdesc.Get(), args.w_dptr, max_algo_cnt, &found_algo_cnt,
            perf_vec.data(), args.ws_dptr, args.params.max_ws_size));
      }
      perf_vec.resize(found_algo_cnt);
      if (perf_vec.size() > 0) {
        *algo_perf = GetBestAlgorithm<perf_t>(args, perf_vec);
      } else {
        SetAlgo4Perf(args, algo_perf, DEFAULT_ALGO);
      }
    }
  }
};

template<typename perf_t>
perf_t FindCudnnConvAlgorithm(CudnnConvArgs* args) {
  auto Infer = [args](const CudnnConvParams& params) {
    perf_t perf;
    args->AllocateIfNeed();
    CudnnConvAlgorithmSearch<perf_t>::FindAlgorithm(*args, &perf);
    CHECK_LE(perf.memory, args->params.max_ws_size);
    return perf;
  };
  size_t cache_size = Global<ResourceDesc>::Get()->thread_local_cache_max_size();
  return ThreadLocalCachedCall(cache_size, Infer, args->params);
}

template cudnnConvolutionFwdAlgoPerf_t FindCudnnConvAlgorithm(CudnnConvArgs* args);
template cudnnConvolutionBwdDataAlgoPerf_t FindCudnnConvAlgorithm(CudnnConvArgs* args);
template cudnnConvolutionBwdFilterAlgoPerf_t FindCudnnConvAlgorithm(CudnnConvArgs* args);

}  // namespace oneflow

#endif  // WITH_CUDA
