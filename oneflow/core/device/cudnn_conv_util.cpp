#ifdef WITH_CUDA
#include "oneflow/core/device/cudnn_conv_util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/common/cached_caller.h"

namespace oneflow {

namespace {

template<typename perf_t>
perf_t GetBestAlgorithm(const CudnnConvArgs& args, const std::vector<perf_t>& perf_vec) {
  int best_algo_idx;
  bool find_deterministic = false;
  if (args.params.deterministic) {
    // iterate over perf results of all algorithms and find the best deterministic algo
    FOR_RANGE(size_t, i, 0, perf_vec.size()) {
      if (perf_vec[i].status == CUDNN_STATUS_SUCCESS
          && perf_vec[i].determinism == CUDNN_DETERMINISTIC) {
        best_algo_idx = i;
        find_deterministic = true;
        break;
      }
    }
    if (!find_deterministic) {
      LOG(FATAL) << "no deterministic convolution algorithms available in cudnn";
    }
  } else {
    best_algo_idx = 0;
  }

// See Note [blacklist fft algorithms for strided dgrad]
#if CUDNN_VERSION < 7500
  if (std::is_same<decltype(perf_vec[best_algo_idx].algo), cudnnConvolutionBwdDataAlgo_t>::value) {
    int stride_dim = args.x_ndims - 2;
    bool blacklist =
        std::any_of(std::begin(args.params.stride), std::begin(args.params.stride) + stride_dim,
                    [](int n) { return n != 1; });
    if (blacklist
        && (static_cast<cudnnConvolutionBwdDataAlgo_t>(perf_vec[best_algo_idx].algo)
                == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING
            || static_cast<cudnnConvolutionBwdDataAlgo_t>(perf_vec[best_algo_idx].algo)
                   == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)) {
      perf_t algo_perf;
      algo_perf.algo = CudnnConvAlgorithmSearch<perf_t>::DEFAULT_ALGO;
      if (args.params.data_type == CUDNN_DATA_HALF) {
        algo_perf.mathType = CUDNN_TENSOR_OP_MATH;
      } else {
        algo_perf.mathType = CUDNN_DEFAULT_MATH;
      }
      CudaCheck(GetConvWorkspaceSize(args, algo_perf.algo, &(algo_perf.memory)));
      return algo_perf;
    }
  }
#endif

  return perf_vec.at(best_algo_idx);
}

}  // namespace

CudnnConvArgs::CudnnConvArgs(const PbMessage& conf, const BlobDesc* x, const BlobDesc* y,
                             const BlobDesc* w, size_t max_ws_size, bool deterministic,
                             bool heuristic, const bool enable_true_half)
    : xdesc(x->data_type(), x->shape(), GetValFromPbMessage<std::string>(conf, "data_format")),
      ydesc(y->data_type(), y->shape(), GetValFromPbMessage<std::string>(conf, "data_format")),
      wdesc(w->data_type(), w->shape(), GetValFromPbMessage<std::string>(conf, "data_format")),
      cdesc(GetConvDescDataType(x->data_type(), enable_true_half), x->shape(), conf),
      x_ndims(x->shape().NumAxes()),
      y_ndims(y->shape().NumAxes()),
      w_ndims(w->shape().NumAxes()),
      ws_size(max_ws_size) {
  CudaCheck(cudnnCreate(&handle));
  need_destroy_handle = true;
  if (heuristic) {
    need_free_memory = false;
  } else {
    CudaCheck(cudaMalloc(&x_dptr, RtBlobDesc(*x).AlignedByteSizeOfBlobBody()));
    CudaCheck(cudaMalloc(&w_dptr, RtBlobDesc(*w).AlignedByteSizeOfBlobBody()));
    CudaCheck(cudaMalloc(&y_dptr, RtBlobDesc(*y).AlignedByteSizeOfBlobBody()));
    CudaCheck(cudaMalloc(&work_space, max_ws_size));
    need_free_memory = true;
  }

  std::memset(&params, 0, sizeof(CudnnConvParams));

  cudnnDataType_t x_data_type;
  int x_dim_size = 0;
  CudaCheck(cudnnGetTensorNdDescriptor(xdesc.Get(), CUDNN_DIM_MAX, &x_data_type, &x_dim_size,
                                       params.x_dim, params.x_stride));
  cudnnDataType_t filter_data_type;
  cudnnTensorFormat_t filter_format;
  int filter_dim_size = 0;
  CudaCheck(cudnnGetFilterNdDescriptor(wdesc.Get(), CUDNN_DIM_MAX, &filter_data_type,
                                       &filter_format, &filter_dim_size, params.weight_dim));
  cudnnConvolutionMode_t mode;
  int conv_dim_size = 0;
  CudaCheck(cudnnGetConvolutionNdDescriptor(cdesc.Get(), CudnnConvParams::max_dim, &conv_dim_size,
                                            params.padding, params.stride, params.dilation, &mode,
                                            &params.data_type));
  CHECK_EQ(x_dim_size, filter_dim_size);
  CHECK_EQ(x_data_type, params.data_type);
  CHECK_EQ(filter_data_type, params.data_type);
  params.deterministic = deterministic;
  params.heuristic = heuristic;
}

CudnnConvArgs::CudnnConvArgs(const PbMessage& conf, cudnnHandle_t handle, const Blob* x,
                             const Blob* y, const Blob* w, Blob* buf, bool deterministic,
                             bool heuristic, const bool enable_true_half)
    : handle(handle),
      xdesc(x->data_type(), x->shape(), GetValFromPbMessage<std::string>(conf, "data_format")),
      ydesc(y->data_type(), y->shape(), GetValFromPbMessage<std::string>(conf, "data_format")),
      wdesc(w->data_type(), w->shape(), GetValFromPbMessage<std::string>(conf, "data_format")),
      cdesc(GetConvDescDataType(x->data_type(), enable_true_half), x->shape(), conf),
      x_ndims(x->shape().NumAxes()),
      y_ndims(y->shape().NumAxes()),
      w_ndims(w->shape().NumAxes()),
      x_dptr(const_cast<void*>(x->dptr())),
      y_dptr(const_cast<void*>(y->dptr())),
      w_dptr(const_cast<void*>(w->dptr())),
      work_space(buf ? buf->mut_dptr() : nullptr),
      ws_size(buf ? buf->ByteSizeOfBlobBody() : 0),
      need_destroy_handle(false),
      need_free_memory(false) {
  std::memset(&params, 0, sizeof(CudnnConvParams));

  cudnnDataType_t x_data_type;
  int x_dim_size = 0;
  CudaCheck(cudnnGetTensorNdDescriptor(xdesc.Get(), CUDNN_DIM_MAX, &x_data_type, &x_dim_size,
                                       params.x_dim, params.x_stride));
  cudnnDataType_t filter_data_type;
  cudnnTensorFormat_t filter_format;
  int filter_dim_size = 0;
  CudaCheck(cudnnGetFilterNdDescriptor(wdesc.Get(), CUDNN_DIM_MAX, &filter_data_type,
                                       &filter_format, &filter_dim_size, params.weight_dim));
  cudnnConvolutionMode_t mode;
  int conv_dim_size = 0;
  CudaCheck(cudnnGetConvolutionNdDescriptor(cdesc.Get(), CudnnConvParams::max_dim, &conv_dim_size,
                                            params.padding, params.stride, params.dilation, &mode,
                                            &params.data_type));
  CHECK_EQ(x_dim_size, filter_dim_size);
  CHECK_EQ(x_data_type, params.data_type);
  CHECK_EQ(filter_data_type, params.data_type);
  params.deterministic = deterministic;
  params.heuristic = heuristic;
}

CudnnConvArgs::~CudnnConvArgs() {
  if (need_free_memory) {
    CudaCheck(cudaFree(x_dptr));
    CudaCheck(cudaFree(w_dptr));
    CudaCheck(cudaFree(y_dptr));
    CudaCheck(cudaFree(work_space));
  }
  if (need_destroy_handle) { CudaCheck(cudnnDestroy(handle)); }
}

bool operator==(const CudnnConvParams& a, const CudnnConvParams& b) {
  auto ptr1 = reinterpret_cast<const uint8_t*>(&a);
  auto ptr2 = reinterpret_cast<const uint8_t*>(&b);
  return memcmp(ptr1, ptr2, sizeof(CudnnConvParams)) == 0;
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
    int max_algo_cnt = 0;
    int found_algo_cnt = 0;
    CudaCheck(cudnnGetConvolutionForwardAlgorithmMaxCount(args.handle, &max_algo_cnt));
    std::vector<perf_t> perf_vec(max_algo_cnt);
    if (args.params.heuristic) {
      CudaCheck(cudnnGetConvolutionForwardAlgorithm_v7(
          args.handle, args.xdesc.Get(), args.wdesc.Get(), args.cdesc.Get(), args.ydesc.Get(),
          max_algo_cnt, &found_algo_cnt, perf_vec.data()));
    } else {
      CudaCheck(cudnnFindConvolutionForwardAlgorithmEx(
          args.handle, args.xdesc.Get(), args.x_dptr, args.wdesc.Get(), args.w_dptr,
          args.cdesc.Get(), args.ydesc.Get(), args.y_dptr, max_algo_cnt, &found_algo_cnt,
          perf_vec.data(), args.work_space, args.ws_size));
    }
    perf_vec.resize(found_algo_cnt);
    if (perf_vec.size() > 0) {
      *algo_perf = GetBestAlgorithm<perf_t>(args, perf_vec);
    } else {
      algo_perf->algo = DEFAULT_ALGO;
      if (args.params.data_type == CUDNN_DATA_HALF) {
        algo_perf->mathType = CUDNN_TENSOR_OP_MATH;
      } else {
        algo_perf->mathType = CUDNN_DEFAULT_MATH;
      }
      CudaCheck(GetConvWorkspaceSize(args, algo_perf->algo, &(algo_perf->memory)));
    }
  }
};

template<>
struct CudnnConvAlgorithmSearch<cudnnConvolutionBwdDataAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdDataAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdDataAlgo_t;
  static constexpr algo_t DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

  static void FindAlgorithm(const CudnnConvArgs& args, perf_t* algo_perf) {
    int max_algo_cnt = 0;
    int found_algo_cnt = 0;
    CudaCheck(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(args.handle, &max_algo_cnt));
    std::vector<perf_t> perf_vec(max_algo_cnt);
    if (args.params.heuristic) {
      CudaCheck(cudnnGetConvolutionBackwardDataAlgorithm_v7(
          args.handle, args.wdesc.Get(), args.ydesc.Get(), args.cdesc.Get(), args.xdesc.Get(),
          max_algo_cnt, &found_algo_cnt, perf_vec.data()));
    } else {
      CudaCheck(cudnnFindConvolutionBackwardDataAlgorithmEx(
          args.handle, args.wdesc.Get(), args.w_dptr, args.ydesc.Get(), args.y_dptr,
          args.cdesc.Get(), args.xdesc.Get(), args.x_dptr, max_algo_cnt, &found_algo_cnt,
          perf_vec.data(), args.work_space, args.ws_size));
    }
    perf_vec.resize(found_algo_cnt);
    if (perf_vec.size() > 0) {
      *algo_perf = GetBestAlgorithm<perf_t>(args, perf_vec);
    } else {
      algo_perf->algo = DEFAULT_ALGO;
      if (args.params.data_type == CUDNN_DATA_HALF) {
        algo_perf->mathType = CUDNN_TENSOR_OP_MATH;
      } else {
        algo_perf->mathType = CUDNN_DEFAULT_MATH;
      }
      CudaCheck(GetConvWorkspaceSize(args, algo_perf->algo, &(algo_perf->memory)));
    }
  }
};

template<>
struct CudnnConvAlgorithmSearch<cudnnConvolutionBwdFilterAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdFilterAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdFilterAlgo_t;
  static constexpr algo_t DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

  static void FindAlgorithm(const CudnnConvArgs& args, perf_t* algo_perf) {
    int max_algo_cnt = 0;
    int found_algo_cnt = 0;
    CudaCheck(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(args.handle, &max_algo_cnt));
    std::vector<perf_t> perf_vec(max_algo_cnt);
    if (args.params.heuristic) {
      CudaCheck(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
          args.handle, args.xdesc.Get(), args.ydesc.Get(), args.cdesc.Get(), args.wdesc.Get(),
          max_algo_cnt, &found_algo_cnt, perf_vec.data()));
    } else {
      CudaCheck(cudnnFindConvolutionBackwardFilterAlgorithmEx(
          args.handle, args.xdesc.Get(), args.x_dptr, args.ydesc.Get(), args.y_dptr,
          args.cdesc.Get(), args.wdesc.Get(), args.w_dptr, max_algo_cnt, &found_algo_cnt,
          perf_vec.data(), args.work_space, args.ws_size));
    }
    perf_vec.resize(found_algo_cnt);
    if (perf_vec.size() > 0) {
      *algo_perf = GetBestAlgorithm<perf_t>(args, perf_vec);
    } else {
      algo_perf->algo = DEFAULT_ALGO;
      if (args.params.data_type == CUDNN_DATA_HALF) {
        algo_perf->mathType = CUDNN_TENSOR_OP_MATH;
      } else {
        algo_perf->mathType = CUDNN_DEFAULT_MATH;
      }
      CudaCheck(GetConvWorkspaceSize(args, algo_perf->algo, &(algo_perf->memory)));
    }
  }
};

template<typename perf_t>
std::shared_ptr<perf_t> FindCudnnConvAlgorithm(const CudnnConvArgs& args) {
  auto Infer = [&args](const CudnnConvParams& params) {
    auto* perf = new perf_t();
    CudnnConvAlgorithmSearch<perf_t>::FindAlgorithm(args, perf);
    return std::shared_ptr<perf_t>(perf);
  };
  size_t cache_size = Global<ResourceDesc>::Get()->thread_local_cache_max_size();
  return ThreadLocalCachedCall(cache_size, Infer, args.params);
}

template std::shared_ptr<cudnnConvolutionFwdAlgoPerf_t> FindCudnnConvAlgorithm(
    const CudnnConvArgs& args);
template std::shared_ptr<cudnnConvolutionBwdDataAlgoPerf_t> FindCudnnConvAlgorithm(
    const CudnnConvArgs& args);
template std::shared_ptr<cudnnConvolutionBwdFilterAlgoPerf_t> FindCudnnConvAlgorithm(
    const CudnnConvArgs& args);

}  // namespace oneflow

#endif  // WITH_CUDA
