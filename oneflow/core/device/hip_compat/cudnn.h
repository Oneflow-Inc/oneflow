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
#pragma once

// The subset of the cuDNN API that oneflow calls, implemented on MIOpen. Only the HIP build sees
// this header (it shadows <cudnn.h> on the HIP include path).
//
// Where MIOpen differs from cuDNN the difference is absorbed here:
//  - MIOpen has no filter descriptor and no compute type or math type on a convolution
//    descriptor, so those are small structs that carry the extra state.
//  - MIOpen convolution and batch norm do not honour beta != 0 (accumulate into the output).
//    Those calls run into a stream-ordered scratch buffer and are then added with miopenOpTensor.
//  - MIOpen selects convolution algorithms only through Find, which runs every candidate and so
//    writes the output tensor. The heuristic query (cudnnGet*Algorithm_v7, which touches no data
//    on cuDNN) therefore runs Find on scratch buffers, never on the caller's tensors.
//  - MIOpen runs a convolution only on a handle where Find has been run for that problem. oneflow
//    searches once per problem on whichever handle is at hand and then runs on the stream's own
//    handle, so the first run of a problem on a handle is preceded by a Find on it, into the
//    output that the run overwrites anyway.
//  - MIOpen max-pooling backward reads indices recorded by its forward pass. cuDNN recomputes
//    them from x and y, so the backward call reruns the forward into scratch to obtain them.
//  - The Ex batch-norm entry points (fused add and activation) have no MIOpen counterpart. The
//    code that uses them is excluded from the HIP build, and the non-fused path runs instead.

#include "oneflow/core/device/cuda_to_hip.h"
#include <cublasLt.h>
#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <miopen/version.h>
#include <algorithm>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define CUDNN_MAJOR MIOPEN_VERSION_MAJOR
#define CUDNN_MINOR MIOPEN_VERSION_MINOR
#define CUDNN_PATCHLEVEL MIOPEN_VERSION_PATCH
// The cuDNN feature level this header provides (bf16 data is available).
#define CUDNN_VERSION 8100

typedef enum libraryPropertyType_t {
  MAJOR_VERSION,
  MINOR_VERSION,
  PATCH_LEVEL
} libraryPropertyType;

typedef miopenStatus_t cudnnStatus_t;
#define CUDNN_STATUS_SUCCESS miopenStatusSuccess
#define CUDNN_STATUS_NOT_INITIALIZED miopenStatusNotInitialized
#define CUDNN_STATUS_ALLOC_FAILED miopenStatusAllocFailed
#define CUDNN_STATUS_BAD_PARAM miopenStatusBadParm
#define CUDNN_STATUS_INTERNAL_ERROR miopenStatusInternalError
#define CUDNN_STATUS_NOT_SUPPORTED miopenStatusUnsupportedOp
#define CUDNN_STATUS_EXECUTION_FAILED miopenStatusUnknownError

typedef miopenDataType_t cudnnDataType_t;
#define CUDNN_DATA_FLOAT miopenFloat
#define CUDNN_DATA_DOUBLE miopenDouble
#define CUDNN_DATA_HALF miopenHalf
#define CUDNN_DATA_BFLOAT16 miopenBFloat16
#define CUDNN_DATA_INT8 miopenInt8
#define CUDNN_DATA_INT32 miopenInt32
// Types MIOpen does not have. Distinct unused values (inside the enum's value range) keep switch
// statements over the type valid; no MIOpen descriptor ever carries them.
#define CUDNN_DATA_UINT8 static_cast<miopenDataType_t>(12)
#define CUDNN_DATA_INT8x4 static_cast<miopenDataType_t>(13)
#define CUDNN_DATA_UINT8x4 static_cast<miopenDataType_t>(14)
#define CUDNN_DATA_INT8x32 static_cast<miopenDataType_t>(15)

typedef enum { CUDNN_TENSOR_NCHW = 0, CUDNN_TENSOR_NHWC = 1 } cudnnTensorFormat_t;
typedef enum { CUDNN_NOT_PROPAGATE_NAN = 0, CUDNN_PROPAGATE_NAN = 1 } cudnnNanPropagation_t;
typedef enum { CUDNN_NON_DETERMINISTIC = 0, CUDNN_DETERMINISTIC = 1 } cudnnDeterminism_t;
typedef enum {
  CUDNN_DEFAULT_MATH = 0,
  CUDNN_TENSOR_OP_MATH = 1,
  CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = 2,
  CUDNN_FMA_MATH = 3
} cudnnMathType_t;
typedef enum { CUDNN_CONVOLUTION = 0, CUDNN_CROSS_CORRELATION = 1 } cudnnConvolutionMode_t;

namespace oneflow_hip_cudnn {

inline bool IsDoubleScalar(miopenTensorDescriptor_t desc) {
  miopenDataType_t data_type;
  int ndim = 0;
  if (miopenGetTensorDescriptorSize(desc, &ndim) != miopenStatusSuccess) { return false; }
  std::vector<int> dims(ndim), strides(ndim);
  if (miopenGetTensorDescriptor(desc, &data_type, dims.data(), strides.data())
      != miopenStatusSuccess) {
    return false;
  }
  return data_type == miopenDouble;
}

// cuDNN scaling factors are double for double tensors and float otherwise.
inline double ReadScalar(const void* p, bool is_double) {
  return is_double ? *static_cast<const double*>(p) : *static_cast<const float*>(p);
}

inline bool IsZero(const void* beta, miopenTensorDescriptor_t desc) {
  return ReadScalar(beta, IsDoubleScalar(desc)) == 0.0;
}

inline bool IsOne(const void* alpha, miopenTensorDescriptor_t desc) {
  return ReadScalar(alpha, IsDoubleScalar(desc)) == 1.0;
}

inline const void* ZeroPtr(miopenTensorDescriptor_t desc) {
  static const float f = 0.f;
  static const double d = 0.0;
  return IsDoubleScalar(desc) ? static_cast<const void*>(&d) : static_cast<const void*>(&f);
}

inline const void* OnePtr(miopenTensorDescriptor_t desc) {
  static const float f = 1.f;
  static const double d = 1.0;
  return IsDoubleScalar(desc) ? static_cast<const void*>(&d) : static_cast<const void*>(&f);
}

inline size_t NumBytes(miopenTensorDescriptor_t desc) {
  size_t bytes = 0;
  if (miopenGetTensorNumBytes(desc, &bytes) != miopenStatusSuccess) { return 0; }
  return bytes;
}

// Runs `op(out)` so that `y = op_result + beta * y`, as cuDNN does for a nonzero beta. With
// beta == 0 the operation writes y directly.
template<typename F>
miopenStatus_t RunWithBeta(miopenHandle_t handle, const void* beta, miopenTensorDescriptor_t y_desc,
                           void* y, const F& op) {
  if (IsZero(beta, y_desc)) { return op(ZeroPtr(y_desc), y); }
  hipStream_t stream = nullptr;
  miopenStatus_t status = miopenGetStream(handle, &stream);
  if (status != miopenStatusSuccess) { return status; }
  void* tmp = nullptr;
  if (hipMallocAsync(&tmp, NumBytes(y_desc), stream) != hipSuccess) {
    return miopenStatusAllocFailed;
  }
  status = op(ZeroPtr(y_desc), tmp);
  if (status == miopenStatusSuccess) {
    status = miopenOpTensor(handle, miopenTensorOpAdd, OnePtr(y_desc), y_desc, tmp, ZeroPtr(y_desc),
                            y_desc, tmp, beta, y_desc, y);
  }
  if (hipFreeAsync(tmp, stream) != hipSuccess && status == miopenStatusSuccess) {
    status = miopenStatusInternalError;
  }
  return status;
}

inline void PackedStrides(cudnnTensorFormat_t format, int nb_dims, const int* dims, int* strides) {
  if (format == CUDNN_TENSOR_NHWC && nb_dims >= 3) {
    // Channels innermost, then the spatial axes in order, then the batch.
    int stride = 1;
    strides[1] = stride;
    stride *= dims[1];
    for (int i = nb_dims - 1; i >= 2; --i) {
      strides[i] = stride;
      stride *= dims[i];
    }
    strides[0] = stride;
  } else {
    int stride = 1;
    for (int i = nb_dims - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= dims[i];
    }
  }
}

}  // namespace oneflow_hip_cudnn

// ---- handle --------------------------------------------------------------------
typedef miopenHandle_t cudnnHandle_t;

inline const char* cudnnGetErrorString(cudnnStatus_t status) {
  return miopenGetErrorString(status);
}

inline cudnnStatus_t cudnnCreate(miopenHandle_t* handle) {
  // miopenCreate creates a hipBLASLt handle (see HandleCreationGuard).
  oneflow_hip_cublaslt::HandleCreationGuard guard;
  return miopenCreate(handle);
}

namespace oneflow_hip_cudnn {

// The convolution problems Find has run for on each live handle (see FirstRunOnHandle). Never
// destroyed, so handles released during exit can still drop their entry.
struct FindRecords {
  std::mutex mutex;
  std::unordered_map<miopenHandle_t, std::unordered_set<std::string>> problems;
};

inline FindRecords* GetFindRecords() {
  static FindRecords* records = new FindRecords;
  return records;
}

}  // namespace oneflow_hip_cudnn

// A destroyed handle's Find record is dropped: the next miopenCreate can return the same address,
// and MIOpen has run no Find on that new handle.
inline cudnnStatus_t cudnnDestroy(miopenHandle_t handle) {
  oneflow_hip_cudnn::FindRecords* records = oneflow_hip_cudnn::GetFindRecords();
  {
    std::lock_guard<std::mutex> lock(records->mutex);
    records->problems.erase(handle);
  }
  return miopenDestroy(handle);
}

inline cudnnStatus_t cudnnSetStream(miopenHandle_t handle, hipStream_t stream) {
  return miopenSetStream(handle, stream);
}

inline cudnnStatus_t cudnnGetProperty(libraryPropertyType type, int* value) {
  size_t major = 0, minor = 0, patch = 0;
  const miopenStatus_t status = miopenGetVersion(&major, &minor, &patch);
  if (status != miopenStatusSuccess) { return status; }
  switch (type) {
    case MAJOR_VERSION: *value = static_cast<int>(major); break;
    case MINOR_VERSION: *value = static_cast<int>(minor); break;
    case PATCH_LEVEL: *value = static_cast<int>(patch); break;
    default: return miopenStatusBadParm;
  }
  return miopenStatusSuccess;
}

// ---- tensor descriptor ---------------------------------------------------------
typedef miopenTensorDescriptor_t cudnnTensorDescriptor_t;

inline cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* desc) {
  return miopenCreateTensorDescriptor(desc);
}

inline cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t desc) {
  return miopenDestroyTensorDescriptor(desc);
}

inline cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t desc,
                                                cudnnTensorFormat_t format,
                                                cudnnDataType_t data_type, int n, int c, int h,
                                                int w) {
  const int dims[4] = {n, c, h, w};
  int strides[4];
  oneflow_hip_cudnn::PackedStrides(format, 4, dims, strides);
  return miopenSetTensorDescriptor(desc, data_type, 4, dims, strides);
}

inline cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t desc,
                                                cudnnDataType_t data_type, int nb_dims,
                                                const int* dims, const int* strides) {
  return miopenSetTensorDescriptor(desc, data_type, nb_dims, dims, strides);
}

inline cudnnStatus_t cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t desc,
                                                int nb_dims_requested, cudnnDataType_t* data_type,
                                                int* nb_dims, int* dims, int* strides) {
  miopenStatus_t status = miopenGetTensorDescriptorSize(desc, nb_dims);
  if (status != miopenStatusSuccess) { return status; }
  std::vector<int> all_dims(*nb_dims), all_strides(*nb_dims);
  status = miopenGetTensorDescriptor(desc, data_type, all_dims.data(), all_strides.data());
  if (status != miopenStatusSuccess) { return status; }
  for (int i = 0; i < std::min(nb_dims_requested, *nb_dims); ++i) {
    dims[i] = all_dims[i];
    strides[i] = all_strides[i];
  }
  return miopenStatusSuccess;
}

inline cudnnStatus_t cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t desc,
                                                cudnnDataType_t* data_type, int* n, int* c, int* h,
                                                int* w, int* n_stride, int* c_stride, int* h_stride,
                                                int* w_stride) {
  int nb_dims = 0;
  int dims[4] = {1, 1, 1, 1};
  int strides[4] = {1, 1, 1, 1};
  const miopenStatus_t status =
      cudnnGetTensorNdDescriptor(desc, 4, data_type, &nb_dims, dims, strides);
  if (status != miopenStatusSuccess) { return status; }
  if (nb_dims != 4) { return miopenStatusBadParm; }
  *n = dims[0];
  *c = dims[1];
  *h = dims[2];
  *w = dims[3];
  *n_stride = strides[0];
  *c_stride = strides[1];
  *h_stride = strides[2];
  *w_stride = strides[3];
  return miopenStatusSuccess;
}

inline cudnnStatus_t cudnnAddTensor(miopenHandle_t handle, const void* alpha,
                                    const cudnnTensorDescriptor_t a_desc, const void* a,
                                    const void* beta, const cudnnTensorDescriptor_t c_desc,
                                    void* c) {
  // C = alpha * A + beta * C. miopenOpTensor broadcasts only its second operand, so C goes first.
  return miopenOpTensor(handle, miopenTensorOpAdd, beta, c_desc, c, alpha, a_desc, a,
                        oneflow_hip_cudnn::ZeroPtr(c_desc), c_desc, c);
}

// ---- filter descriptor -----------------------------------------------------------
struct cudnnFilterStruct {
  miopenTensorDescriptor_t desc;
  cudnnTensorFormat_t format;
};
typedef cudnnFilterStruct* cudnnFilterDescriptor_t;

inline cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* desc) {
  *desc = new cudnnFilterStruct{nullptr, CUDNN_TENSOR_NCHW};
  const miopenStatus_t status = miopenCreateTensorDescriptor(&(*desc)->desc);
  if (status != miopenStatusSuccess) {
    delete *desc;
    *desc = nullptr;
  }
  return status;
}

inline cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t desc) {
  if (desc == nullptr) { return miopenStatusSuccess; }
  const miopenStatus_t status = miopenDestroyTensorDescriptor(desc->desc);
  delete desc;
  return status;
}

inline cudnnStatus_t cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t desc,
                                                cudnnDataType_t data_type,
                                                cudnnTensorFormat_t format, int nb_dims,
                                                const int* dims) {
  std::vector<int> strides(nb_dims);
  oneflow_hip_cudnn::PackedStrides(format, nb_dims, dims, strides.data());
  desc->format = format;
  return miopenSetTensorDescriptor(desc->desc, data_type, nb_dims, dims, strides.data());
}

inline cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t desc,
                                                cudnnDataType_t data_type,
                                                cudnnTensorFormat_t format, int k, int c, int h,
                                                int w) {
  const int dims[4] = {k, c, h, w};
  return cudnnSetFilterNdDescriptor(desc, data_type, format, 4, dims);
}

inline cudnnStatus_t cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t desc,
                                                int nb_dims_requested, cudnnDataType_t* data_type,
                                                cudnnTensorFormat_t* format, int* nb_dims,
                                                int* dims) {
  std::vector<int> strides(nb_dims_requested);
  *format = desc->format;
  return cudnnGetTensorNdDescriptor(desc->desc, nb_dims_requested, data_type, nb_dims, dims,
                                    strides.data());
}

// ---- convolution descriptor ----------------------------------------------------
struct cudnnConvolutionStruct {
  miopenConvolutionDescriptor_t desc;
  cudnnDataType_t compute_type;
  cudnnMathType_t math_type;
};
typedef cudnnConvolutionStruct* cudnnConvolutionDescriptor_t;

inline cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* desc) {
  *desc = new cudnnConvolutionStruct{nullptr, CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH};
  const miopenStatus_t status = miopenCreateConvolutionDescriptor(&(*desc)->desc);
  if (status != miopenStatusSuccess) {
    delete *desc;
    *desc = nullptr;
  }
  return status;
}

inline cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t desc) {
  if (desc == nullptr) { return miopenStatusSuccess; }
  const miopenStatus_t status = miopenDestroyConvolutionDescriptor(desc->desc);
  delete desc;
  return status;
}

inline cudnnStatus_t cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t desc,
                                                     int array_length, const int* pads,
                                                     const int* strides, const int* dilations,
                                                     cudnnConvolutionMode_t mode,
                                                     cudnnDataType_t compute_type) {
  // MIOpen's miopenConvolution is cross-correlation; true convolution has no MIOpen mode.
  if (mode != CUDNN_CROSS_CORRELATION) { return miopenStatusUnsupportedOp; }
  desc->compute_type = compute_type;
  return miopenInitConvolutionNdDescriptor(desc->desc, array_length, pads, strides, dilations,
                                           miopenConvolution);
}

inline cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t desc, int pad_h,
                                                     int pad_w, int u, int v, int dilation_h,
                                                     int dilation_w, cudnnConvolutionMode_t mode,
                                                     cudnnDataType_t compute_type) {
  const int pads[2] = {pad_h, pad_w};
  const int strides[2] = {u, v};
  const int dilations[2] = {dilation_h, dilation_w};
  return cudnnSetConvolutionNdDescriptor(desc, 2, pads, strides, dilations, mode, compute_type);
}

inline cudnnStatus_t cudnnGetConvolutionNdDescriptor(const cudnnConvolutionDescriptor_t desc,
                                                     int array_length_requested, int* array_length,
                                                     int* pads, int* strides, int* dilations,
                                                     cudnnConvolutionMode_t* mode,
                                                     cudnnDataType_t* compute_type) {
  // MIOpen rejects a request for more dimensions than the descriptor has; cuDNN returns the
  // dimensions it has.
  int spatial_dim = 0;
  miopenStatus_t status = miopenGetConvolutionSpatialDim(desc->desc, &spatial_dim);
  if (status != miopenStatusSuccess) { return status; }
  miopenConvolutionMode_t miopen_mode;
  status =
      miopenGetConvolutionNdDescriptor(desc->desc, std::min(array_length_requested, spatial_dim),
                                       array_length, pads, strides, dilations, &miopen_mode);
  if (status != miopenStatusSuccess) { return status; }
  *mode = CUDNN_CROSS_CORRELATION;
  *compute_type = desc->compute_type;
  return miopenStatusSuccess;
}

inline cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t desc,
                                                   int group_count) {
  return miopenSetConvolutionGroupCount(desc->desc, group_count);
}

inline cudnnStatus_t cudnnGetConvolutionGroupCount(const cudnnConvolutionDescriptor_t desc,
                                                   int* group_count) {
  return miopenGetConvolutionGroupCount(desc->desc, group_count);
}

// MIOpen chooses the arithmetic from the data type; the math type is kept only so that it reads
// back as set.
inline cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t desc,
                                                 cudnnMathType_t math_type) {
  desc->math_type = math_type;
  return miopenStatusSuccess;
}

// ---- convolution algorithms ----------------------------------------------------
typedef miopenConvFwdAlgorithm_t cudnnConvolutionFwdAlgo_t;
typedef miopenConvBwdDataAlgorithm_t cudnnConvolutionBwdDataAlgo_t;
typedef miopenConvBwdWeightsAlgorithm_t cudnnConvolutionBwdFilterAlgo_t;
#define CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM miopenConvolutionFwdAlgoImplicitGEMM
#define CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 miopenConvolutionBwdDataAlgoImplicitGEMM
#define CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 miopenConvolutionBwdWeightsAlgoImplicitGEMM

#define OF_HIP_CUDNN_DEFINE_PERF(perf_t, algo_t) \
  struct perf_t {                                \
    algo_t algo;                                 \
    cudnnStatus_t status;                        \
    float time;                                  \
    size_t memory;                               \
    cudnnDeterminism_t determinism;              \
    cudnnMathType_t mathType;                    \
  };
OF_HIP_CUDNN_DEFINE_PERF(cudnnConvolutionFwdAlgoPerf_t, cudnnConvolutionFwdAlgo_t)
OF_HIP_CUDNN_DEFINE_PERF(cudnnConvolutionBwdDataAlgoPerf_t, cudnnConvolutionBwdDataAlgo_t)
OF_HIP_CUDNN_DEFINE_PERF(cudnnConvolutionBwdFilterAlgoPerf_t, cudnnConvolutionBwdFilterAlgo_t)
#undef OF_HIP_CUDNN_DEFINE_PERF

namespace oneflow_hip_cudnn {

// Number of results asked of a MIOpen Find (it returns the applicable algorithms, fastest first).
constexpr int kMaxAlgoCount = 8;

inline cudnnDataType_t MathTypeDataType(miopenTensorDescriptor_t desc) {
  miopenDataType_t data_type = miopenFloat;
  int ndim = 0;
  if (miopenGetTensorDescriptorSize(desc, &ndim) == miopenStatusSuccess) {
    std::vector<int> dims(ndim), strides(ndim);
    miopenGetTensorDescriptor(desc, &data_type, dims.data(), strides.data());
  }
  return data_type;
}

template<typename perf_t>
void FillPerf(const miopenConvAlgoPerf_t& in, decltype(perf_t::algo) algo,
              miopenTensorDescriptor_t desc, perf_t* out) {
  out->algo = algo;
  out->status = miopenStatusSuccess;
  out->time = in.time;
  out->memory = in.memory;
  out->determinism = CUDNN_DETERMINISTIC;
  const cudnnDataType_t data_type = MathTypeDataType(desc);
  out->mathType = (data_type == CUDNN_DATA_HALF || data_type == CUDNN_DATA_BFLOAT16)
                      ? CUDNN_TENSOR_OP_MATH
                      : CUDNN_DEFAULT_MATH;
}

// Device scratch that lives for one Find on scratch data.
class ScratchBuffers {
 public:
  ScratchBuffers() = default;
  ~ScratchBuffers() {
    for (void* p : ptrs_) { (void)hipFree(p); }
  }
  void* Alloc(size_t bytes) {
    void* p = nullptr;
    if (hipMalloc(&p, bytes == 0 ? 1 : bytes) != hipSuccess) { return nullptr; }
    ptrs_.push_back(p);
    return p;
  }

 private:
  std::vector<void*> ptrs_;
};

}  // namespace oneflow_hip_cudnn

inline cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount(miopenHandle_t, int* count) {
  *count = oneflow_hip_cudnn::kMaxAlgoCount;
  return miopenStatusSuccess;
}

inline cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithmMaxCount(miopenHandle_t, int* count) {
  *count = oneflow_hip_cudnn::kMaxAlgoCount;
  return miopenStatusSuccess;
}

inline cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(miopenHandle_t,
                                                                        int* count) {
  *count = oneflow_hip_cudnn::kMaxAlgoCount;
  return miopenStatusSuccess;
}

inline cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
    miopenHandle_t handle, const cudnnTensorDescriptor_t x_desc,
    const cudnnFilterDescriptor_t w_desc, const cudnnConvolutionDescriptor_t conv_desc,
    const cudnnTensorDescriptor_t y_desc, cudnnConvolutionFwdAlgo_t, size_t* size) {
  return miopenConvolutionForwardGetWorkSpaceSize(handle, w_desc->desc, x_desc, conv_desc->desc,
                                                  y_desc, size);
}

inline cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
    miopenHandle_t handle, const cudnnFilterDescriptor_t w_desc,
    const cudnnTensorDescriptor_t dy_desc, const cudnnConvolutionDescriptor_t conv_desc,
    const cudnnTensorDescriptor_t dx_desc, cudnnConvolutionBwdDataAlgo_t, size_t* size) {
  return miopenConvolutionBackwardDataGetWorkSpaceSize(handle, dy_desc, w_desc->desc,
                                                       conv_desc->desc, dx_desc, size);
}

inline cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
    miopenHandle_t handle, const cudnnTensorDescriptor_t x_desc,
    const cudnnTensorDescriptor_t dy_desc, const cudnnConvolutionDescriptor_t conv_desc,
    const cudnnFilterDescriptor_t dw_desc, cudnnConvolutionBwdFilterAlgo_t, size_t* size) {
  return miopenConvolutionBackwardWeightsGetWorkSpaceSize(handle, dy_desc, x_desc, conv_desc->desc,
                                                          dw_desc->desc, size);
}

inline cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx(
    miopenHandle_t handle, const cudnnTensorDescriptor_t x_desc, const void* x,
    const cudnnFilterDescriptor_t w_desc, const void* w,
    const cudnnConvolutionDescriptor_t conv_desc, const cudnnTensorDescriptor_t y_desc, void* y,
    int requested_count, int* returned_count, cudnnConvolutionFwdAlgoPerf_t* perf_results,
    void* workspace, size_t workspace_size) {
  miopenConvAlgoPerf_t results[oneflow_hip_cudnn::kMaxAlgoCount];
  const int count = std::min(requested_count, oneflow_hip_cudnn::kMaxAlgoCount);
  const miopenStatus_t status = miopenFindConvolutionForwardAlgorithm(
      handle, x_desc, x, w_desc->desc, w, conv_desc->desc, y_desc, y, count, returned_count,
      results, workspace, workspace_size, true);
  if (status != miopenStatusSuccess) { return status; }
  for (int i = 0; i < *returned_count; ++i) {
    oneflow_hip_cudnn::FillPerf(results[i], results[i].fwd_algo, x_desc, &perf_results[i]);
  }
  return miopenStatusSuccess;
}

inline cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx(
    miopenHandle_t handle, const cudnnFilterDescriptor_t w_desc, const void* w,
    const cudnnTensorDescriptor_t dy_desc, const void* dy,
    const cudnnConvolutionDescriptor_t conv_desc, const cudnnTensorDescriptor_t dx_desc, void* dx,
    int requested_count, int* returned_count, cudnnConvolutionBwdDataAlgoPerf_t* perf_results,
    void* workspace, size_t workspace_size) {
  miopenConvAlgoPerf_t results[oneflow_hip_cudnn::kMaxAlgoCount];
  const int count = std::min(requested_count, oneflow_hip_cudnn::kMaxAlgoCount);
  const miopenStatus_t status = miopenFindConvolutionBackwardDataAlgorithm(
      handle, dy_desc, dy, w_desc->desc, w, conv_desc->desc, dx_desc, dx, count, returned_count,
      results, workspace, workspace_size, true);
  if (status != miopenStatusSuccess) { return status; }
  for (int i = 0; i < *returned_count; ++i) {
    oneflow_hip_cudnn::FillPerf(results[i], results[i].bwd_data_algo, dx_desc, &perf_results[i]);
  }
  return miopenStatusSuccess;
}

inline cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx(
    miopenHandle_t handle, const cudnnTensorDescriptor_t x_desc, const void* x,
    const cudnnTensorDescriptor_t dy_desc, const void* dy,
    const cudnnConvolutionDescriptor_t conv_desc, const cudnnFilterDescriptor_t dw_desc, void* dw,
    int requested_count, int* returned_count, cudnnConvolutionBwdFilterAlgoPerf_t* perf_results,
    void* workspace, size_t workspace_size) {
  miopenConvAlgoPerf_t results[oneflow_hip_cudnn::kMaxAlgoCount];
  const int count = std::min(requested_count, oneflow_hip_cudnn::kMaxAlgoCount);
  const miopenStatus_t status = miopenFindConvolutionBackwardWeightsAlgorithm(
      handle, dy_desc, dy, x_desc, x, conv_desc->desc, dw_desc->desc, dw, count, returned_count,
      results, workspace, workspace_size, true);
  if (status != miopenStatusSuccess) { return status; }
  for (int i = 0; i < *returned_count; ++i) {
    oneflow_hip_cudnn::FillPerf(results[i], results[i].bwd_weights_algo, x_desc, &perf_results[i]);
  }
  return miopenStatusSuccess;
}

// The heuristic queries take no data pointers, so the Find runs on scratch buffers and a
// workspace as large as MIOpen asks for; the caller still filters the results by its own
// workspace limit.
inline cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
    miopenHandle_t handle, const cudnnTensorDescriptor_t x_desc,
    const cudnnFilterDescriptor_t w_desc, const cudnnConvolutionDescriptor_t conv_desc,
    const cudnnTensorDescriptor_t y_desc, int requested_count, int* returned_count,
    cudnnConvolutionFwdAlgoPerf_t* perf_results) {
  size_t ws_size = 0;
  miopenStatus_t status = miopenConvolutionForwardGetWorkSpaceSize(
      handle, w_desc->desc, x_desc, conv_desc->desc, y_desc, &ws_size);
  if (status != miopenStatusSuccess) { return status; }
  oneflow_hip_cudnn::ScratchBuffers scratch;
  void* x = scratch.Alloc(oneflow_hip_cudnn::NumBytes(x_desc));
  void* w = scratch.Alloc(oneflow_hip_cudnn::NumBytes(w_desc->desc));
  void* y = scratch.Alloc(oneflow_hip_cudnn::NumBytes(y_desc));
  void* ws = scratch.Alloc(ws_size);
  if (x == nullptr || w == nullptr || y == nullptr || ws == nullptr) {
    return miopenStatusAllocFailed;
  }
  miopenConvAlgoPerf_t results[oneflow_hip_cudnn::kMaxAlgoCount];
  const int count = std::min(requested_count, oneflow_hip_cudnn::kMaxAlgoCount);
  status = miopenFindConvolutionForwardAlgorithm(handle, x_desc, x, w_desc->desc, w,
                                                 conv_desc->desc, y_desc, y, count, returned_count,
                                                 results, ws, ws_size, false);
  if (status != miopenStatusSuccess) { return status; }
  for (int i = 0; i < *returned_count; ++i) {
    oneflow_hip_cudnn::FillPerf(results[i], results[i].fwd_algo, x_desc, &perf_results[i]);
  }
  return miopenStatusSuccess;
}

inline cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(
    miopenHandle_t handle, const cudnnFilterDescriptor_t w_desc,
    const cudnnTensorDescriptor_t dy_desc, const cudnnConvolutionDescriptor_t conv_desc,
    const cudnnTensorDescriptor_t dx_desc, int requested_count, int* returned_count,
    cudnnConvolutionBwdDataAlgoPerf_t* perf_results) {
  size_t ws_size = 0;
  miopenStatus_t status = miopenConvolutionBackwardDataGetWorkSpaceSize(
      handle, dy_desc, w_desc->desc, conv_desc->desc, dx_desc, &ws_size);
  if (status != miopenStatusSuccess) { return status; }
  oneflow_hip_cudnn::ScratchBuffers scratch;
  void* w = scratch.Alloc(oneflow_hip_cudnn::NumBytes(w_desc->desc));
  void* dy = scratch.Alloc(oneflow_hip_cudnn::NumBytes(dy_desc));
  void* dx = scratch.Alloc(oneflow_hip_cudnn::NumBytes(dx_desc));
  void* ws = scratch.Alloc(ws_size);
  if (w == nullptr || dy == nullptr || dx == nullptr || ws == nullptr) {
    return miopenStatusAllocFailed;
  }
  miopenConvAlgoPerf_t results[oneflow_hip_cudnn::kMaxAlgoCount];
  const int count = std::min(requested_count, oneflow_hip_cudnn::kMaxAlgoCount);
  status = miopenFindConvolutionBackwardDataAlgorithm(handle, dy_desc, dy, w_desc->desc, w,
                                                      conv_desc->desc, dx_desc, dx, count,
                                                      returned_count, results, ws, ws_size, false);
  if (status != miopenStatusSuccess) { return status; }
  for (int i = 0; i < *returned_count; ++i) {
    oneflow_hip_cudnn::FillPerf(results[i], results[i].bwd_data_algo, dx_desc, &perf_results[i]);
  }
  return miopenStatusSuccess;
}

inline cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    miopenHandle_t handle, const cudnnTensorDescriptor_t x_desc,
    const cudnnTensorDescriptor_t dy_desc, const cudnnConvolutionDescriptor_t conv_desc,
    const cudnnFilterDescriptor_t dw_desc, int requested_count, int* returned_count,
    cudnnConvolutionBwdFilterAlgoPerf_t* perf_results) {
  size_t ws_size = 0;
  miopenStatus_t status = miopenConvolutionBackwardWeightsGetWorkSpaceSize(
      handle, dy_desc, x_desc, conv_desc->desc, dw_desc->desc, &ws_size);
  if (status != miopenStatusSuccess) { return status; }
  oneflow_hip_cudnn::ScratchBuffers scratch;
  void* x = scratch.Alloc(oneflow_hip_cudnn::NumBytes(x_desc));
  void* dy = scratch.Alloc(oneflow_hip_cudnn::NumBytes(dy_desc));
  void* dw = scratch.Alloc(oneflow_hip_cudnn::NumBytes(dw_desc->desc));
  void* ws = scratch.Alloc(ws_size);
  if (x == nullptr || dy == nullptr || dw == nullptr || ws == nullptr) {
    return miopenStatusAllocFailed;
  }
  miopenConvAlgoPerf_t results[oneflow_hip_cudnn::kMaxAlgoCount];
  const int count = std::min(requested_count, oneflow_hip_cudnn::kMaxAlgoCount);
  status = miopenFindConvolutionBackwardWeightsAlgorithm(
      handle, dy_desc, dy, x_desc, x, conv_desc->desc, dw_desc->desc, dw, count, returned_count,
      results, ws, ws_size, false);
  if (status != miopenStatusSuccess) { return status; }
  for (int i = 0; i < *returned_count; ++i) {
    oneflow_hip_cudnn::FillPerf(results[i], results[i].bwd_weights_algo, x_desc, &perf_results[i]);
  }
  return miopenStatusSuccess;
}

namespace oneflow_hip_cudnn {

inline void AppendTensor(std::ostringstream* os, miopenTensorDescriptor_t desc) {
  int ndim = 0;
  miopenGetTensorDescriptorSize(desc, &ndim);
  std::vector<int> dims(ndim), strides(ndim);
  miopenDataType_t data_type = miopenFloat;
  miopenGetTensorDescriptor(desc, &data_type, dims.data(), strides.data());
  *os << static_cast<int>(data_type) << ':';
  for (int i = 0; i < ndim; ++i) { *os << dims[i] << '/' << strides[i] << ','; }
  *os << ';';
}

// Returns true the first time a convolution problem in a direction is seen on a handle.
inline bool FirstRunOnHandle(miopenHandle_t handle, int direction, miopenTensorDescriptor_t a,
                             miopenTensorDescriptor_t b, miopenTensorDescriptor_t c,
                             miopenConvolutionDescriptor_t conv) {
  std::ostringstream os;
  os << direction << '|';
  AppendTensor(&os, a);
  AppendTensor(&os, b);
  AppendTensor(&os, c);
  int spatial_dim = 0;
  miopenGetConvolutionSpatialDim(conv, &spatial_dim);
  std::vector<int> pads(spatial_dim), strides(spatial_dim), dilations(spatial_dim);
  miopenConvolutionMode_t mode;
  int returned = 0;
  miopenGetConvolutionNdDescriptor(conv, spatial_dim, &returned, pads.data(), strides.data(),
                                   dilations.data(), &mode);
  int groups = 1;
  miopenGetConvolutionGroupCount(conv, &groups);
  for (int i = 0; i < spatial_dim; ++i) {
    os << pads[i] << '/' << strides[i] << '/' << dilations[i] << ',';
  }
  os << groups;
  FindRecords* records = GetFindRecords();
  std::lock_guard<std::mutex> lock(records->mutex);
  return records->problems[handle].insert(os.str()).second;
}

}  // namespace oneflow_hip_cudnn

// ---- convolution -----------------------------------------------------------------
inline cudnnStatus_t cudnnConvolutionForward(miopenHandle_t handle, const void* alpha,
                                             const cudnnTensorDescriptor_t x_desc, const void* x,
                                             const cudnnFilterDescriptor_t w_desc, const void* w,
                                             const cudnnConvolutionDescriptor_t conv_desc,
                                             cudnnConvolutionFwdAlgo_t algo, void* workspace,
                                             size_t workspace_size, const void* beta,
                                             const cudnnTensorDescriptor_t y_desc, void* y) {
  if (!oneflow_hip_cudnn::IsOne(alpha, y_desc)) { return miopenStatusUnsupportedOp; }
  return oneflow_hip_cudnn::RunWithBeta(handle, beta, y_desc, y, [&](const void* zero, void* out) {
    if (oneflow_hip_cudnn::FirstRunOnHandle(handle, 0, x_desc, w_desc->desc, y_desc,
                                            conv_desc->desc)) {
      miopenConvAlgoPerf_t perf[oneflow_hip_cudnn::kMaxAlgoCount];
      int count = 0;
      const miopenStatus_t status = miopenFindConvolutionForwardAlgorithm(
          handle, x_desc, x, w_desc->desc, w, conv_desc->desc, y_desc, out,
          oneflow_hip_cudnn::kMaxAlgoCount, &count, perf, workspace, workspace_size, false);
      if (status != miopenStatusSuccess) { return status; }
    }
    return miopenConvolutionForward(handle, alpha, x_desc, x, w_desc->desc, w, conv_desc->desc,
                                    algo, zero, y_desc, out, workspace, workspace_size);
  });
}

inline cudnnStatus_t cudnnConvolutionBackwardData(
    miopenHandle_t handle, const void* alpha, const cudnnFilterDescriptor_t w_desc, const void* w,
    const cudnnTensorDescriptor_t dy_desc, const void* dy,
    const cudnnConvolutionDescriptor_t conv_desc, cudnnConvolutionBwdDataAlgo_t algo,
    void* workspace, size_t workspace_size, const void* beta, const cudnnTensorDescriptor_t dx_desc,
    void* dx) {
  if (!oneflow_hip_cudnn::IsOne(alpha, dx_desc)) { return miopenStatusUnsupportedOp; }
  return oneflow_hip_cudnn::RunWithBeta(
      handle, beta, dx_desc, dx, [&](const void* zero, void* out) {
        if (oneflow_hip_cudnn::FirstRunOnHandle(handle, 1, dy_desc, w_desc->desc, dx_desc,
                                                conv_desc->desc)) {
          miopenConvAlgoPerf_t perf[oneflow_hip_cudnn::kMaxAlgoCount];
          int count = 0;
          const miopenStatus_t status = miopenFindConvolutionBackwardDataAlgorithm(
              handle, dy_desc, dy, w_desc->desc, w, conv_desc->desc, dx_desc, out,
              oneflow_hip_cudnn::kMaxAlgoCount, &count, perf, workspace, workspace_size, false);
          if (status != miopenStatusSuccess) { return status; }
        }
        return miopenConvolutionBackwardData(handle, alpha, dy_desc, dy, w_desc->desc, w,
                                             conv_desc->desc, algo, zero, dx_desc, out, workspace,
                                             workspace_size);
      });
}

inline cudnnStatus_t cudnnConvolutionBackwardFilter(
    miopenHandle_t handle, const void* alpha, const cudnnTensorDescriptor_t x_desc, const void* x,
    const cudnnTensorDescriptor_t dy_desc, const void* dy,
    const cudnnConvolutionDescriptor_t conv_desc, cudnnConvolutionBwdFilterAlgo_t algo,
    void* workspace, size_t workspace_size, const void* beta, const cudnnFilterDescriptor_t dw_desc,
    void* dw) {
  if (!oneflow_hip_cudnn::IsOne(alpha, dw_desc->desc)) { return miopenStatusUnsupportedOp; }
  return oneflow_hip_cudnn::RunWithBeta(
      handle, beta, dw_desc->desc, dw, [&](const void* zero, void* out) {
        if (oneflow_hip_cudnn::FirstRunOnHandle(handle, 2, dy_desc, x_desc, dw_desc->desc,
                                                conv_desc->desc)) {
          miopenConvAlgoPerf_t perf[oneflow_hip_cudnn::kMaxAlgoCount];
          int count = 0;
          const miopenStatus_t status = miopenFindConvolutionBackwardWeightsAlgorithm(
              handle, dy_desc, dy, x_desc, x, conv_desc->desc, dw_desc->desc, out,
              oneflow_hip_cudnn::kMaxAlgoCount, &count, perf, workspace, workspace_size, false);
          if (status != miopenStatusSuccess) { return status; }
        }
        return miopenConvolutionBackwardWeights(handle, alpha, dy_desc, dy, x_desc, x,
                                                conv_desc->desc, algo, zero, dw_desc->desc, out,
                                                workspace, workspace_size);
      });
}

// ---- activation (descriptor only; used by the Ex batch-norm API) ----------------
typedef miopenActivationDescriptor_t cudnnActivationDescriptor_t;
typedef enum { CUDNN_ACTIVATION_RELU = 1 } cudnnActivationMode_t;

inline cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t* desc) {
  return miopenCreateActivationDescriptor(desc);
}

inline cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t desc) {
  return miopenDestroyActivationDescriptor(desc);
}

inline cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t desc,
                                                  cudnnActivationMode_t mode, cudnnNanPropagation_t,
                                                  double coef) {
  if (mode != CUDNN_ACTIVATION_RELU) { return miopenStatusUnsupportedOp; }
  return miopenSetActivationDescriptor(desc, miopenActivationRELU, coef, 0.0, 0.0);
}

// ---- pooling -----------------------------------------------------------------------
typedef enum {
  CUDNN_POOLING_MAX = 0,
  CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
  CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2,
} cudnnPoolingMode_t;

struct cudnnPoolingStruct {
  miopenPoolingDescriptor_t desc;
  cudnnPoolingMode_t mode;
};
typedef cudnnPoolingStruct* cudnnPoolingDescriptor_t;

inline cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* desc) {
  *desc = new cudnnPoolingStruct{nullptr, CUDNN_POOLING_MAX};
  const miopenStatus_t status = miopenCreatePoolingDescriptor(&(*desc)->desc);
  if (status != miopenStatusSuccess) {
    delete *desc;
    *desc = nullptr;
  }
  return status;
}

inline cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t desc) {
  if (desc == nullptr) { return miopenStatusSuccess; }
  const miopenStatus_t status = miopenDestroyPoolingDescriptor(desc->desc);
  delete desc;
  return status;
}

inline cudnnStatus_t cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t desc,
                                                 cudnnPoolingMode_t mode, cudnnNanPropagation_t,
                                                 int nb_dims, const int* window, const int* pads,
                                                 const int* strides) {
  miopenPoolingMode_t miopen_mode = miopenPoolingMax;
  switch (mode) {
    case CUDNN_POOLING_MAX: miopen_mode = miopenPoolingMax; break;
    case CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING:
      miopen_mode = miopenPoolingAverageInclusive;
      break;
    case CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING: miopen_mode = miopenPoolingAverage; break;
    default: return miopenStatusBadParm;
  }
  desc->mode = mode;
  miopenStatus_t status =
      miopenSetNdPoolingDescriptor(desc->desc, miopen_mode, nb_dims, window, pads, strides);
  if (status != miopenStatusSuccess) { return status; }
  // Record the arg-max as a position inside the window; 32-bit indices cover any window.
  status = miopenSetPoolingWorkSpaceIndexMode(desc->desc, miopenPoolingWorkspaceIndexMask);
  if (status != miopenStatusSuccess) { return status; }
  return miopenSetPoolingIndexType(desc->desc, miopenIndexUint32);
}

inline cudnnStatus_t cudnnPoolingForward(miopenHandle_t handle,
                                         const cudnnPoolingDescriptor_t pool_desc,
                                         const void* alpha, const cudnnTensorDescriptor_t x_desc,
                                         const void* x, const void* beta,
                                         const cudnnTensorDescriptor_t y_desc, void* y) {
  if (!oneflow_hip_cudnn::IsOne(alpha, y_desc)) { return miopenStatusUnsupportedOp; }
  hipStream_t stream = nullptr;
  miopenStatus_t status = miopenGetStream(handle, &stream);
  if (status != miopenStatusSuccess) { return status; }
  // MIOpen pools NHWC tensors by transposing through the workspace, so the forward pass needs one
  // even when it records no indices.
  size_t workspace_size = 0;
  status = miopenPoolingGetWorkSpaceSizeV2(pool_desc->desc, y_desc, &workspace_size);
  if (status != miopenStatusSuccess) { return status; }
  void* workspace = nullptr;
  if (workspace_size > 0 && hipMallocAsync(&workspace, workspace_size, stream) != hipSuccess) {
    return miopenStatusAllocFailed;
  }
  status =
      oneflow_hip_cudnn::RunWithBeta(handle, beta, y_desc, y, [&](const void* zero, void* out) {
        return miopenPoolingForward(handle, pool_desc->desc, alpha, x_desc, x, zero, y_desc, out,
                                    false, workspace, workspace_size);
      });
  if (workspace != nullptr) { (void)hipFreeAsync(workspace, stream); }
  return status;
}

inline cudnnStatus_t cudnnPoolingBackward(miopenHandle_t handle,
                                          const cudnnPoolingDescriptor_t pool_desc,
                                          const void* alpha, const cudnnTensorDescriptor_t y_desc,
                                          const void* y, const cudnnTensorDescriptor_t dy_desc,
                                          const void* dy, const cudnnTensorDescriptor_t x_desc,
                                          const void* x, const void* beta,
                                          const cudnnTensorDescriptor_t dx_desc, void* dx) {
  if (!oneflow_hip_cudnn::IsOne(alpha, dx_desc)) { return miopenStatusUnsupportedOp; }
  hipStream_t stream = nullptr;
  miopenStatus_t status = miopenGetStream(handle, &stream);
  if (status != miopenStatusSuccess) { return status; }
  void* workspace = nullptr;
  void* y_tmp = nullptr;
  size_t workspace_size = 0;
  status = miopenPoolingGetWorkSpaceSizeV2(pool_desc->desc, y_desc, &workspace_size);
  if (status != miopenStatusSuccess) { return status; }
  if (workspace_size > 0 && hipMallocAsync(&workspace, workspace_size, stream) != hipSuccess) {
    return miopenStatusAllocFailed;
  }
  if (pool_desc->mode == CUDNN_POOLING_MAX) {
    if (hipMallocAsync(&y_tmp, oneflow_hip_cudnn::NumBytes(y_desc), stream) != hipSuccess) {
      if (workspace != nullptr) { (void)hipFreeAsync(workspace, stream); }
      return miopenStatusAllocFailed;
    }
    status = miopenPoolingForward(handle, pool_desc->desc, oneflow_hip_cudnn::OnePtr(y_desc),
                                  x_desc, x, oneflow_hip_cudnn::ZeroPtr(y_desc), y_desc, y_tmp,
                                  true, workspace, workspace_size);
  }
  if (status == miopenStatusSuccess) {
    status =
        oneflow_hip_cudnn::RunWithBeta(handle, beta, dx_desc, dx, [&](const void* zero, void* out) {
          return miopenPoolingBackward(handle, pool_desc->desc, alpha, y_desc,
                                       y_tmp != nullptr ? y_tmp : y, dy_desc, dy, x_desc, x, zero,
                                       dx_desc, out, workspace);
        });
  }
  if (workspace != nullptr) { (void)hipFreeAsync(workspace, stream); }
  if (y_tmp != nullptr) { (void)hipFreeAsync(y_tmp, stream); }
  return status;
}

// ---- batch normalization -------------------------------------------------------
typedef miopenBatchNormMode_t cudnnBatchNormMode_t;
#define CUDNN_BATCHNORM_PER_ACTIVATION miopenBNPerActivation
#define CUDNN_BATCHNORM_SPATIAL miopenBNSpatial
#define CUDNN_BATCHNORM_SPATIAL_PERSISTENT miopenBNSpatial

// MIOpen has no double batch norm (it finds no solver), and its derived parameter descriptor is
// always float. For double tensors the derived descriptor is double, as on cuDNN, and the batch
// norm calls run on the kernels below. MIOpen also refuses training when every parameter slot
// normalizes a single value (N * H * W == 1 in spatial mode, N == 1 per activation), which cuDNN
// computes; those calls run on the same kernels for every type. They are only compiled in
// HIP-language translation units, which are the only callers.
#if defined(__HIPCC__)

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

namespace oneflow_hip_cudnn {

// Tensor element type T and the type P of the parameters and the arithmetic: float for the 16-bit
// types, whose parameter tensors are float, as on cuDNN.
template<typename T>
struct BatchNormType {
  using P = T;
  __device__ static P Load(T v) { return v; }
  __device__ static T Store(P v) { return v; }
};

template<>
struct BatchNormType<__half> {
  using P = float;
  __device__ static P Load(__half v) { return __half2float(v); }
  __device__ static __half Store(P v) { return __float2half(v); }
};

template<>
struct BatchNormType<__hip_bfloat16> {
  using P = float;
  __device__ static P Load(__hip_bfloat16 v) { return __bfloat162float(v); }
  __device__ static __hip_bfloat16 Store(P v) { return __float2bfloat16(v); }
};

// A 4-D tensor seen as P parameter slots by R reduced elements: spatial mode reduces over N, H and
// W for each channel, per-activation mode over N for each (C, H, W).
struct BatchNormLayout {
  int64_t p = 0;
  int64_t r = 0;
  int dims[4] = {1, 1, 1, 1};
  bool spatial = true;
};

struct BatchNormStrides {
  int64_t s[4] = {0, 0, 0, 0};
};

inline bool GetBatchNorm4d(miopenTensorDescriptor_t desc, miopenDataType_t* data_type, int* dims,
                           int64_t* strides) {
  int ndim = 0;
  if (miopenGetTensorDescriptorSize(desc, &ndim) != miopenStatusSuccess || ndim != 4) {
    return false;
  }
  int st[4];
  if (miopenGetTensorDescriptor(desc, data_type, dims, st) != miopenStatusSuccess) { return false; }
  for (int i = 0; i < 4; ++i) { strides[i] = st[i]; }
  return true;
}

inline bool MakeBatchNormLayout(miopenTensorDescriptor_t x_desc, miopenBatchNormMode_t mode,
                                BatchNormLayout* layout, BatchNormStrides* x_strides,
                                miopenDataType_t* data_type = nullptr) {
  miopenDataType_t type;
  if (!GetBatchNorm4d(x_desc, &type, layout->dims, x_strides->s)) { return false; }
  if (data_type != nullptr) { *data_type = type; }
  const int* d = layout->dims;
  layout->spatial = (mode == miopenBNSpatial);
  layout->p = layout->spatial ? d[1] : static_cast<int64_t>(d[1]) * d[2] * d[3];
  layout->r = layout->spatial ? static_cast<int64_t>(d[0]) * d[2] * d[3] : d[0];
  return true;
}

inline bool GetBatchNormStrides(miopenTensorDescriptor_t desc, const BatchNormLayout& layout,
                                BatchNormStrides* strides) {
  miopenDataType_t data_type;
  int dims[4];
  if (!GetBatchNorm4d(desc, &data_type, dims, strides->s)) { return false; }
  for (int i = 0; i < 4; ++i) {
    if (dims[i] != layout.dims[i]) { return false; }
  }
  return true;
}

__device__ __forceinline__ int64_t BatchNormOffset(const BatchNormLayout& l,
                                                   const BatchNormStrides& s, int64_t p,
                                                   int64_t r) {
  int64_t n, c, h, w;
  if (l.spatial) {
    c = p;
    w = r % l.dims[3];
    h = (r / l.dims[3]) % l.dims[2];
    n = r / (static_cast<int64_t>(l.dims[3]) * l.dims[2]);
  } else {
    n = r;
    w = p % l.dims[3];
    h = (p / l.dims[3]) % l.dims[2];
    c = p / (static_cast<int64_t>(l.dims[3]) * l.dims[2]);
  }
  return n * s.s[0] + c * s.s[1] + h * s.s[2] + w * s.s[3];
}

constexpr int kBatchNormBlock = 256;

// Sum over the block; every thread gets the result. Wavefront-width independent.
template<typename T>
__device__ __forceinline__ T BatchNormBlockSum(T v, T* shared) {
  __syncthreads();
  shared[threadIdx.x] = v;
  __syncthreads();
  for (int stride = kBatchNormBlock / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) { shared[threadIdx.x] += shared[threadIdx.x + stride]; }
    __syncthreads();
  }
  return shared[0];
}

template<typename T, typename P = typename BatchNormType<T>::P>
__global__ void __launch_bounds__(kBatchNormBlock)
    BatchNormForwardKernel(BatchNormLayout l, BatchNormStrides xs, BatchNormStrides ys, const T* x,
                           T* y, P beta, const P* scale, const P* bias, bool training, P factor,
                           P* running_mean, P* running_variance, const P* estimated_mean,
                           const P* estimated_variance, P epsilon, P* save_mean,
                           P* save_inv_variance) {
  using Type = BatchNormType<T>;
  __shared__ P shared[kBatchNormBlock];
  for (int64_t p = blockIdx.x; p < l.p; p += gridDim.x) {
    P mean = 0;
    P inv_std = 0;
    if (training) {
      P sum = 0;
      for (int64_t r = threadIdx.x; r < l.r; r += kBatchNormBlock) {
        sum += Type::Load(x[BatchNormOffset(l, xs, p, r)]);
      }
      mean = BatchNormBlockSum(sum, shared) / l.r;
      P sq = 0;
      for (int64_t r = threadIdx.x; r < l.r; r += kBatchNormBlock) {
        const P d = Type::Load(x[BatchNormOffset(l, xs, p, r)]) - mean;
        sq += d * d;
      }
      sq = BatchNormBlockSum(sq, shared);
      inv_std = 1.0 / sqrt(sq / l.r + epsilon);
      if (threadIdx.x == 0) {
        if (save_mean != nullptr) { save_mean[p] = mean; }
        if (save_inv_variance != nullptr) { save_inv_variance[p] = inv_std; }
        if (running_mean != nullptr) {
          running_mean[p] = (1.0 - factor) * running_mean[p] + factor * mean;
        }
        if (running_variance != nullptr) {
          const P unbiased = sq / (l.r > 1 ? l.r - 1 : 1);
          running_variance[p] = (1.0 - factor) * running_variance[p] + factor * unbiased;
        }
      }
    } else {
      mean = estimated_mean[p];
      inv_std = 1.0 / sqrt(estimated_variance[p] + epsilon);
    }
    const P a = scale[p] * inv_std;
    for (int64_t r = threadIdx.x; r < l.r; r += kBatchNormBlock) {
      T* out = y + BatchNormOffset(l, ys, p, r);
      const P v = (Type::Load(x[BatchNormOffset(l, xs, p, r)]) - mean) * a + bias[p];
      *out = Type::Store((beta == 0) ? v : v + beta * Type::Load(*out));
    }
  }
}

template<typename T, typename P = typename BatchNormType<T>::P>
__global__ void __launch_bounds__(kBatchNormBlock)
    BatchNormBackwardKernel(BatchNormLayout l, BatchNormStrides xs, BatchNormStrides dys,
                            BatchNormStrides dxs, const T* x, const T* dy, T* dx, const P* scale,
                            P* scale_diff, P* bias_diff, const P* saved_mean,
                            const P* saved_inv_variance) {
  using Type = BatchNormType<T>;
  __shared__ P shared[kBatchNormBlock];
  for (int64_t p = blockIdx.x; p < l.p; p += gridDim.x) {
    const P mean = saved_mean[p];
    const P inv_std = saved_inv_variance[p];
    P sum_dy = 0;
    P sum_dy_xhat = 0;
    for (int64_t r = threadIdx.x; r < l.r; r += kBatchNormBlock) {
      const P g = Type::Load(dy[BatchNormOffset(l, dys, p, r)]);
      sum_dy += g;
      sum_dy_xhat += g * (Type::Load(x[BatchNormOffset(l, xs, p, r)]) - mean) * inv_std;
    }
    sum_dy = BatchNormBlockSum(sum_dy, shared);
    sum_dy_xhat = BatchNormBlockSum(sum_dy_xhat, shared);
    if (threadIdx.x == 0) {
      scale_diff[p] = sum_dy_xhat;
      bias_diff[p] = sum_dy;
    }
    const P a = scale[p] * inv_std;
    const P mean_dy = sum_dy / l.r;
    const P mean_dy_xhat = sum_dy_xhat / l.r;
    for (int64_t r = threadIdx.x; r < l.r; r += kBatchNormBlock) {
      const P xhat = (Type::Load(x[BatchNormOffset(l, xs, p, r)]) - mean) * inv_std;
      dx[BatchNormOffset(l, dxs, p, r)] = Type::Store(
          a * (Type::Load(dy[BatchNormOffset(l, dys, p, r)]) - mean_dy - xhat * mean_dy_xhat));
    }
  }
}

inline unsigned int BatchNormGrid(int64_t p) {
  return static_cast<unsigned int>(std::max<int64_t>(1, std::min<int64_t>(p, 65535)));
}

// Whether the call runs on the kernels above rather than on MIOpen.
inline bool UseNativeBatchNorm(miopenTensorDescriptor_t x_desc, miopenBatchNormMode_t mode,
                               bool training) {
  if (IsDoubleScalar(x_desc)) { return true; }
  BatchNormLayout layout;
  BatchNormStrides strides;
  return training && MakeBatchNormLayout(x_desc, mode, &layout, &strides) && layout.r == 1;
}

template<typename T, typename P = typename BatchNormType<T>::P>
void LaunchBatchNormForward(hipStream_t stream, const BatchNormLayout& layout,
                            const BatchNormStrides& xs, const BatchNormStrides& ys, const void* x,
                            void* y, double beta, const void* scale, const void* bias,
                            bool training, double factor, void* running_mean,
                            void* running_variance, const void* estimated_mean,
                            const void* estimated_variance, double epsilon, void* save_mean,
                            void* save_inv_variance) {
  hipLaunchKernelGGL(
      BatchNormForwardKernel<T>, dim3(BatchNormGrid(layout.p)), dim3(kBatchNormBlock), 0, stream,
      layout, xs, ys, static_cast<const T*>(x), static_cast<T*>(y), static_cast<P>(beta),
      static_cast<const P*>(scale), static_cast<const P*>(bias), training, static_cast<P>(factor),
      static_cast<P*>(running_mean), static_cast<P*>(running_variance),
      static_cast<const P*>(estimated_mean), static_cast<const P*>(estimated_variance),
      static_cast<P>(epsilon), static_cast<P*>(save_mean), static_cast<P*>(save_inv_variance));
}

inline miopenStatus_t BatchNormNativeForward(miopenHandle_t handle, miopenBatchNormMode_t mode,
                                             const void* beta, miopenTensorDescriptor_t x_desc,
                                             const void* x, miopenTensorDescriptor_t y_desc,
                                             void* y, const void* scale, const void* bias,
                                             bool training, double factor, void* running_mean,
                                             void* running_variance, const void* estimated_mean,
                                             const void* estimated_variance, double epsilon,
                                             void* save_mean, void* save_inv_variance) {
  BatchNormLayout layout;
  BatchNormStrides xs, ys;
  miopenDataType_t data_type;
  if (!MakeBatchNormLayout(x_desc, mode, &layout, &xs, &data_type)
      || !GetBatchNormStrides(y_desc, layout, &ys)) {
    return miopenStatusBadParm;
  }
  hipStream_t stream = nullptr;
  const miopenStatus_t status = miopenGetStream(handle, &stream);
  if (status != miopenStatusSuccess) { return status; }
  const double beta_value = ReadScalar(beta, data_type == miopenDouble);
  auto launch = [&](auto* type) {
    using T = std::remove_pointer_t<decltype(type)>;
    LaunchBatchNormForward<T>(stream, layout, xs, ys, x, y, beta_value, scale, bias, training,
                              factor, running_mean, running_variance, estimated_mean,
                              estimated_variance, epsilon, save_mean, save_inv_variance);
  };
  switch (data_type) {
    case miopenDouble: launch(static_cast<double*>(nullptr)); break;
    case miopenFloat: launch(static_cast<float*>(nullptr)); break;
    case miopenHalf: launch(static_cast<__half*>(nullptr)); break;
    case miopenBFloat16: launch(static_cast<__hip_bfloat16*>(nullptr)); break;
    default: return miopenStatusNotImplemented;
  }
  return hipGetLastError() == hipSuccess ? miopenStatusSuccess : miopenStatusUnknownError;
}

inline miopenStatus_t BatchNormNativeBackward(miopenHandle_t handle, miopenBatchNormMode_t mode,
                                              miopenTensorDescriptor_t x_desc, const void* x,
                                              miopenTensorDescriptor_t dy_desc, const void* dy,
                                              miopenTensorDescriptor_t dx_desc, void* dx,
                                              const void* scale, void* scale_diff, void* bias_diff,
                                              const void* saved_mean,
                                              const void* saved_inv_variance) {
  if (saved_mean == nullptr || saved_inv_variance == nullptr) { return miopenStatusBadParm; }
  BatchNormLayout layout;
  BatchNormStrides xs, dys, dxs;
  miopenDataType_t data_type;
  if (!MakeBatchNormLayout(x_desc, mode, &layout, &xs, &data_type)
      || !GetBatchNormStrides(dy_desc, layout, &dys)
      || !GetBatchNormStrides(dx_desc, layout, &dxs)) {
    return miopenStatusBadParm;
  }
  hipStream_t stream = nullptr;
  const miopenStatus_t status = miopenGetStream(handle, &stream);
  if (status != miopenStatusSuccess) { return status; }
  auto launch = [&](auto* type) {
    using T = std::remove_pointer_t<decltype(type)>;
    using P = typename BatchNormType<T>::P;
    hipLaunchKernelGGL(BatchNormBackwardKernel<T>, dim3(BatchNormGrid(layout.p)),
                       dim3(kBatchNormBlock), 0, stream, layout, xs, dys, dxs,
                       static_cast<const T*>(x), static_cast<const T*>(dy), static_cast<T*>(dx),
                       static_cast<const P*>(scale), static_cast<P*>(scale_diff),
                       static_cast<P*>(bias_diff), static_cast<const P*>(saved_mean),
                       static_cast<const P*>(saved_inv_variance));
  };
  switch (data_type) {
    case miopenDouble: launch(static_cast<double*>(nullptr)); break;
    case miopenFloat: launch(static_cast<float*>(nullptr)); break;
    case miopenHalf: launch(static_cast<__half*>(nullptr)); break;
    case miopenBFloat16: launch(static_cast<__hip_bfloat16*>(nullptr)); break;
    default: return miopenStatusNotImplemented;
  }
  return hipGetLastError() == hipSuccess ? miopenStatusSuccess : miopenStatusUnknownError;
}

}  // namespace oneflow_hip_cudnn

inline cudnnStatus_t cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derived_desc,
                                                   const cudnnTensorDescriptor_t x_desc,
                                                   cudnnBatchNormMode_t mode) {
  if (oneflow_hip_cudnn::IsDoubleScalar(x_desc)) {
    oneflow_hip_cudnn::BatchNormLayout layout;
    oneflow_hip_cudnn::BatchNormStrides strides;
    if (!oneflow_hip_cudnn::MakeBatchNormLayout(x_desc, mode, &layout, &strides)) {
      return miopenStatusBadParm;
    }
    const int* d = layout.dims;
    return layout.spatial
               ? miopenSet4dTensorDescriptor(derived_desc, miopenDouble, 1, d[1], 1, 1)
               : miopenSet4dTensorDescriptor(derived_desc, miopenDouble, 1, d[1], d[2], d[3]);
  }
  return miopenDeriveBNTensorDescriptor(derived_desc, x_desc, mode);
}

inline cudnnStatus_t cudnnBatchNormalizationForwardTraining(
    miopenHandle_t handle, cudnnBatchNormMode_t mode, const void* alpha, const void* beta,
    const cudnnTensorDescriptor_t x_desc, const void* x, const cudnnTensorDescriptor_t y_desc,
    void* y, const cudnnTensorDescriptor_t param_desc, const void* scale, const void* bias,
    double exponential_average_factor, void* running_mean, void* running_variance, double epsilon,
    void* save_mean, void* save_inv_variance) {
  if (!oneflow_hip_cudnn::IsOne(alpha, y_desc)) { return miopenStatusUnsupportedOp; }
  if (oneflow_hip_cudnn::UseNativeBatchNorm(x_desc, mode, /*training=*/true)) {
    return oneflow_hip_cudnn::BatchNormNativeForward(
        handle, mode, beta, x_desc, x, y_desc, y, scale, bias, /*training=*/true,
        exponential_average_factor, running_mean, running_variance, nullptr, nullptr, epsilon,
        save_mean, save_inv_variance);
  }
  return oneflow_hip_cudnn::RunWithBeta(handle, beta, y_desc, y, [&](const void* zero, void* out) {
    return miopenBatchNormalizationForwardTraining(
        handle, mode, const_cast<void*>(alpha), const_cast<void*>(zero), x_desc, x, y_desc, out,
        param_desc, const_cast<void*>(scale), const_cast<void*>(bias), exponential_average_factor,
        running_mean, running_variance, epsilon, save_mean, save_inv_variance);
  });
}

inline cudnnStatus_t cudnnBatchNormalizationForwardInference(
    miopenHandle_t handle, cudnnBatchNormMode_t mode, const void* alpha, const void* beta,
    const cudnnTensorDescriptor_t x_desc, const void* x, const cudnnTensorDescriptor_t y_desc,
    void* y, const cudnnTensorDescriptor_t param_desc, const void* scale, const void* bias,
    const void* estimated_mean, const void* estimated_variance, double epsilon) {
  if (!oneflow_hip_cudnn::IsOne(alpha, y_desc)) { return miopenStatusUnsupportedOp; }
  if (oneflow_hip_cudnn::UseNativeBatchNorm(x_desc, mode, /*training=*/false)) {
    return oneflow_hip_cudnn::BatchNormNativeForward(
        handle, mode, beta, x_desc, x, y_desc, y, scale, bias, /*training=*/false, 0.0, nullptr,
        nullptr, estimated_mean, estimated_variance, epsilon, nullptr, nullptr);
  }
  return oneflow_hip_cudnn::RunWithBeta(handle, beta, y_desc, y, [&](const void* zero, void* out) {
    return miopenBatchNormalizationForwardInference(
        handle, mode, const_cast<void*>(alpha), const_cast<void*>(zero), x_desc, x, y_desc, out,
        param_desc, const_cast<void*>(scale), const_cast<void*>(bias),
        const_cast<void*>(estimated_mean), const_cast<void*>(estimated_variance), epsilon);
  });
}

inline cudnnStatus_t cudnnBatchNormalizationBackward(
    miopenHandle_t handle, cudnnBatchNormMode_t mode, const void* alpha_data_diff,
    const void* beta_data_diff, const void* alpha_param_diff, const void* beta_param_diff,
    const cudnnTensorDescriptor_t x_desc, const void* x, const cudnnTensorDescriptor_t dy_desc,
    const void* dy, const cudnnTensorDescriptor_t dx_desc, void* dx,
    const cudnnTensorDescriptor_t param_desc, const void* scale, void* scale_diff, void* bias_diff,
    double epsilon, const void* saved_mean, const void* saved_inv_variance) {
  if (!oneflow_hip_cudnn::IsOne(alpha_data_diff, dx_desc)
      || !oneflow_hip_cudnn::IsZero(beta_data_diff, dx_desc)
      || !oneflow_hip_cudnn::IsOne(alpha_param_diff, dx_desc)
      || !oneflow_hip_cudnn::IsZero(beta_param_diff, dx_desc)) {
    return miopenStatusUnsupportedOp;
  }
  if (oneflow_hip_cudnn::UseNativeBatchNorm(x_desc, mode, /*training=*/true)) {
    return oneflow_hip_cudnn::BatchNormNativeBackward(handle, mode, x_desc, x, dy_desc, dy, dx_desc,
                                                      dx, scale, scale_diff, bias_diff, saved_mean,
                                                      saved_inv_variance);
  }
  return miopenBatchNormalizationBackward(handle, mode, alpha_data_diff, beta_data_diff,
                                          alpha_param_diff, beta_param_diff, x_desc, x, dy_desc, dy,
                                          dx_desc, dx, param_desc, scale, scale_diff, bias_diff,
                                          epsilon, saved_mean, saved_inv_variance);
}

#endif  // defined(__HIPCC__)
