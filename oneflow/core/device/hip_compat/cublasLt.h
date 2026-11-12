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
#include "oneflow/core/device/cuda_to_hip.h"
#include <cublas_v2.h>
#include <hipblaslt/hipblaslt.h>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

// The subset of the cuBLASLt API that oneflow's fused matmul kernels call, implemented on
// hipBLASLt. Only the HIP build sees this header (it shadows <cublasLt.h> on the HIP include path).
//
// The handle and preference objects are hipBLASLt's own. The matmul descriptor and the matrix
// layout are small structs that also record the attributes oneflow sets: hipBLASLt fixes the
// compute type when the descriptor is created (cuBLASLt lets it change afterwards), and it cannot
// read a layout's data type back, which the fallback kernels below need.
//
// Epilogues run in hipBLASLt where it has them for the data type. The rest run as a plain hipBLASLt
// GEMM followed by a fallback kernel in this file:
//  - RELU_AUX / RELU_AUX_BIAS and DRELU / DRELU_BGRAD. cuBLASLt's auxiliary output is a ReLU bit
//    mask (one bit per element, leading dimension a multiple of 128) that oneflow allocates and
//    that oneflow's own dropout kernel also writes. hipBLASLt's auxiliary output is the
//    pre-activation matrix instead, so these always use the fallback, which reads and writes the
//    bit mask.
//  - BIAS, RELU, RELU_BIAS, BGRADA and BGRADB for data types hipBLASLt has no such kernel for.
//  - COMPUTE_16F, which hipBLASLt rejects: the GEMM runs with 32-bit compute and float scalars.
//
// A 16-bit GEMM for which OfUseLongKGemm holds runs as OfLongKGemm, as in oneflow's matmul, with
// every requested epilogue in the fallback kernels after it. OfLongKGemm also calls rocBLAS, so
// each handle has a rocBLAS handle of its own, created with it: a kernel may first run while its
// stream is being captured into a graph, when the memory a rocBLAS handle allocates cannot be.

// The handle is created and destroyed by host code (the stream object); the rest is used only by
// the HIP-language kernels.
typedef hipblasLtHandle_t cublasLtHandle_t;

namespace oneflow_hip_cublaslt {

inline std::mutex& BlasHandlesMutex() {
  static std::mutex mutex;
  return mutex;
}

inline std::unordered_map<hipblasLtHandle_t, hipblasHandle_t>& BlasHandles() {
  static std::unordered_map<hipblasLtHandle_t, hipblasHandle_t> handles;
  return handles;
}

// The rocBLAS handle created with `handle`, or null.
inline hipblasHandle_t BlasHandle(hipblasLtHandle_t handle) {
  std::lock_guard<std::mutex> lock(BlasHandlesMutex());
  const auto it = BlasHandles().find(handle);
  return it == BlasHandles().end() ? nullptr : it->second;
}

// hipblasLtCreate zero-fills the memory it allocates with a synchronous hipMemset on the null
// stream. While any thread is capturing a blocking stream into a graph, that memset fails with
// hipErrorStreamCaptureImplicit and hipBLASLt exits the process. miopenCreate creates a hipBLASLt
// handle as well, and so do hipblasCreate and hipsolverDnCreate when rocBLAS uses hipBLASLt.
// hipfftXtMakePlanMany fails the same way, but returns HIPFFT_PARSE_ERROR and leaves the other
// thread's capture invalidated. So a stream holds CaptureMutex shared while it captures, and
// creating any of these handles or plans holds it exclusively.
inline std::shared_mutex& CaptureMutex() {
  static std::shared_mutex mutex;
  return mutex;
}

inline bool& ThreadIsCapturing() {
  thread_local bool capturing = false;
  return capturing;
}

inline void BeginCapture() {
  CaptureMutex().lock_shared();
  ThreadIsCapturing() = true;
}

inline void EndCapture() {
  ThreadIsCapturing() = false;
  CaptureMutex().unlock_shared();
}

class HandleCreationGuard final {
 public:
  // A thread that is capturing cannot wait for its own capture to end.
  HandleCreationGuard() : lock_(CaptureMutex(), std::defer_lock) {
    if (!ThreadIsCapturing()) { lock_.lock(); }
  }

 private:
  std::unique_lock<std::shared_mutex> lock_;
};

}  // namespace oneflow_hip_cublaslt

inline hipblasStatus_t cublasLtCreate(hipblasLtHandle_t* handle) {
  oneflow_hip_cublaslt::HandleCreationGuard guard;
  hipblasStatus_t status = hipblasLtCreate(handle);
  if (status != HIPBLAS_STATUS_SUCCESS) { return status; }
  hipblasHandle_t blas = nullptr;
  status = hipblasCreate(&blas);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    hipblasLtDestroy(*handle);
    *handle = nullptr;
    return status;
  }
  std::lock_guard<std::mutex> lock(oneflow_hip_cublaslt::BlasHandlesMutex());
  oneflow_hip_cublaslt::BlasHandles()[*handle] = blas;
  return HIPBLAS_STATUS_SUCCESS;
}

inline hipblasStatus_t cublasLtDestroy(hipblasLtHandle_t handle) {
  hipblasHandle_t blas = nullptr;
  {
    std::lock_guard<std::mutex> lock(oneflow_hip_cublaslt::BlasHandlesMutex());
    const auto it = oneflow_hip_cublaslt::BlasHandles().find(handle);
    if (it != oneflow_hip_cublaslt::BlasHandles().end()) {
      blas = it->second;
      oneflow_hip_cublaslt::BlasHandles().erase(it);
    }
  }
  const hipblasStatus_t status = blas == nullptr ? HIPBLAS_STATUS_SUCCESS : hipblasDestroy(blas);
  const hipblasStatus_t lt_status = hipblasLtDestroy(handle);
  return status != HIPBLAS_STATUS_SUCCESS ? status : lt_status;
}

#if defined(__HIPCC__)

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <algorithm>

typedef hipblasLtEpilogue_t cublasLtEpilogue_t;
#define CUBLASLT_EPILOGUE_DEFAULT HIPBLASLT_EPILOGUE_DEFAULT
#define CUBLASLT_EPILOGUE_RELU HIPBLASLT_EPILOGUE_RELU
#define CUBLASLT_EPILOGUE_BIAS HIPBLASLT_EPILOGUE_BIAS
#define CUBLASLT_EPILOGUE_RELU_BIAS HIPBLASLT_EPILOGUE_RELU_BIAS
#define CUBLASLT_EPILOGUE_RELU_AUX HIPBLASLT_EPILOGUE_RELU_AUX
#define CUBLASLT_EPILOGUE_RELU_AUX_BIAS HIPBLASLT_EPILOGUE_RELU_AUX_BIAS
#define CUBLASLT_EPILOGUE_DRELU HIPBLASLT_EPILOGUE_DRELU
#define CUBLASLT_EPILOGUE_DRELU_BGRAD HIPBLASLT_EPILOGUE_DRELU_BGRAD
#define CUBLASLT_EPILOGUE_BGRADA HIPBLASLT_EPILOGUE_BGRADA
#define CUBLASLT_EPILOGUE_BGRADB HIPBLASLT_EPILOGUE_BGRADB

typedef hipblasLtMatmulDescAttributes_t cublasLtMatmulDescAttributes_t;
#define CUBLASLT_MATMUL_DESC_TRANSA HIPBLASLT_MATMUL_DESC_TRANSA
#define CUBLASLT_MATMUL_DESC_TRANSB HIPBLASLT_MATMUL_DESC_TRANSB
#define CUBLASLT_MATMUL_DESC_EPILOGUE HIPBLASLT_MATMUL_DESC_EPILOGUE
#define CUBLASLT_MATMUL_DESC_BIAS_POINTER HIPBLASLT_MATMUL_DESC_BIAS_POINTER
#define CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER
#define CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD
// hipBLASLt has no compute-type attribute (the type is fixed at creation); an unused value of the
// attribute enum stands for it and the descriptor wrapper below handles it.
#define CUBLASLT_MATMUL_DESC_COMPUTE_TYPE static_cast<hipblasLtMatmulDescAttributes_t>(127)

#define CUBLASLT_MATRIX_LAYOUT_TYPE HIPBLASLT_MATRIX_LAYOUT_TYPE
#define CUBLASLT_MATRIX_LAYOUT_ROWS HIPBLASLT_MATRIX_LAYOUT_ROWS
#define CUBLASLT_MATRIX_LAYOUT_COLS HIPBLASLT_MATRIX_LAYOUT_COLS
#define CUBLASLT_MATRIX_LAYOUT_LD HIPBLASLT_MATRIX_LAYOUT_LD

typedef hipblasLtMatmulPreference_t cublasLtMatmulPreference_t;
#define CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
// Pointer-mode masks restrict cuBLASLt's search to host scalars, which is all hipBLASLt offers; the
// attribute is accepted and ignored (HIPBLASLT_MATMUL_PREF_MAX is not an attribute itself).
#define CUBLASLT_MATMUL_PREF_POINTER_MODE_MASK HIPBLASLT_MATMUL_PREF_MAX
#define CUBLASLT_POINTER_MODE_MASK_HOST 1
#define cublasLtMatmulPreferenceCreate hipblasLtMatmulPreferenceCreate
#define cublasLtMatmulPreferenceDestroy hipblasLtMatmulPreferenceDestroy

typedef hipblasLtMatmulHeuristicResult_t cublasLtMatmulHeuristicResult_t;

struct cublasLtMatrixLayoutOpaque {
  hipblasLtMatrixLayout_t layout;
  hipDataType type;
  uint64_t rows;
  uint64_t cols;
  int64_t ld;
};
typedef cublasLtMatrixLayoutOpaque* cublasLtMatrixLayout_t;

inline hipblasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* layout, hipDataType type,
                                                  uint64_t rows, uint64_t cols, int64_t ld) {
  *layout = new cublasLtMatrixLayoutOpaque{nullptr, type, rows, cols, ld};
  const hipblasStatus_t status =
      hipblasLtMatrixLayoutCreate(&(*layout)->layout, type, rows, cols, ld);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    delete *layout;
    *layout = nullptr;
  }
  return status;
}

inline hipblasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t layout) {
  if (layout == nullptr) { return HIPBLAS_STATUS_SUCCESS; }
  const hipblasStatus_t status = hipblasLtMatrixLayoutDestroy(layout->layout);
  delete layout;
  return status;
}

inline hipblasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t layout,
                                                        hipblasLtMatrixLayoutAttribute_t attr,
                                                        const void* buf, size_t size) {
  switch (attr) {
    case HIPBLASLT_MATRIX_LAYOUT_TYPE:
      layout->type = static_cast<hipDataType>(*static_cast<const uint32_t*>(buf));
      break;
    case HIPBLASLT_MATRIX_LAYOUT_ROWS: layout->rows = *static_cast<const uint64_t*>(buf); break;
    case HIPBLASLT_MATRIX_LAYOUT_COLS: layout->cols = *static_cast<const uint64_t*>(buf); break;
    case HIPBLASLT_MATRIX_LAYOUT_LD: layout->ld = *static_cast<const int64_t*>(buf); break;
    default: break;
  }
  return hipblasLtMatrixLayoutSetAttribute(layout->layout, attr, buf, size);
}

struct cublasLtMatmulDescOpaque {
  hipblasLtMatmulDesc_t desc;
  hipblasComputeType_t compute_type;
  hipblasOperation_t trans_a;
  hipblasOperation_t trans_b;
  hipblasLtEpilogue_t epilogue;
  const void* bias;
  const void* aux;
  int64_t aux_ld;
};
typedef cublasLtMatmulDescOpaque* cublasLtMatmulDesc_t;

namespace oneflow_hip_cublaslt {

inline hipDataType ScaleType(hipblasComputeType_t compute_type) {
  return compute_type == HIPBLAS_COMPUTE_64F ? HIP_R_64F : HIP_R_32F;
}

// COMPUTE_16F (half accumulation) is not available in hipBLASLt; accumulate in float instead.
inline hipblasComputeType_t SupportedComputeType(hipblasComputeType_t compute_type) {
  return compute_type == HIPBLAS_COMPUTE_16F ? HIPBLAS_COMPUTE_32F : compute_type;
}

inline hipblasStatus_t ResetDesc(cublasLtMatmulDesc_t d, hipblasComputeType_t compute_type) {
  if (d->desc != nullptr) {
    const hipblasStatus_t status = hipblasLtMatmulDescDestroy(d->desc);
    d->desc = nullptr;
    if (status != HIPBLAS_STATUS_SUCCESS) { return status; }
  }
  d->compute_type = compute_type;
  const hipblasComputeType_t supported = SupportedComputeType(compute_type);
  return hipblasLtMatmulDescCreate(&d->desc, supported, ScaleType(supported));
}

}  // namespace oneflow_hip_cublaslt

inline hipblasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* desc,
                                                hipblasComputeType_t compute_type,
                                                hipDataType /*scale_type*/) {
  *desc = new cublasLtMatmulDescOpaque{nullptr,
                                       compute_type,
                                       HIPBLAS_OP_N,
                                       HIPBLAS_OP_N,
                                       HIPBLASLT_EPILOGUE_DEFAULT,
                                       nullptr,
                                       nullptr,
                                       0};
  const hipblasStatus_t status = oneflow_hip_cublaslt::ResetDesc(*desc, compute_type);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    delete *desc;
    *desc = nullptr;
  }
  return status;
}

inline hipblasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t desc) {
  if (desc == nullptr) { return HIPBLAS_STATUS_SUCCESS; }
  const hipblasStatus_t status = hipblasLtMatmulDescDestroy(desc->desc);
  delete desc;
  return status;
}

inline hipblasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t desc,
                                                      hipblasLtMatmulDescAttributes_t attr,
                                                      const void* buf, size_t size) {
  switch (static_cast<int>(attr)) {
    case static_cast<int>(CUBLASLT_MATMUL_DESC_COMPUTE_TYPE): {
      if (size != sizeof(hipblasComputeType_t)) { return HIPBLAS_STATUS_INVALID_VALUE; }
      const auto compute_type = *static_cast<const hipblasComputeType_t*>(buf);
      if (compute_type == desc->compute_type) { return HIPBLAS_STATUS_SUCCESS; }
      return oneflow_hip_cublaslt::ResetDesc(desc, compute_type);
    }
    case HIPBLASLT_MATMUL_DESC_TRANSA:
      desc->trans_a = *static_cast<const hipblasOperation_t*>(buf);
      return HIPBLAS_STATUS_SUCCESS;
    case HIPBLASLT_MATMUL_DESC_TRANSB:
      desc->trans_b = *static_cast<const hipblasOperation_t*>(buf);
      return HIPBLAS_STATUS_SUCCESS;
    case HIPBLASLT_MATMUL_DESC_EPILOGUE:
      desc->epilogue = *static_cast<const hipblasLtEpilogue_t*>(buf);
      return HIPBLAS_STATUS_SUCCESS;
    case HIPBLASLT_MATMUL_DESC_BIAS_POINTER:
      desc->bias = *static_cast<const void* const*>(buf);
      return HIPBLAS_STATUS_SUCCESS;
    case HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER:
      desc->aux = *static_cast<const void* const*>(buf);
      return HIPBLAS_STATUS_SUCCESS;
    case HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD:
      desc->aux_ld = size == sizeof(int64_t) ? *static_cast<const int64_t*>(buf)
                                             : *static_cast<const int32_t*>(buf);
      return HIPBLAS_STATUS_SUCCESS;
    default: return hipblasLtMatmulDescSetAttribute(desc->desc, attr, buf, size);
  }
}

inline hipblasStatus_t cublasLtMatmulPreferenceSetAttribute(
    hipblasLtMatmulPreference_t pref, hipblasLtMatmulPreferenceAttributes_t attr, const void* buf,
    size_t size) {
  if (attr == CUBLASLT_MATMUL_PREF_POINTER_MODE_MASK) { return HIPBLAS_STATUS_SUCCESS; }
  return hipblasLtMatmulPreferenceSetAttribute(pref, attr, buf, size);
}

namespace oneflow_hip_cublaslt {

// What runs for one requested epilogue: the hipBLASLt epilogue and the fallback steps after it.
struct Plan {
  hipblasLtEpilogue_t native = HIPBLASLT_EPILOGUE_DEFAULT;
  bool post_bias = false;        // D += bias (bias indexed by row)
  bool post_relu = false;        // D = max(D, 0)
  bool post_write_mask = false;  // aux bit (row, col) = D > 0
  bool post_apply_mask = false;  // D = aux bit ? D : 0
  bool post_bgrad_rows = false;  // bias[row] = sum over columns of D
  bool post_bgrad_a = false;     // bias[row] = sum over k of op(A)
  bool post_bgrad_b = false;     // bias[col] = sum over k of op(B)
};

inline bool Has(hipblasLtEpilogue_t e, hipblasLtEpilogue_t bits) {
  return (static_cast<uint32_t>(e) & static_cast<uint32_t>(bits)) == static_cast<uint32_t>(bits);
}

inline hipblasStatus_t ApplyDesc(cublasLtMatmulDesc_t d, hipblasLtEpilogue_t epilogue) {
  hipblasStatus_t status = hipblasLtMatmulDescSetAttribute(d->desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                           &d->trans_a, sizeof(d->trans_a));
  if (status != HIPBLAS_STATUS_SUCCESS) { return status; }
  status = hipblasLtMatmulDescSetAttribute(d->desc, HIPBLASLT_MATMUL_DESC_TRANSB, &d->trans_b,
                                           sizeof(d->trans_b));
  if (status != HIPBLAS_STATUS_SUCCESS) { return status; }
  status = hipblasLtMatmulDescSetAttribute(d->desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                           sizeof(epilogue));
  if (status != HIPBLAS_STATUS_SUCCESS) { return status; }
  const void* bias = epilogue == HIPBLASLT_EPILOGUE_DEFAULT ? nullptr : d->bias;
  return hipblasLtMatmulDescSetAttribute(d->desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
                                         sizeof(bias));
}

// Whether hipBLASLt has a kernel for the epilogue within the workspace the call will be given.
inline bool NativeAvailable(hipblasLtHandle_t handle, cublasLtMatmulDesc_t d,
                            hipblasLtEpilogue_t epilogue, cublasLtMatrixLayout_t a,
                            cublasLtMatrixLayout_t b, cublasLtMatrixLayout_t c,
                            cublasLtMatrixLayout_t dl, uint64_t workspace) {
  if (ApplyDesc(d, epilogue) != HIPBLAS_STATUS_SUCCESS) { return false; }
  hipblasLtMatmulPreference_t pref;
  if (hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS) { return false; }
  hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
                                        sizeof(workspace));
  hipblasLtMatmulHeuristicResult_t result;
  int count = 0;
  const hipblasStatus_t status = hipblasLtMatmulAlgoGetHeuristic(
      handle, d->desc, a->layout, b->layout, c->layout, dl->layout, pref, 1, &result, &count);
  hipblasLtMatmulPreferenceDestroy(pref);
  return status == HIPBLAS_STATUS_SUCCESS && count > 0;
}

// The heuristic query and the matmul call must reach the same plan, so both pass the workspace
// limit the caller uses for the pair (oneflow gives both the stream's cuBLAS workspace size).
// Without `native`, the fallback kernels do every step they have.
inline Plan MakePlan(hipblasLtHandle_t handle, cublasLtMatmulDesc_t d, cublasLtMatrixLayout_t a,
                     cublasLtMatrixLayout_t b, cublasLtMatrixLayout_t c, cublasLtMatrixLayout_t dl,
                     uint64_t workspace, bool native = true) {
  Plan plan;
  const hipblasLtEpilogue_t e = d->epilogue;
  auto native_or = [&](hipblasLtEpilogue_t wanted) {
    return native && NativeAvailable(handle, d, wanted, a, b, c, dl, workspace);
  };
  switch (e) {
    case HIPBLASLT_EPILOGUE_DEFAULT: break;
    case HIPBLASLT_EPILOGUE_BIAS:
    case HIPBLASLT_EPILOGUE_RELU:
    case HIPBLASLT_EPILOGUE_RELU_BIAS:
    case HIPBLASLT_EPILOGUE_RELU_AUX:
    case HIPBLASLT_EPILOGUE_RELU_AUX_BIAS: {
      const bool bias = Has(e, HIPBLASLT_EPILOGUE_BIAS);
      const hipblasLtEpilogue_t native =
          bias ? HIPBLASLT_EPILOGUE_RELU_BIAS : HIPBLASLT_EPILOGUE_RELU;
      const hipblasLtEpilogue_t wanted =
          Has(e, HIPBLASLT_EPILOGUE_RELU) ? native : HIPBLASLT_EPILOGUE_BIAS;
      if (native_or(wanted)) {
        plan.native = wanted;
      } else {
        plan.post_bias = bias;
        plan.post_relu = Has(e, HIPBLASLT_EPILOGUE_RELU);
      }
      plan.post_write_mask =
          (e == HIPBLASLT_EPILOGUE_RELU_AUX || e == HIPBLASLT_EPILOGUE_RELU_AUX_BIAS);
      break;
    }
    case HIPBLASLT_EPILOGUE_DRELU: plan.post_apply_mask = true; break;
    case HIPBLASLT_EPILOGUE_DRELU_BGRAD:
      plan.post_apply_mask = true;
      plan.post_bgrad_rows = true;
      break;
    case HIPBLASLT_EPILOGUE_BGRADA:
    case HIPBLASLT_EPILOGUE_BGRADB:
      if (native_or(e)) {
        plan.native = e;
      } else {
        plan.post_bgrad_a = (e == HIPBLASLT_EPILOGUE_BGRADA);
        plan.post_bgrad_b = (e == HIPBLASLT_EPILOGUE_BGRADB);
      }
      break;
    default: plan.native = e; break;
  }
  return plan;
}

template<typename T>
struct AccType {
  using type = float;
};
template<>
struct AccType<double> {
  using type = double;
};

template<typename T>
__device__ __forceinline__ typename AccType<T>::type ToAcc(T v) {
  return static_cast<typename AccType<T>::type>(v);
}
template<typename T>
__device__ __forceinline__ T FromAcc(typename AccType<T>::type v) {
  return static_cast<T>(v);
}

// Elementwise fallback over the rows x cols column-major matrix D.
template<typename T>
__global__ void FallbackEpilogueKernel(T* d, int64_t rows, int64_t cols, int64_t ld, const T* bias,
                                       bool relu, const int32_t* apply_mask, int64_t aux_ld) {
  const int64_t n = rows * cols;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < n;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = idx % rows;
    const int64_t col = idx / rows;
    T* p = d + col * ld + row;
    auto v = ToAcc(*p);
    if (bias != nullptr) { v += ToAcc(bias[row]); }
    if (relu && !(v > 0)) { v = 0; }
    if (apply_mask != nullptr) {
      const int64_t bit = col * aux_ld + row;
      if (((apply_mask[bit / 32] >> (bit % 32)) & 1) == 0) { v = 0; }
    }
    *p = FromAcc<T>(v);
  }
}

// One thread per 32-bit word of the bit mask; aux_ld is a multiple of 128, so a word never spans
// two columns. Bits past the last row are written as zero.
template<typename T>
__global__ void FallbackWriteReluMaskKernel(const T* d, int64_t rows, int64_t cols, int64_t ld,
                                            int32_t* mask, int64_t aux_ld) {
  const int64_t words = cols * (aux_ld / 32);
  for (int64_t w = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; w < words;
       w += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t col = (w * 32) / aux_ld;
    const int64_t row0 = (w * 32) % aux_ld;
    uint32_t bits = 0;
    for (int b = 0; b < 32; ++b) {
      const int64_t row = row0 + b;
      if (row < rows && ToAcc(d[col * ld + row]) > 0) { bits |= (1u << b); }
    }
    mask[w] = static_cast<int32_t>(bits);
  }
}

// out[i] = sum_j src[i * stride_i + j * stride_j] for i < n_out, j < n_reduce.
template<typename T>
__global__ void FallbackBiasGradKernel(const T* src, int64_t n_out, int64_t n_reduce,
                                       int64_t stride_i, int64_t stride_j, T* out) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n_out;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    typename AccType<T>::type sum = 0;
    for (int64_t j = 0; j < n_reduce; ++j) { sum += ToAcc(src[i * stride_i + j * stride_j]); }
    out[i] = FromAcc<T>(sum);
  }
}

inline unsigned int NumBlocks(int64_t n, int block) {
  const int64_t blocks = (n + block - 1) / block;
  return static_cast<unsigned int>(std::max<int64_t>(1, std::min<int64_t>(blocks, 65536)));
}

template<typename T>
hipblasStatus_t RunFallback(const Plan& plan, cublasLtMatmulDesc_t desc,
                            const cublasLtMatrixLayoutOpaque& la, const void* a,
                            const cublasLtMatrixLayoutOpaque& lb, const void* b,
                            const cublasLtMatrixLayoutOpaque& ld, void* d, hipStream_t stream) {
  constexpr int kBlock = 256;
  T* dp = static_cast<T*>(d);
  const int64_t rows = static_cast<int64_t>(ld.rows);
  const int64_t cols = static_cast<int64_t>(ld.cols);
  if (plan.post_bias || plan.post_relu || plan.post_apply_mask) {
    hipLaunchKernelGGL(FallbackEpilogueKernel<T>, dim3(NumBlocks(rows * cols, kBlock)),
                       dim3(kBlock), 0, stream, dp, rows, cols, ld.ld,
                       plan.post_bias ? static_cast<const T*>(desc->bias) : nullptr, plan.post_relu,
                       plan.post_apply_mask ? static_cast<const int32_t*>(desc->aux) : nullptr,
                       desc->aux_ld);
  }
  if (plan.post_write_mask) {
    const int64_t words = cols * (desc->aux_ld / 32);
    hipLaunchKernelGGL(FallbackWriteReluMaskKernel<T>, dim3(NumBlocks(words, kBlock)), dim3(kBlock),
                       0, stream, dp, rows, cols, ld.ld,
                       static_cast<int32_t*>(const_cast<void*>(desc->aux)), desc->aux_ld);
  }
  T* bias_out = static_cast<T*>(const_cast<void*>(desc->bias));
  if (plan.post_bgrad_rows) {
    hipLaunchKernelGGL(FallbackBiasGradKernel<T>, dim3(NumBlocks(rows, kBlock)), dim3(kBlock), 0,
                       stream, dp, rows, cols, int64_t(1), ld.ld, bias_out);
  }
  if (plan.post_bgrad_a) {
    // op(A) is rows x k; A is stored column-major as la.rows x la.cols.
    const bool trans = desc->trans_a != HIPBLAS_OP_N;
    const int64_t k = trans ? la.rows : la.cols;
    hipLaunchKernelGGL(FallbackBiasGradKernel<T>, dim3(NumBlocks(rows, kBlock)), dim3(kBlock), 0,
                       stream, static_cast<const T*>(a), rows, k, trans ? la.ld : int64_t(1),
                       trans ? int64_t(1) : la.ld, bias_out);
  }
  if (plan.post_bgrad_b) {
    // op(B) is k x cols; B is stored column-major as lb.rows x lb.cols.
    const bool trans = desc->trans_b != HIPBLAS_OP_N;
    const int64_t k = trans ? lb.cols : lb.rows;
    hipLaunchKernelGGL(FallbackBiasGradKernel<T>, dim3(NumBlocks(cols, kBlock)), dim3(kBlock), 0,
                       stream, static_cast<const T*>(b), cols, k, trans ? int64_t(1) : lb.ld,
                       trans ? lb.ld : int64_t(1), bias_out);
  }
  return hipGetLastError() == hipSuccess ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_EXECUTION_FAILED;
}

inline hipblasStatus_t RunFallback(const Plan& plan, cublasLtMatmulDesc_t desc,
                                   const cublasLtMatrixLayoutOpaque& la, const void* a,
                                   const cublasLtMatrixLayoutOpaque& lb, const void* b,
                                   const cublasLtMatrixLayoutOpaque& ld, void* d,
                                   hipStream_t stream) {
  if (!(plan.post_bias || plan.post_relu || plan.post_write_mask || plan.post_apply_mask
        || plan.post_bgrad_rows || plan.post_bgrad_a || plan.post_bgrad_b)) {
    return HIPBLAS_STATUS_SUCCESS;
  }
  switch (ld.type) {
    case HIP_R_32F: return RunFallback<float>(plan, desc, la, a, lb, b, ld, d, stream);
    case HIP_R_64F: return RunFallback<double>(plan, desc, la, a, lb, b, ld, d, stream);
    case HIP_R_16F: return RunFallback<__half>(plan, desc, la, a, lb, b, ld, d, stream);
    case HIP_R_16BF: return RunFallback<__hip_bfloat16>(plan, desc, la, a, lb, b, ld, d, stream);
    default: return HIPBLAS_STATUS_NOT_SUPPORTED;
  }
}

// D = alpha * op(A) * op(B) + beta * C through OfLongKGemm, then the epilogue in the fallback
// kernels; NOT_SUPPORTED when OfUseLongKGemm does not select the GEMM. The scalars are float.
inline hipblasStatus_t LongKMatmul(hipblasLtHandle_t handle, cublasLtMatmulDesc_t desc,
                                   const float* alpha, const void* a, cublasLtMatrixLayout_t la,
                                   const void* b, cublasLtMatrixLayout_t lb, const float* beta,
                                   const void* c, cublasLtMatrixLayout_t lc, void* d,
                                   cublasLtMatrixLayout_t ld, void* workspace,
                                   size_t workspace_size, hipStream_t stream) {
  const int64_t m = static_cast<int64_t>(ld->rows);
  const int64_t n = static_cast<int64_t>(ld->cols);
  const int64_t k = static_cast<int64_t>(desc->trans_a != HIPBLAS_OP_N ? la->rows : la->cols);
  if (SupportedComputeType(desc->compute_type) != HIPBLAS_COMPUTE_32F || la->type != ld->type
      || lb->type != ld->type || !OfUseLongKGemm(ld->type, k, la->ld, lb->ld)) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
  }
  const Plan plan = MakePlan(handle, desc, la, lb, lc, ld, workspace_size, /*native=*/false);
  if (plan.native != HIPBLASLT_EPILOGUE_DEFAULT) { return HIPBLAS_STATUS_NOT_SUPPORTED; }
  const hipblasHandle_t blas = BlasHandle(handle);
  if (blas == nullptr) { return HIPBLAS_STATUS_NOT_INITIALIZED; }
  hipblasStatus_t status = hipblasSetStream(blas, stream);
  if (status != HIPBLAS_STATUS_SUCCESS) { return status; }
  // OfLongKGemm adds beta * D, so C goes to D first.
  if (*beta != 0.0f && c != d) {
    const size_t size = 2;
    if (hipMemcpy2DAsync(d, ld->ld * size, c, lc->ld * size, m * size, n, hipMemcpyDeviceToDevice,
                         stream)
        != hipSuccess) {
      return HIPBLAS_STATUS_EXECUTION_FAILED;
    }
  }
  status =
      OfLongKGemm(blas, handle, stream, workspace, workspace_size, desc->trans_a, desc->trans_b, m,
                  n, k, *alpha, a, la->ld, 0, b, lb->ld, 0, *beta, d, ld->type, ld->ld, 0, 1);
  if (status != HIPBLAS_STATUS_SUCCESS) { return status; }
  return RunFallback(plan, desc, *la, a, *lb, b, *ld, d, stream);
}

}  // namespace oneflow_hip_cublaslt

inline hipblasStatus_t cublasLtMatmulAlgoGetHeuristic(
    hipblasLtHandle_t handle, cublasLtMatmulDesc_t desc, cublasLtMatrixLayout_t a,
    cublasLtMatrixLayout_t b, cublasLtMatrixLayout_t c, cublasLtMatrixLayout_t d,
    hipblasLtMatmulPreference_t pref, int requested, hipblasLtMatmulHeuristicResult_t* results,
    int* returned) {
  uint64_t workspace = 0;
  size_t written = 0;
  hipblasStatus_t status = hipblasLtMatmulPreferenceGetAttribute(
      pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace), &written);
  if (status != HIPBLAS_STATUS_SUCCESS) { return status; }
  const oneflow_hip_cublaslt::Plan plan =
      oneflow_hip_cublaslt::MakePlan(handle, desc, a, b, c, d, workspace);
  status = oneflow_hip_cublaslt::ApplyDesc(desc, plan.native);
  if (status != HIPBLAS_STATUS_SUCCESS) { return status; }
  return hipblasLtMatmulAlgoGetHeuristic(handle, desc->desc, a->layout, b->layout, c->layout,
                                         d->layout, pref, requested, results, returned);
}

inline hipblasStatus_t cublasLtMatmul(hipblasLtHandle_t handle, cublasLtMatmulDesc_t desc,
                                      const void* alpha, const void* a, cublasLtMatrixLayout_t la,
                                      const void* b, cublasLtMatrixLayout_t lb, const void* beta,
                                      const void* c, cublasLtMatrixLayout_t lc, void* d,
                                      cublasLtMatrixLayout_t ld, const hipblasLtMatmulAlgo_t* algo,
                                      void* workspace, size_t workspace_size, hipStream_t stream) {
  using namespace oneflow_hip_cublaslt;
  float alpha_f = 0.f;
  float beta_f = 0.f;
  if (desc->compute_type == HIPBLAS_COMPUTE_16F) {
    // The descriptor computes in float, so the scalars have to be float as well.
    alpha_f = __half2float(*static_cast<const __half*>(alpha));
    beta_f = __half2float(*static_cast<const __half*>(beta));
    alpha = &alpha_f;
    beta = &beta_f;
  }
  hipblasStatus_t status =
      LongKMatmul(handle, desc, static_cast<const float*>(alpha), a, la, b, lb,
                  static_cast<const float*>(beta), c, lc, d, ld, workspace, workspace_size, stream);
  if (status != HIPBLAS_STATUS_NOT_SUPPORTED) { return status; }
  const Plan plan = MakePlan(handle, desc, la, lb, lc, ld, workspace_size);
  status = ApplyDesc(desc, plan.native);
  if (status != HIPBLAS_STATUS_SUCCESS) { return status; }
  // cuBLASLt treats a null algo as a request for its heuristic's choice. hipBLASLt does not do that
  // on every GPU: on gfx1100 every double GEMM fails with a null algo. Make the query here.
  hipblasLtMatmulHeuristicResult_t heuristic;
  if (algo == nullptr) {
    hipblasLtMatmulPreference_t pref;
    status = hipblasLtMatmulPreferenceCreate(&pref);
    if (status != HIPBLAS_STATUS_SUCCESS) { return status; }
    uint64_t max_workspace = workspace_size;
    status = hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                   &max_workspace, sizeof(max_workspace));
    int count = 0;
    if (status == HIPBLAS_STATUS_SUCCESS) {
      status = hipblasLtMatmulAlgoGetHeuristic(handle, desc->desc, la->layout, lb->layout,
                                               lc->layout, ld->layout, pref, 1, &heuristic, &count);
    }
    hipblasLtMatmulPreferenceDestroy(pref);
    if (status != HIPBLAS_STATUS_SUCCESS) { return status; }
    if (count == 0) { return HIPBLAS_STATUS_NOT_SUPPORTED; }
    algo = &heuristic.algo;
  }
  status = hipblasLtMatmul(handle, desc->desc, alpha, a, la->layout, b, lb->layout, beta, c,
                           lc->layout, d, ld->layout, algo, workspace, workspace_size, stream);
  if (status != HIPBLAS_STATUS_SUCCESS) { return status; }
  return RunFallback(plan, desc, *la, a, *lb, b, *ld, d, stream);
}

#endif  // defined(__HIPCC__)
