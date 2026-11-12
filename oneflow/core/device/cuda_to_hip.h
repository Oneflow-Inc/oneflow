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
#ifndef ONEFLOW_CORE_DEVICE_CUDA_TO_HIP_H_
#define ONEFLOW_CORE_DEVICE_CUDA_TO_HIP_H_

// Single CUDA-to-HIP compatibility header. It is force-included
// on every HIP translation unit (CMAKE_HIP_FLAGS -include ...), so the aliases and
// macro shims below precede each file's own includes regardless of include order.
// On a real CUDA build this file is never compiled (it is gated on USE_HIP) and the
// NVIDIA path is unchanged.
//
// oneflow's whole GPU backend is gated on WITH_CUDA; we define WITH_CUDA FOR the HIP
// build (see cmake/cuda.cmake) so the ~200 shared host/device blocks compile unchanged
// and DeviceType::kCUDA keeps meaning "the GPU backend". This header retargets the
// cuda* runtime/library symbols to their hip* equivalents.

#if defined(USE_HIP)

// This header is force-included on BOTH oneflow's .cu TUs (compiled -x hip, which
// auto-defines __HIP_PLATFORM_AMD__) and the host .cpp TUs (compiled as plain C++ by the
// same ROCm clang, which does NOT). <hip/hip_runtime.h> hard-#errors unless exactly one
// HIP platform macro is set, so on the plain-C++ pass we must select the AMD platform
// ourselves before including any HIP header.
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__
#endif

// libc host declarations must be seen before HIP's __device__ memcpy/memset overloads,
// so include them before <hip/hip_runtime.h>.
#include <cstdio>
#include <cstring>
#include <cstdlib>

#include <hip/hip_runtime.h>

// The HIP device-math headers (hip_fp16/hip_bf16/hip_complex/hip_math_constants) and the
// device-side ROCm library headers below are implemented with clang builtins
// (__builtin_elementwise_*, etc.) and only parse under clang, never a g++ CXX compile.
// The whole monolith is built with the ROCm clang (build_rocm.sh), so __clang__ holds on
// every TU and they are pulled in everywhere; the gate stays as a guard so that a future
// g++ host build (or any non-clang CXX) does not hard-error on these clang-only headers.
#if defined(__clang__)
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_complex.h>
#include <hip/hip_math_constants.h>
#endif

// __CUDACC__ is what header-only CUDA libs key device definitions on; hipcc defines
// __HIPCC__ (both passes) but not __CUDACC__. Define it so those paths are taken.
#if defined(__HIPCC__) && !defined(__CUDACC__)
#define __CUDACC__ 1
#endif

// __CUDA_ARCH__ is undefined during HIP device compilation (HIP uses
// __HIP_DEVICE_COMPILE__). Many device-vs-host #ifdef __CUDA_ARCH__ selections would
// otherwise silently take the host branch in the device pass. Define __CUDA_ARCH__ in
// the HIP device pass only (1 in device pass, absent in host pass).
// Numeric `#if __CUDA_ARCH__ >= NNN` intrinsic gates are separately guarded
// with !defined(USE_HIP) at their sites so the portable fallback runs.
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ && !defined(__CUDA_ARCH__)
#define __CUDA_ARCH__ 1
#endif

// Eigen has a native HIP device path keyed on __HIP_DEVICE_COMPILE__. Defining
// __CUDA_ARCH__ above would otherwise steer Eigen into its CUDA path and pull in the
// absent <math_constants.h>; EIGEN_NO_CUDA makes Eigen fall through to its HIP path.
#ifndef EIGEN_NO_CUDA
#define EIGEN_NO_CUDA
#endif

// Full-warp lane mask. ROCm 7.x __shfl_*_sync static_assert sizeof(mask)==8 regardless
// of wave width, so the CUDA literal 0xffffffff fails to compile.
// Keyed on USE_HIP, not on wave width.
#define OF_SHFL_FULL_MASK 0xffffffffffffffffULL

// CUDA_VERSION is a preprocessor macro from <cuda.h> on NVIDIA; it is absent on HIP.
// Several kernels read it both as an `#if` gate and as a runtime value. Define it to a
// recent CUDA release so the version gates take the modern (cuBLAS v11+) code paths.
#ifndef CUDA_VERSION
#define CUDA_VERSION 11080
#endif
// CUDART_VERSION is the CUDA-runtime counterpart oneflow's runtime-version check keys on.
#ifndef CUDART_VERSION
#define CUDART_VERSION CUDA_VERSION
#endif

#include <hip/library_types.h>

// ---- CUDA runtime --------------------------------------------------------------
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceReset hipDeviceReset
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaSetDeviceFlags hipSetDeviceFlags
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define cudaRuntimeGetVersion hipRuntimeGetVersion

#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#define cudaDevAttrMaxSharedMemoryPerBlock hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaDevAttrMaxThreadsPerBlock hipDeviceAttributeMaxThreadsPerBlock
#define cudaDevAttrMaxThreadsPerMultiProcessor hipDeviceAttributeMaxThreadsPerMultiProcessor
#define cudaDevAttrMemoryPoolsSupported hipDeviceAttributeMemoryPoolsSupported
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount

#define cudaErrorCudartUnloading hipErrorDeinitialized
#define cudaErrorInsufficientDriver hipErrorInsufficientDriver
#define cudaErrorInvalidDevice hipErrorInvalidDevice
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaErrorMemoryAllocation hipErrorMemoryAllocation
#define cudaErrorNoDevice hipErrorNoDevice
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorNotSupported hipErrorNotSupported
#define cudaErrorUnknown hipErrorUnknown

#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventBlockingSync hipEventBlockingSync
#define cudaEventDisableTiming hipEventDisableTiming

#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamGetFlags hipStreamGetFlags
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamCaptureMode hipStreamCaptureMode
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaThreadExchangeStreamCaptureMode hipThreadExchangeStreamCaptureMode

#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost hipHostMalloc
#define cudaHostAlloc hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaMallocManaged hipMallocManaged
#define cudaFreeAsync hipFreeAsync
#define cudaMallocFromPoolAsync hipMallocFromPoolAsync
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemset2DAsync hipMemset2DAsync
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost

#define cudaMemPool_t hipMemPool_t
#define cudaMemPoolCreate hipMemPoolCreate
#define cudaMemPoolDestroy hipMemPoolDestroy
#define cudaMemPoolProps hipMemPoolProps
#define cudaMemPoolSetAttribute hipMemPoolSetAttribute
#define cudaMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold
#define cudaMemPoolReuseAllowInternalDependencies hipMemPoolReuseAllowInternalDependencies
#define cudaMemPoolReuseAllowOpportunistic hipMemPoolReuseAllowOpportunistic
#define cudaMemPoolReuseFollowEventDependencies hipMemPoolReuseFollowEventDependencies
#define cudaMemAllocationTypePinned hipMemAllocationTypePinned
#define cudaMemHandleTypePosixFileDescriptor hipMemHandleTypePosixFileDescriptor
#define cudaMemLocationTypeDevice hipMemLocationTypeDevice

#define cudaIpcMemHandle_t hipIpcMemHandle_t
#define cudaIpcGetMemHandle hipIpcGetMemHandle
#define cudaIpcOpenMemHandle hipIpcOpenMemHandle
#define cudaIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess
#define CUdeviceptr hipDeviceptr_t
#define CU_POINTER_ATTRIBUTE_RANGE_START_ADDR HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR

#define cudaFuncAttributes hipFuncAttributes
#define cudaFuncGetAttributes hipFuncGetAttributes
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaLaunchKernel hipLaunchKernel
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor OfHipOccupancyMaxActiveBlocksPerMultiprocessor

// The shared-memory softmax and normalization kernels are chosen whenever the occupancy query
// reports at least one resident block, which on CUDA means the block fits. On RDNA the HIP query
// measures dynamic shared memory against the whole workgroup processor (128 KiB on gfx11) rather
// than the per-block limit (64 KiB), so it reports a block that then fails to launch or aborts
// the queue. Report zero blocks when dynamic plus static shared memory exceeds the device's
// per-block limit, as CUDA does.
template<typename F>
inline hipError_t OfHipOccupancyMaxActiveBlocksPerMultiprocessor(int* num_blocks, F func,
                                                                 int block_size,
                                                                 size_t dynamic_smem_size) {
  const void* func_ptr = reinterpret_cast<const void*>(func);
  hipError_t err = hipOccupancyMaxActiveBlocksPerMultiprocessor(num_blocks, func_ptr, block_size,
                                                                dynamic_smem_size);
  if (err != hipSuccess || *num_blocks <= 0 || dynamic_smem_size == 0) { return err; }
  int dev = 0;
  err = hipGetDevice(&dev);
  if (err != hipSuccess) { return err; }
  int max_smem_per_block = 0;
  err = hipDeviceGetAttribute(&max_smem_per_block, hipDeviceAttributeMaxSharedMemoryPerBlock, dev);
  if (err != hipSuccess) { return err; }
  hipFuncAttributes attr{};
  err = hipFuncGetAttributes(&attr, func_ptr);
  if (err != hipSuccess) { return err; }
  if (dynamic_smem_size + attr.sharedSizeBytes > static_cast<size_t>(max_smem_per_block)) {
    *num_blocks = 0;
  }
  return hipSuccess;
}

#define cudaGraph_t hipGraph_t
#define cudaGraphExec_t hipGraphExec_t
#define cudaGraphNode_t hipGraphNode_t
#define cudaGraphInstantiate hipGraphInstantiate
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphExecUpdate hipGraphExecUpdate
#define cudaGraphExecUpdateResult hipGraphExecUpdateResult
#define cudaGraphExecUpdateSuccess hipGraphExecUpdateSuccess

#define cudaProfilerStart hipProfilerStart
#define cudaProfilerStop hipProfilerStop

#define cudaDataType hipDataType
#define cudaDataType_t hipDataType
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_16BF HIP_R_16BF
#define CUDA_R_32F HIP_R_32F
#define CUDA_R_64F HIP_R_64F
#define CUDA_C_32F HIP_C_32F
#define CUDA_C_64F HIP_C_64F

// bf16 spelling -> HIP's modern bf16 type. In CUDA nv_bfloat16 is a typedef of
// __nv_bfloat16; both map to __hip_bfloat16 (the type __float2bfloat16 etc. return), NOT
// the legacy hip_bfloat16 struct, so functor return/param types match the conversions.
#define __nv_bfloat16 __hip_bfloat16
#define __nv_bfloat162 __hip_bfloat162
#define nv_bfloat16 __hip_bfloat16
#define nv_bfloat162 __hip_bfloat162
// CUDA's __float2bfloat162_rn broadcasts a scalar float to bf16x2; HIP has no such name
// (its __float22bfloat162_rn takes a float2). Provide the scalar-broadcast helper.
// (__float2half2_rn, __half22float2, __bfloat1622float2 are HIP-native.) This is a
// device intrinsic over the bf16 type; only emit it on a HIP-language TU (-x hip, where
// __device__ is a valid keyword). A plain C++ clang compile of a host .cpp defines
// __clang__ but not __HIP__, so __device__ would be rejected there -- and host code never
// calls this.
#if defined(__HIP__)
__device__ __forceinline__ __hip_bfloat162 __float2bfloat162_rn(float v) {
  __hip_bfloat16 h = __float2bfloat16(v);
  return __hip_bfloat162{h, h};
}
#endif

// Complex types (hip_complex.h).
#define cuComplex hipComplex
#define cuFloatComplex hipFloatComplex
#define cuDoubleComplex hipDoubleComplex
#define make_cuComplex make_hipComplex
#define make_cuFloatComplex make_hipFloatComplex
#define make_cuDoubleComplex make_hipDoubleComplex
#define cuCmulf hipCmulf
#define cuCmul hipCmul
#define cuCdivf hipCdivf
#define cuCdiv hipCdiv
#define cuCaddf hipCaddf
#define cuCadd hipCadd
#define cuCrealf hipCrealf
#define cuCimagf hipCimagf
#define cuCreal hipCreal
#define cuCimag hipCimag
#define cuComplexFloatToDouble hipComplexFloatToDouble
#define cuComplexDoubleToFloat hipComplexDoubleToFloat

// math_constants.h infinity/NaN constants (hip_math_constants.h).
#define CUDART_INF_F HIP_INF_F
#define CUDART_NAN_F HIP_NAN_F
#define CUDART_INF HIP_INF
#define CUDART_NAN HIP_NAN

// __trap() (device abort) has no HIP spelling; abort() works in HIP device code.
#define __trap abort

// ---- cuBLAS --------------------------------------------------------------------
#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define cublasMath_t hipblasMath_t
#define CUBLAS_DEFAULT_MATH HIPBLAS_DEFAULT_MATH
#define cublasOperation_t hipblasOperation_t
#define cublasComputeType_t hipblasComputeType_t
#define cublasGemmAlgo_t hipblasGemmAlgo_t
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasSetWorkspace hipblasSetWorkspace
#define cublasSetMathMode hipblasSetMathMode
#define cublasGetMathMode hipblasGetMathMode
#define cublasGetVersion hipblasGetVersion
#define cublasGemmEx hipblasGemmEx
#define cublasGemmBatchedEx hipblasGemmBatchedEx
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#define cublasSgetrfBatched hipblasSgetrfBatched
#define cublasDgetrfBatched hipblasDgetrfBatched
#define cublasSgetriBatched hipblasSgetriBatched
#define cublasDgetriBatched hipblasDgetriBatched
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_OP_C HIPBLAS_OP_C
#define CUBLAS_TENSOR_OP_MATH HIPBLAS_TENSOR_OP_MATH
#define CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION \
  HIPBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DFALT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_64F HIPBLAS_COMPUTE_64F
// hipblas's C enum has no TF32 / pedantic compute type (TF32 is an NVIDIA Ampere feature,
// and pedantic just disables tensor-core auto-use). Map both to plain fp32 compute so AMD
// runs full-precision fp32 GEMM -- the numerically correct, arch-unified choice.
#define CUBLAS_COMPUTE_32F_FAST_TF32 HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F_PEDANTIC HIPBLAS_COMPUTE_32F
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED

// ---- cuRAND --------------------------------------------------------------------
#define curand hiprand
#define curand4 hiprand4
#define curandGenerator_t hiprandGenerator_t
#define curandStatus_t hiprandStatus_t
#define curandState hiprandState_t
#define curandStatePhilox4_32_10_t hiprandStatePhilox4_32_10_t
#define curandCreateGenerator hiprandCreateGenerator
#define curandDestroyGenerator hiprandDestroyGenerator
#define curandSetStream hiprandSetStream
#define curandSetPseudoRandomGeneratorSeed hiprandSetPseudoRandomGeneratorSeed
#define curandGenerateUniform hiprandGenerateUniform
#define curandGenerateUniformDouble hiprandGenerateUniformDouble
#define curand_init hiprand_init
#define curand_uniform hiprand_uniform
#define curand_uniform4 hiprand_uniform4
#define curand_uniform2_double hiprand_uniform2_double
#define curand_normal hiprand_normal
#define curand_normal4 hiprand_normal4
#define curand_normal2_double hiprand_normal2_double
#define CURAND_RNG_PSEUDO_DEFAULT HIPRAND_RNG_PSEUDO_DEFAULT
#define CURAND_STATUS_SUCCESS HIPRAND_STATUS_SUCCESS
#define CURAND_STATUS_ALLOCATION_FAILED HIPRAND_STATUS_ALLOCATION_FAILED
#define CURAND_STATUS_ARCH_MISMATCH HIPRAND_STATUS_ARCH_MISMATCH
#define CURAND_STATUS_DOUBLE_PRECISION_REQUIRED HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED
#define CURAND_STATUS_INITIALIZATION_FAILED HIPRAND_STATUS_INITIALIZATION_FAILED
#define CURAND_STATUS_INTERNAL_ERROR HIPRAND_STATUS_INTERNAL_ERROR
#define CURAND_STATUS_LAUNCH_FAILURE HIPRAND_STATUS_LAUNCH_FAILURE
#define CURAND_STATUS_LENGTH_NOT_MULTIPLE HIPRAND_STATUS_LENGTH_NOT_MULTIPLE
#define CURAND_STATUS_NOT_INITIALIZED HIPRAND_STATUS_NOT_INITIALIZED
#define CURAND_STATUS_OUT_OF_RANGE HIPRAND_STATUS_OUT_OF_RANGE
#define CURAND_STATUS_PREEXISTING_FAILURE HIPRAND_STATUS_PREEXISTING_FAILURE
#define CURAND_STATUS_TYPE_ERROR HIPRAND_STATUS_TYPE_ERROR
#define CURAND_STATUS_VERSION_MISMATCH HIPRAND_STATUS_VERSION_MISMATCH

// ---- cuFFT ---------------------------------------------------------------------
#define cufftHandle hipfftHandle
#define cufftResult_t hipfftResult_t
#define cufftComplex hipfftComplex
#define cufftDoubleComplex hipfftDoubleComplex
#define cufftCreate hipfftCreate
#define cufftDestroy hipfftDestroy
#define cufftSetStream hipfftSetStream
#define cufftSetAutoAllocation hipfftSetAutoAllocation
#define cufftSetWorkArea hipfftSetWorkArea
#define cufftXtExec hipfftXtExec
#define CUFFT_FORWARD HIPFFT_FORWARD
#define CUFFT_INVERSE HIPFFT_BACKWARD
#define CUFFT_SUCCESS HIPFFT_SUCCESS
#define CUFFT_ALLOC_FAILED HIPFFT_ALLOC_FAILED
#define CUFFT_INVALID_PLAN HIPFFT_INVALID_PLAN
#define CUFFT_INVALID_VALUE HIPFFT_INVALID_VALUE
#define CUFFT_INVALID_TYPE HIPFFT_INVALID_TYPE
#define CUFFT_INVALID_SIZE HIPFFT_INVALID_SIZE
#define CUFFT_INVALID_DEVICE HIPFFT_INVALID_DEVICE
#define CUFFT_INTERNAL_ERROR HIPFFT_INTERNAL_ERROR
#define CUFFT_EXEC_FAILED HIPFFT_EXEC_FAILED
#define CUFFT_SETUP_FAILED HIPFFT_SETUP_FAILED
#define CUFFT_NO_WORKSPACE HIPFFT_NO_WORKSPACE
#define CUFFT_NOT_IMPLEMENTED HIPFFT_NOT_IMPLEMENTED
#define CUFFT_NOT_SUPPORTED HIPFFT_NOT_SUPPORTED
#define CUFFT_UNALIGNED_DATA HIPFFT_UNALIGNED_DATA
#define CUFFT_INCOMPLETE_PARAMETER_LIST HIPFFT_INCOMPLETE_PARAMETER_LIST
#define CUFFT_PARSE_ERROR HIPFFT_PARSE_ERROR

// cuDNN's batchnorm minimum-epsilon floor, used by a few normalization kernels purely as a
// numeric CHECK bound (CHECK_GE(epsilon, CUDNN_BN_MIN_EPSILON)), independent of any cuDNN
// call. cuDNN lowered this from 1e-5 to 0.0 in 7.5; oneflow targets CUDA 11.x (cuDNN 8+),
// so match the modern value -- 1e-5 would wrongly reject the common eps=1e-6 layer norm.
#ifndef CUDNN_BN_MIN_EPSILON
#define CUDNN_BN_MIN_EPSILON 0.0
#endif

// ---- cuSOLVER (hipSOLVER's Dn compatibility API has the same signatures) ----------
#define cusolverStatus_t hipsolverStatus_t
#define cusolverDnCreate hipsolverDnCreate
#define cusolverDnDestroy hipsolverDnDestroy
#define cusolverDnSetStream hipsolverDnSetStream
#define cusolverDnSgetrf_bufferSize hipsolverDnSgetrf_bufferSize
#define cusolverDnDgetrf_bufferSize hipsolverDnDgetrf_bufferSize
#define cusolverDnSgetrf hipsolverDnSgetrf
#define cusolverDnDgetrf hipsolverDnDgetrf
#define CUSOLVER_STATUS_SUCCESS HIPSOLVER_STATUS_SUCCESS
#define CUSOLVER_STATUS_NOT_INITIALIZED HIPSOLVER_STATUS_NOT_INITIALIZED
#define CUSOLVER_STATUS_ALLOC_FAILED HIPSOLVER_STATUS_ALLOC_FAILED
#define CUSOLVER_STATUS_INVALID_VALUE HIPSOLVER_STATUS_INVALID_VALUE
#define CUSOLVER_STATUS_ARCH_MISMATCH HIPSOLVER_STATUS_ARCH_MISMATCH
#define CUSOLVER_STATUS_EXECUTION_FAILED HIPSOLVER_STATUS_EXECUTION_FAILED
#define CUSOLVER_STATUS_INTERNAL_ERROR HIPSOLVER_STATUS_INTERNAL_ERROR
#define CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED

#endif  // USE_HIP

#endif  // ONEFLOW_CORE_DEVICE_CUDA_TO_HIP_H_
