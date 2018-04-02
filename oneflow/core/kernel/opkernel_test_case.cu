#include "oneflow/core/kernel/opkernel_test_case.h"
#include <random>
#include "oneflow/core/device/cuda_device_context.h"

namespace oneflow {

namespace test {

#if defined(WITH_CUDA)

template<>
Blob* OpKernelTestCase<DeviceType::kGPU>::CreateBlob(const BlobDesc* blob_desc,
                                                     Regst* regst) {
  void* mem_ptr = nullptr;
  CudaCheck(cudaMalloc(&mem_ptr, blob_desc->TotalByteSize()));
  return NewBlob(regst, blob_desc, static_cast<char*>(mem_ptr), nullptr,
                 DeviceType::kGPU);
}

template<>
void OpKernelTestCase<DeviceType::kGPU>::BuildKernelCtx(KernelCtx* ctx) {
  cudaStream_t* cuda_stream = new cudaStream_t;
  cublasHandle_t* cublas_pmh_handle = new cublasHandle_t;
  cublasHandle_t* cublas_pmd_handle = new cublasHandle_t;
  cudnnHandle_t* cudnn_handle = new cudnnHandle_t;
  CudaCheck(cudaStreamCreate(cuda_stream));
  CudaCheck(cublasCreate(cublas_pmh_handle));
  CudaCheck(cublasCreate(cublas_pmd_handle));
  CudaCheck(cublasSetStream(*cublas_pmh_handle, *cuda_stream));
  CudaCheck(cublasSetStream(*cublas_pmd_handle, *cuda_stream));
  CudaCheck(
      cublasSetPointerMode(*cublas_pmd_handle, CUBLAS_POINTER_MODE_DEVICE));
  CudaCheck(cudnnCreate(cudnn_handle));
  CudaCheck(cudnnSetStream(*cudnn_handle, *cuda_stream));
  ctx->device_ctx = new CudaDeviceCtx(-1, cuda_stream, cublas_pmh_handle,
                                      cublas_pmd_handle, cudnn_handle, nullptr);
}

template<>
void OpKernelTestCase<DeviceType::kGPU>::SyncStream(KernelCtx* ctx) {
  CudaCheck(cudaStreamSynchronize(ctx->device_ctx->cuda_stream()));
}

template<>
template<typename T>
Blob* OpKernelTestCase<DeviceType::kGPU>::CreateBlobWithSpecifiedValPtr(
    const BlobDesc* blob_desc, T* val, Regst* regst) {
  Blob* ret = CreateBlob(blob_desc, regst);
  CudaCheck(cudaMemcpy(ret->mut_dptr(), val, ret->ByteSizeOfDataContentField(),
                       cudaMemcpyHostToDevice));
  return ret;
}

template<>
template<typename T>
void OpKernelTestCase<DeviceType::kGPU>::BlobCmp(const std::string& blob_name,
                                                 const Blob* lhs,
                                                 const Blob* rhs) {
  Blob* cpu_lhs = OpKernelTestCase<DeviceType::kCPU>::CreateBlob(
      lhs->blob_desc_ptr(), nullptr);
  Blob* cpu_rhs = OpKernelTestCase<DeviceType::kCPU>::CreateBlob(
      rhs->blob_desc_ptr(), nullptr);
  CudaCheck(cudaMemcpy(cpu_lhs->mut_dptr(), lhs->dptr(),
                       lhs->ByteSizeOfDataContentField(),
                       cudaMemcpyDeviceToHost));
  CudaCheck(cudaMemcpy(cpu_rhs->mut_dptr(), rhs->dptr(),
                       rhs->ByteSizeOfDataContentField(),
                       cudaMemcpyDeviceToHost));
  OpKernelTestCase<DeviceType::kCPU>::template BlobCmp<T>(blob_name, cpu_lhs,
                                                          cpu_rhs);
}

template<>
template<typename T>
void OpKernelTestCase<DeviceType::kGPU>::CheckInitializeResult(
    const Blob* blob, const InitializerConf& initializer_conf) {
  Blob* cpu_blob = OpKernelTestCase<DeviceType::kCPU>::CreateBlob(
      blob->blob_desc_ptr(), nullptr);
  CudaCheck(cudaMemcpy(cpu_blob->mut_dptr(), blob->dptr(),
                       blob->ByteSizeOfDataContentField(),
                       cudaMemcpyDeviceToHost));
  OpKernelTestCase<DeviceType::kCPU>::template CheckInitializeResult<T>(
      cpu_blob, initializer_conf);
}

#define INSTANTIATE_METHODS(type_cpp, type_proto)                      \
  template Blob*                                                       \
  OpKernelTestCase<DeviceType::kGPU>::CreateBlobWithSpecifiedValPtr(   \
      const BlobDesc* blob_desc, type_cpp* val, Regst* regst);         \
  template void OpKernelTestCase<DeviceType::kGPU>::BlobCmp<type_cpp>( \
      const std::string& blob_name, const Blob* lhs, const Blob* rhs); \
  template void                                                        \
  OpKernelTestCase<DeviceType::kGPU>::CheckInitializeResult<type_cpp>( \
      const Blob* blob, const InitializerConf& initializer_conf);
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_METHODS, ALL_DATA_TYPE_SEQ);

#endif

}  // namespace test

}  // namespace oneflow
