#include "oneflow/core/kernel/opkernel_test_case.h"
#include <random>
#include "oneflow/core/device/cuda_device_context.h"

namespace oneflow {

namespace test {

#if defined(WITH_CUDA)

template<>
Blob* CreateBlob<DeviceType::kGPU>(const BlobDesc* blob_desc) {
  void* mem_ptr = nullptr;
  CudaCheck(cudaMalloc(&mem_ptr, blob_desc->TotalByteSize()));
  return NewBlob(nullptr, blob_desc, static_cast<char*>(mem_ptr), nullptr,
                 DeviceType::kGPU);
}

template<>
void BuildKernelCtx<DeviceType::kGPU>(KernelCtx* ctx) {
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
void SyncStream<DeviceType::kGPU>(KernelCtx* ctx) {
  CudaCheck(cudaStreamSynchronize(ctx->device_ctx->cuda_stream()));
}

template<>
template<typename T>
Blob* OpKernelTestCase<DeviceType::kGPU>::CreateBlobWithSpecifiedValPtr(
    const BlobDesc* blob_desc, T* val) const {
  Blob* ret = CreateBlob<DeviceType::kGPU>(blob_desc);
  CudaCheck(cudaMemcpy(ret->mut_dptr(), val, ret->ByteSizeOfDataContentField(),
                       cudaMemcpyHostToDevice));
  return ret;
}

template<>
template<typename T>
void OpKernelTestCase<DeviceType::kGPU>::BlobCmp(const std::string& blob_name,
                                                 const Blob* lhs,
                                                 const Blob* rhs) const {
  Blob* cpu_lhs = CreateBlob<DeviceType::kCPU>(lhs->blob_desc_ptr());
  Blob* cpu_rhs = CreateBlob<DeviceType::kCPU>(rhs->blob_desc_ptr());
  CudaCheck(cudaMemcpy(cpu_lhs->mut_dptr(), lhs->dptr(),
                       lhs->ByteSizeOfDataContentField(),
                       cudaMemcpyDeviceToHost));
  CudaCheck(cudaMemcpy(cpu_rhs->mut_dptr(), rhs->dptr(),
                       rhs->ByteSizeOfDataContentField(),
                       cudaMemcpyDeviceToHost));
  OpKernelTestCase<DeviceType::kCPU>().template BlobCmp<T>(blob_name, cpu_lhs,
                                                           cpu_rhs);
}

#define INSTANTIATE_METHODS(type_cpp, type_proto)                      \
  template Blob*                                                       \
  OpKernelTestCase<DeviceType::kGPU>::CreateBlobWithSpecifiedValPtr(   \
      const BlobDesc* blob_desc, type_cpp* val) const;                 \
  template void OpKernelTestCase<DeviceType::kGPU>::BlobCmp<type_cpp>( \
      const std::string& blob_name, const Blob* lhs, const Blob* rhs) const;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_METHODS, ALL_DATA_TYPE_SEQ);

template<typename T>
class KTCommon<DeviceType::kGPU, T> final {
 public:
  static void CheckInitializeResult(const Blob* blob,
                                    const InitializerConf& initializer_conf) {
    Blob* cpu_blob = CreateBlob<DeviceType::kCPU>(blob->blob_desc_ptr());
    CudaCheck(cudaMemcpy(cpu_blob->mut_dptr(), blob->dptr(),
                         blob->ByteSizeOfDataContentField(),
                         cudaMemcpyDeviceToHost));
    KTCommon<DeviceType::kCPU, T>::CheckInitializeResult(cpu_blob,
                                                         initializer_conf);
  }
};

#define INSTANTIATE_KTCOMMON(type_cpp, type_proto) \
  template class KTCommon<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KTCOMMON, ALL_DATA_TYPE_SEQ)

#endif

}  // namespace test

}  // namespace oneflow
