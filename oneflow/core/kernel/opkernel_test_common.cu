#include <random>
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/opkernel_test_common.h"

namespace oneflow {

namespace test {

template<>
Blob* CreateBlob<DeviceType::kGPU>(const BlobDesc* blob_desc) {
  void* mem_ptr = nullptr;
  CudaCheck(cudaMalloc(&mem_ptr, blob_desc->TotalByteSize()));
  return new Blob(nullptr, blob_desc, static_cast<char*>(mem_ptr));
}

template<>
void BuildKernelCtx<DeviceType::kGPU>(KernelCtx* ctx) {
  cudaStream_t* cuda_stream = new cudaStream_t;
  cublasHandle_t* cublas_handle = new cublasHandle_t;
  CudaCheck(cudaStreamCreate(cuda_stream));
  CudaCheck(cublasCreate(cublas_handle));
  CudaCheck(cublasSetStream(*cublas_handle, *cuda_stream));
  ctx->device_ctx = new CudaDeviceCtx(-1, cuda_stream, cublas_handle, nullptr);
}

template<>
void SyncStream<DeviceType::kGPU>(KernelCtx* ctx) {
  CudaCheck(cudaStreamSynchronize(ctx->device_ctx->cuda_stream()));
}

template<typename T>
class KTCommon<DeviceType::kGPU, T> final {
 public:
  static Blob* CreateBlobWithSpecifiedVal(const BlobDesc* blob_desc, T* val) {
    Blob* ret = CreateBlob<DeviceType::kGPU>(blob_desc);
    CudaCheck(cudaMemcpy(ret->mut_dptr(), val,
                         ret->ByteSizeOfDataContentField(),
                         cudaMemcpyHostToDevice));
    return ret;
  }

  static void BlobCmp(const Blob* lhs, const Blob* rhs) {
    Blob* cpu_lhs = CreateBlob<DeviceType::kCPU>(lhs->blob_desc_ptr());
    Blob* cpu_rhs = CreateBlob<DeviceType::kCPU>(rhs->blob_desc_ptr());
    CudaCheck(cudaMemcpy(cpu_lhs->mut_dptr(), lhs->dptr(),
                         lhs->ByteSizeOfDataContentField(),
                         cudaMemcpyDeviceToHost));
    CudaCheck(cudaMemcpy(cpu_rhs->mut_dptr(), rhs->dptr(),
                         rhs->ByteSizeOfDataContentField(),
                         cudaMemcpyDeviceToHost));
    KTCommon<DeviceType::kCPU, T>::BlobCmp(cpu_lhs, cpu_rhs);
  }

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

}  // namespace test

}  // namespace oneflow
