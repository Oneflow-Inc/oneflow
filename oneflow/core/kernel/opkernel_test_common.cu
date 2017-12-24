#include <random>
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/opkernel_test_common.h"

namespace oneflow {

namespace test {

template<>
void* MallocAndClean<DeviceType::kGPU>(size_t sz) {
  void* mem_ptr = nullptr;
  CudaCheck(cudaMalloc(&mem_ptr, sz));
  CudaCheck(cudaMemset(mem_ptr, 0, sz));
  return mem_ptr;
}

template<>
Blob* CreateBlob<DeviceType::kGPU>(const BlobDesc* blob_desc) {
  void* mem_ptr = MallocAndClean<DeviceType::kGPU>(blob_desc->TotalByteSize());
  return new Blob(blob_desc, static_cast<char*>(mem_ptr));
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

template<>
void CopyFromHost<DeviceType::kGPU>(void* dst, const void* src, size_t sz) {
  CudaCheck(cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice));
}

template<typename T>
class KTCommon<DeviceType::kGPU, T> final {
 public:
  static void BlobCmp(const Blob* lhs, const Blob* rhs) {
    Blob* cpu_lhs = CreateBlob<DeviceType::kCPU>(lhs->blob_desc_ptr());
    Blob* cpu_rhs = CreateBlob<DeviceType::kCPU>(rhs->blob_desc_ptr());
    CudaCheck(cudaMemcpy(cpu_lhs->mut_memory_ptr(), lhs->memory_ptr(),
                         lhs->TotalByteSize(), cudaMemcpyDeviceToHost));
    CudaCheck(cudaMemcpy(cpu_rhs->mut_memory_ptr(), rhs->memory_ptr(),
                         rhs->TotalByteSize(), cudaMemcpyDeviceToHost));
    KTCommon<DeviceType::kCPU, T>::BlobCmp(cpu_lhs, cpu_rhs);
  }

  static void CheckFillResult(const Blob* blob, const FillConf& fill_conf) {
    Blob* cpu_blob = CreateBlob<DeviceType::kCPU>(blob->blob_desc_ptr());
    CudaCheck(cudaMemcpy(cpu_blob->mut_dptr(), blob->dptr(),
                         blob->ByteSizeOfDataContentField(),
                         cudaMemcpyDeviceToHost));
    KTCommon<DeviceType::kCPU, T>::CheckFillResult(cpu_blob, fill_conf);
  }
};

#define INSTANTIATE_KTCOMMON(type_cpp, type_proto) \
  template class KTCommon<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KTCOMMON, ALL_DATA_TYPE_SEQ)

}  // namespace test

}  // namespace oneflow
