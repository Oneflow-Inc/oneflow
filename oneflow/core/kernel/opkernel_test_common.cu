#include <random>
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/opkernel_test_common.h"

namespace oneflow {

namespace test {

template<>
Blob* CreateBlob<DeviceType::kGPU>(const BlobDesc* blob_desc) {
  void* mem_ptr = nullptr;
  CudaCheck(cudaMalloc(&mem_ptr, blob_desc->TotalByteSize()));
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

template<typename T>
class KTCommon<DeviceType::kGPU, T> final {
 public:
  static Blob* CreateBlobWithSpecifiedVal(const BlobDesc* blob_desc, T* val,
                                          std::vector<std::string>* data_id) {
    Blob* ret = CreateBlob<DeviceType::kGPU>(blob_desc);
    CudaCheck(cudaMemcpy(ret->mut_dptr(), val,
                         ret->ByteSizeOfDataContentField(),
                         cudaMemcpyHostToDevice));
    if (ret->has_data_id() && data_id != nullptr) {
      CHECK_EQ(data_id->size(), ret->shape().At(0));
      FOR_RANGE(size_t, i, 0, data_id->size()) {
        std::string data_id_str = data_id->at(i);
        CudaCheck(cudaMemcpy(ret->mut_data_id(i), data_id_str.c_str(),
                             JobDesc::Singleton()->SizeOfOneDataId(),
                             cudaMemcpyHostToDevice));
        CHECK_LE(data_id_str.size(), JobDesc::Singleton()->SizeOfOneDataId());
        if (data_id_str.size() < JobDesc::Singleton()->SizeOfOneDataId()) {
          CudaCheck(cudaMemcpy(ret->mut_data_id(i) + data_id_str.size(), "\0",
                               1, cudaMemcpyHostToDevice));
        }
      }
    }
    return ret;
  }

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
