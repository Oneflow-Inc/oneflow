#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

#if defined(WITH_CUDA)

template<>
Blob* OpKernelTestUtil<DeviceType::kGPU>::CreateBlob(const BlobDesc* blob_desc,
                                                     Regst* regst) {
  void* mem_ptr = nullptr;
  CudaCheck(cudaMalloc(&mem_ptr, blob_desc->TotalByteSize()));
  return NewBlob(regst, blob_desc, static_cast<char*>(mem_ptr), nullptr,
                 DeviceType::kGPU);
}

template<>
void OpKernelTestUtil<DeviceType::kGPU>::SyncStream(KernelCtx* ctx) {
  CudaCheck(cudaStreamSynchronize(ctx->device_ctx->cuda_stream()));
}

template<>
template<typename T>
Blob* OpKernelTestUtil<DeviceType::kGPU>::CreateBlobWithSpecifiedValPtr(
    const BlobDesc* blob_desc, T* val, Regst* regst) {
  Blob* ret = CreateBlob(blob_desc, regst);
  CudaCheck(cudaMemcpy(ret->mut_dptr(), val, ret->ByteSizeOfDataContentField(),
                       cudaMemcpyHostToDevice));
  return ret;
}

template<>
template<typename T>
void OpKernelTestUtil<DeviceType::kGPU>::BlobCmp(const std::string& blob_name,
                                                 const Blob* lhs,
                                                 DeviceType lhs_device_type,
                                                 const Blob* rhs,
                                                 DeviceType rhs_device_type) {
  const Blob* cpu_lhs = lhs;
  if (lhs_device_type == DeviceType::kGPU) {
    Blob* mut_cpu_lhs = OpKernelTestUtil<DeviceType::kCPU>::CreateBlob(
        lhs->blob_desc_ptr(), nullptr);
    CudaCheck(cudaMemcpy(mut_cpu_lhs->mut_dptr(), lhs->dptr(),
                         lhs->ByteSizeOfDataContentField(),
                         cudaMemcpyDeviceToHost));
    cpu_lhs = mut_cpu_lhs;
  }
  const Blob* cpu_rhs = rhs;
  if (rhs_device_type == DeviceType::kGPU) {
    Blob* mut_cpu_rhs = OpKernelTestUtil<DeviceType::kCPU>::CreateBlob(
        rhs->blob_desc_ptr(), nullptr);
    CudaCheck(cudaMemcpy(mut_cpu_rhs->mut_dptr(), rhs->dptr(),
                         rhs->ByteSizeOfDataContentField(),
                         cudaMemcpyDeviceToHost));
    cpu_rhs = mut_cpu_rhs;
  }
  OpKernelTestUtil<DeviceType::kCPU>::template BlobCmp<T>(
      blob_name, cpu_lhs, DeviceType::kCPU, cpu_rhs, DeviceType::kCPU);
}

template<>
template<typename T>
void OpKernelTestUtil<DeviceType::kGPU>::CheckInitializeResult(
    const Blob* blob, const InitializerConf& initializer_conf) {
  Blob* cpu_blob = OpKernelTestUtil<DeviceType::kCPU>::CreateBlob(
      blob->blob_desc_ptr(), nullptr);
  CudaCheck(cudaMemcpy(cpu_blob->mut_dptr(), blob->dptr(),
                       blob->ByteSizeOfDataContentField(),
                       cudaMemcpyDeviceToHost));
  OpKernelTestUtil<DeviceType::kCPU>::template CheckInitializeResult<T>(
      cpu_blob, initializer_conf);
}

#define INSTANTIATE_METHODS(type_cpp, type_proto)                      \
  template Blob*                                                       \
  OpKernelTestUtil<DeviceType::kGPU>::CreateBlobWithSpecifiedValPtr(   \
      const BlobDesc* blob_desc, type_cpp* val, Regst* regst);         \
  template void OpKernelTestUtil<DeviceType::kGPU>::BlobCmp<type_cpp>( \
      const std::string& blob_name, const Blob* lhs, DeviceType,       \
      const Blob* rhs, DeviceType);                                    \
  template void                                                        \
  OpKernelTestUtil<DeviceType::kGPU>::CheckInitializeResult<type_cpp>( \
      const Blob* blob, const InitializerConf& initializer_conf);
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_METHODS, ALL_DATA_TYPE_SEQ);

#endif

}  // namespace test

}  // namespace oneflow
