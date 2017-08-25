#include <random>
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

template<typename FloatingPointType>
class KernelTestCommon<DeviceType::kGPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelTestCommon);
  KernelTestCommon() = delete;

  static Blob* CreateBlobWithVector(const BlobDesc* blob_desc,
                                    FloatingPointType* data_vec) {
    char* dptr;
    int64_t elem_cnt = blob_desc->shape().elem_cnt();
    size_t dptr_size = elem_cnt * sizeof(FloatingPointType);
    CudaCheck(cudaMallocHost(&dptr, dptr_size));
    CudaCheck(cudaMemcpy(dptr, data_vec, dptr_size, cudaMemcpyHostToDevice));
    return new Blob(blob_desc, dptr);
  }

  static Blob* CreateBlobWithSameValue(const BlobDesc* blob_desc,
                                       FloatingPointType value) {
    int64_t elem_cnt = blob_desc->shape().elem_cnt();
    FloatingPointType* data_vec = new FloatingPointType[elem_cnt];
    std::fill(data_vec, data_vec + elem_cnt, value);
    return CreateBlobWithVector(blob_desc, data_vec);
  }

  static Blob* CreateBlobWithRandomValue(const BlobDesc* blob_desc) {
    int64_t elem_cnt = blob_desc->shape().elem_cnt();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<FloatingPointType> dis(0, 10);
    FloatingPointType* data_vec = new FloatingPointType[elem_cnt];
    for (int64_t i = 0; i != elem_cnt; ++i) { data_vec[i] = dis(gen); }
    return CreateBlobWithVector(blob_desc, data_vec);
  }

  static void BuildKernelCtx(KernelCtx* ctx) {
    cudaStream_t* cuda_stream = new cudaStream_t;
    cublasHandle_t* cublas_handle = new cublasHandle_t;
    CudaCheck(cudaStreamCreate(cuda_stream));
    CudaCheck(cublasCreate(cublas_handle));
    CudaCheck(cublasSetStream(*cublas_handle, *cuda_stream));
    ctx->device_ctx = new CudaDeviceCtx(cuda_stream, cublas_handle, nullptr);
  }

  static void SyncStream(KernelCtx* ctx) {
    CudaCheck(cudaStreamSynchronize(ctx->device_ctx->cuda_stream()));
  }

  static void BlobCmp(Blob* lhs, Blob* rhs) {
    using KTCommonCpu = KernelTestCommon<DeviceType::kCPU, FloatingPointType>;

    FloatingPointType* dptr;
    size_t dptr_size = lhs->shape().elem_cnt() * sizeof(FloatingPointType);
    CudaCheck(cudaMallocHost(&dptr, dptr_size));
    memset(dptr, 0, dptr_size);
    Blob* copy_lhs = KTCommonCpu::CreateBlobWithVector(&lhs->blob_desc(), dptr);
    Blob* copy_rhs = KTCommonCpu::CreateBlobWithVector(&rhs->blob_desc(), dptr);
    CudaCheck(cudaMemcpy(copy_lhs->mut_dptr(), lhs->dptr(), dptr_size,
                         cudaMemcpyDeviceToHost));
    CudaCheck(cudaMemcpy(copy_rhs->mut_dptr(), rhs->dptr(), dptr_size,
                         cudaMemcpyDeviceToHost));

    KTCommonCpu::BlobCmp(copy_lhs, copy_rhs);
  }

  static void CheckResult(
      std::function<Blob*(const std::string&)> BnInOp2BlobPtr,
      const std::string& check, const std::string& expected) {
    BlobCmp(BnInOp2BlobPtr(check), BnInOp2BlobPtr(expected));
  }

  static void CheckFillResult(const Blob& check_blob,
                              const FillConf& fill_conf) {
    using KTCommonCpu = KernelTestCommon<DeviceType::kCPU, FloatingPointType>;

    char* host_raw_dptr = nullptr;
    size_t byte_size =
        check_blob.shape().elem_cnt() * sizeof(FloatingPointType);
    CudaCheck(cudaMallocHost(&host_raw_dptr, byte_size));
    memset(host_raw_dptr, 0, byte_size);
    std::unique_ptr<char, std::function<void(char*)>> host_unique_ptr(
        host_raw_dptr, [&](char* dptr) { CudaCheck(cudaFreeHost(dptr)); });
    std::unique_ptr<Shape> host_blob_shape(new Shape(check_blob.shape()));

    std::unique_ptr<Blob> copy_check_blob(
        new Blob(&check_blob.blob_desc(), host_unique_ptr.get()));
    CudaCheck(cudaMemcpy(copy_check_blob->mut_dptr(), check_blob.dptr(),
                         byte_size, cudaMemcpyDeviceToHost));

    KTCommonCpu::CheckFillResult(*copy_check_blob, fill_conf);
  }
};  // namespace test

template class KernelTestCommon<DeviceType::kGPU, float>;
template class KernelTestCommon<DeviceType::kGPU, double>;

}  // namespace test
}  // namespace oneflow
