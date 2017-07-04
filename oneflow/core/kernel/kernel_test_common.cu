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

  static Blob* CreateBlobWithVector(const std::vector<int64_t>& dim_vec,
                                    FloatingPointType* data_vec) {
    void* dptr;
    Shape* shape = new Shape(dim_vec);
    size_t dptr_size = shape->elem_cnt() * sizeof(FloatingPointType);
    CudaCheck(cudaMalloc(&dptr, dptr_size));
    CudaCheck(cudaMemcpy(dptr, data_vec, dptr_size, cudaMemcpyHostToDevice));
    return new Blob(dptr, shape);
  }

  static Blob* CreateBlobWithSameValue(const std::vector<int64_t>& dim_vec,
                                       FloatingPointType value) {
    Shape* shape = new Shape(dim_vec);
    FloatingPointType* data_vec = new FloatingPointType[shape->elem_cnt()];
    std::fill(data_vec, data_vec + shape->elem_cnt(), value);
    return CreateBlobWithVector(dim_vec, data_vec);
  }

  static Blob* CreateBlobWithRandomValue(const std::vector<int64_t>& dim_vec) {
    Shape* shape = new Shape(dim_vec);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<FloatingPointType> dis(0, 10);
    FloatingPointType* data_vec = new FloatingPointType[shape->elem_cnt()];
    for (int64_t i = 0; i != shape->elem_cnt(); ++i) { data_vec[i] = dis(gen); }
    return CreateBlobWithVector(dim_vec, data_vec);
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
    Blob* copy_lhs =
        KTCommonCpu::CreateBlobWithVector(lhs->shape().dim_vec(), dptr);
    Blob* copy_rhs =
        KTCommonCpu::CreateBlobWithVector(rhs->shape().dim_vec(), dptr);
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
};

template class KernelTestCommon<DeviceType::kGPU, float>;
template class KernelTestCommon<DeviceType::kGPU, double>;

}  // namespace test
}  // namespace oneflow
