#include "oneflow/core/kernel/kernel_test_common.h"
#include <random>
#include "oneflow/core/device/cpu_device_context.h"

namespace oneflow {

namespace test {
/*
template<typename FloatingPointType>
class KernelTestCommon<DeviceType::kCPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelTestCommon);
  KernelTestCommon() = delete;

  static Blob* CreateBlobWithVector(const std::vector<int64_t>& dim_vec,
                                    FloatingPointType* data_vec) {
    void* dptr;
    Shape* shape = new Shape(dim_vec);
    size_t dptr_size = shape->elem_cnt() * sizeof(FloatingPointType);
    CudaCheck(cudaMallocHost(&dptr, dptr_size));
    CudaCheck(cudaMemcpy(dptr, data_vec, dptr_size, cudaMemcpyHostToHost));
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
    auto cpu_stream = new AsyncCpuStream;
    ctx->device_ctx = new CpuDeviceCtx(cpu_stream);
  }

  static void SyncStream(KernelCtx* ctx) {
    ctx->device_ctx->cpu_stream()->CloseSendEnd();
    auto cpu_thread = std::thread([&] {
      std::function<void()> work;
      while (ctx->device_ctx->cpu_stream()->ReceiveWork(&work) == 0) { work(); }
    });
    cpu_thread.join();
  }

  static void BlobCmp(Blob* lhs, Blob* rhs) {
    const FloatingPointType* dptr_lhs = lhs->dptr<FloatingPointType>();
    const FloatingPointType* dptr_rhs = rhs->dptr<FloatingPointType>();
    size_t dptr_size = lhs->shape().elem_cnt();

    for (size_t i = 0; i < dptr_size; ++i) {
      ASSERT_FLOAT_EQ(dptr_lhs[i], dptr_rhs[i]);
    }
  }

  static void CheckResult(
      std::function<Blob*(const std::string&)> BnInOp2BlobPtr,
      const std::string& check, const std::string& expected) {
    return BlobCmp(BnInOp2BlobPtr(check), BnInOp2BlobPtr(expected));
  }

  static void CheckFillResult(const Blob& check_blob,
                              const FillConf& fill_conf) {
    size_t dptr_size = check_blob.shape().elem_cnt();
    const FloatingPointType* dptr =
        static_cast<const FloatingPointType*>(check_blob.dptr());
    if (fill_conf.has_constant_conf()) {
      for (size_t i = 0; i < dptr_size; ++i) {
        ASSERT_FLOAT_EQ(dptr[i], fill_conf.constant_conf().value());
      }
    } else if (fill_conf.has_uniform_conf()) {
      TODO();
    } else if (fill_conf.has_gaussian_conf()) {
      TODO();
    } else {
      UNEXPECTED_RUN();
    }
  }
};

template class KernelTestCommon<DeviceType::kCPU, float>;
template class KernelTestCommon<DeviceType::kCPU, double>;
*/
}  // namespace test
}  // namespace oneflow
