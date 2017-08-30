#include "oneflow/core/kernel/kernel_test_common.h"
#include <random>
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/device/cpu_device_context.h"

namespace oneflow {

namespace test {

template<>
Blob* CreateBlob<DeviceType::kCPU>(const BlobDesc* blob_desc) {
  void* mem_ptr = nullptr;
  CudaCheck(cudaMallocHost(&mem_ptr, blob_desc->TotalByteSize()));
  return new Blob(blob_desc, static_cast<char*>(mem_ptr));
}

template<>
void BuildKernelCtx<DeviceType::kCPU>(KernelCtx* ctx) {
  auto cpu_stream = new AsyncCpuStream;
  ctx->device_ctx = new CpuDeviceCtx(cpu_stream);
}

template<>
void SyncStream<DeviceType::kCPU>(KernelCtx* ctx) {
  ctx->device_ctx->cpu_stream()->CloseSendEnd();
  auto cpu_thread = std::thread([&] {
    std::function<void()> work;
    while (ctx->device_ctx->cpu_stream()->ReceiveWork(&work) == 0) { work(); }
  });
  cpu_thread.join();
  ctx->device_ctx->cpu_stream()->CloseReceiveEnd();
}

template<typename T>
BlobDesc* CreateDefaultBlobDescWithShape(const std::vector<int64_t>& v) {
  BlobDesc* ret = new BlobDesc();
  ret->mut_shape() = Shape(v);
  ret->set_data_type(GetDataType<T>::val);
  return ret;
}

#define INSTANTIATE_TEMPLATE_MEM_FUNC(func, type_cpp, ...) \
  template BlobDesc* func<type_cpp>(__VA_ARGS__);
#define BLOB_DESC_FUNC_DATA_TYPE(type_cpp, type_proto)                    \
  INSTANTIATE_TEMPLATE_MEM_FUNC(CreateDefaultBlobDescWithShape, type_cpp, \
                                const std::vector<int64_t>&)
FOR_EACH_PAIR(BLOB_DESC_FUNC_DATA_TYPE, ALL_DATA_TYPE_PAIR());

template<typename T>
class KTCommon<DeviceType::kCPU, T> final {
 public:
  static Blob* CreateBlobWithSpecifiedVal(const BlobDesc* blob_desc, T* val) {
    Blob* ret = CreateBlob<DeviceType::kCPU>(blob_desc);
    CudaCheck(cudaMemcpy(ret->mut_dptr(), val, ret->ByteSizeOfDataField(),
                         cudaMemcpyHostToHost));
    return ret;
  }

  static void BlobCmp(const Blob* lhs, const Blob* rhs) {
    ASSERT_EQ(lhs->blob_desc().data_type(), rhs->blob_desc().data_type());
    ASSERT_EQ(lhs->blob_desc().shape(), rhs->blob_desc().shape());
    ASSERT_EQ(lhs->blob_desc().has_data_id(), rhs->blob_desc().has_data_id());
    CHECK_EQ(lhs->data_type(), GetDataType<T>::val);
    if (IsFloatingPoint(lhs->data_type())) {
      for (int64_t i = 0; i < lhs->shape().elem_cnt(); ++i) {
        ASSERT_DOUBLE_EQ(lhs->dptr<T>()[i], rhs->dptr<T>()[i]);
      }
    } else {
      ASSERT_EQ(memcmp(lhs->dptr(), rhs->dptr(), lhs->ByteSizeOfDataField()),
                0);
    }
  }

  static void CheckFillResult(const Blob* blob, const FillConf& fill_conf) {
    if (fill_conf.has_constant_conf()) {
      for (int64_t i = 0; i < blob->shape().elem_cnt(); ++i) {
        ASSERT_DOUBLE_EQ(blob->dptr<T>()[i], fill_conf.constant_conf().value());
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

#define INSTANTIATE_KTCOMMON(type_cpp, type_proto) \
  template class KTCommon<DeviceType::kCPU, type_cpp>;
FOR_EACH_PAIR(INSTANTIATE_KTCOMMON, ALL_DATA_TYPE_PAIR())

}  // namespace test

}  // namespace oneflow
