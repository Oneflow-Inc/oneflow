#include "oneflow/core/kernel/opkernel_test_common.h"
#include <random>
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/device/cpu_device_context.h"

namespace oneflow {

namespace test {

std::function<BlobDesc*(const std::string&)> ConstructBn2BlobDescFunc(
    std::shared_ptr<Operator> op) {
  auto InsertBnsWithEmptyBlobDesc2Map =
      [](const std::vector<std::string>& bns,
         HashMap<std::string, BlobDesc*>* bn2blobdesc_map) {
        for (const std::string& bn : bns) {
          CHECK(bn2blobdesc_map->insert({bn, new BlobDesc}).second);
        }
      };
  auto bn2blobdesc_map = new HashMap<std::string, BlobDesc*>();
  InsertBnsWithEmptyBlobDesc2Map(op->data_tmp_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->input_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->input_diff_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->output_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->output_diff_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->model_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->model_diff_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->model_tmp_bns(), bn2blobdesc_map);
  return [bn2blobdesc_map](const std::string& bn) {
    return bn2blobdesc_map->at(bn);
  };
}

std::function<Blob*(const std::string&)> ConstructBn2BlobFunc() {
  auto bn2blob_map = new HashMap<std::string, Blob*>();
  return [bn2blob_map](const std::string& bn) {
    if (bn2blob_map->find(bn) == bn2blob_map->end()) {
      bn2blob_map->insert({bn, new Blob});
    }
    return bn2blob_map->at(bn);
  };
}

template<>
void* MallocAndClean<DeviceType::kCPU>(size_t sz) {
  void* mem_ptr = nullptr;
  CudaCheck(cudaMallocHost(&mem_ptr, sz));
  memset(mem_ptr, 0, sz);
  return mem_ptr;
}

template<>
Blob* CreateBlob<DeviceType::kCPU>(const BlobDesc* blob_desc) {
  void* mem_ptr = MallocAndClean<DeviceType::kCPU>(blob_desc->TotalByteSize());
  return new Blob(blob_desc, static_cast<char*>(mem_ptr));
}

template<>
void BuildKernelCtx<DeviceType::kCPU>(KernelCtx* ctx) {
  ctx->device_ctx = new CpuDeviceCtx(-1);
}

template<>
void SyncStream<DeviceType::kCPU>(KernelCtx* ctx) {}

template<>
void CopyFromHost<DeviceType::kCPU>(void* dst, const void* src, size_t sz) {
  CudaCheck(cudaMemcpy(dst, src, sz, cudaMemcpyHostToHost));
}

template<typename T>
class KTCommon<DeviceType::kCPU, T> final {
 public:
  static void BlobCmp(const Blob* lhs, const Blob* rhs) {
    ASSERT_EQ(lhs->blob_desc(), rhs->blob_desc());
    CHECK_EQ(lhs->data_type(), GetDataType<T>::val);
    CHECK_EQ(lhs->has_data_id(), rhs->has_data_id());
    if (IsFloatingPoint(lhs->data_type())) {
      for (int64_t i = 0; i < lhs->shape().elem_cnt(); ++i) {
        ASSERT_FLOAT_EQ(lhs->dptr<T>()[i], rhs->dptr<T>()[i]);
      }
    } else {
      ASSERT_EQ(
          memcmp(lhs->dptr(), rhs->dptr(), lhs->ByteSizeOfDataContentField()),
          0);
    }
    if (lhs->has_data_id() == false) { return; }
    CHECK_EQ(
        memcmp(lhs->data_id(0), rhs->data_id(0), lhs->ByteSizeOfDataIdField()),
        0);
  }

  static void CheckFillResult(const Blob* blob, const FillConf& fill_conf) {
    if (fill_conf.has_constant_conf()) {
      for (int64_t i = 0; i < blob->shape().elem_cnt(); ++i) {
        ASSERT_FLOAT_EQ(blob->dptr<T>()[i], fill_conf.constant_conf().value());
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
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KTCOMMON, ALL_DATA_TYPE_SEQ)

}  // namespace test

}  // namespace oneflow
