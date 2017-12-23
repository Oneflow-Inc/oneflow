#ifndef ONEFLOW_CORE_KERNEL_OPKERNEL_TEST_COMMON_H_
#define ONEFLOW_CORE_KERNEL_OPKERNEL_TEST_COMMON_H_

#include "oneflow/core/common/test_util.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

namespace test {

std::function<BlobDesc*(const std::string&)> ConstructBn2BlobDescFunc(
    std::shared_ptr<Operator>);

std::function<Blob*(const std::string&)> ConstructBn2BlobFunc();

template<DeviceType device_type>
void* Malloc(size_t);

template<DeviceType device_type>
Blob* CreateBlob(const BlobDesc*);

template<DeviceType device_type>
void BuildKernelCtx(KernelCtx* ctx);

template<DeviceType device_type>
void SyncStream(KernelCtx* ctx);

template<DeviceType device_type>
void SetBlobDataId(DeviceCtx* ctx, Blob* blob,
                   const std::vector<std::string>& data_ids) {
  CHECK_EQ(blob->has_data_id(), true);
  CHECK_EQ(data_ids.size(), blob->shape().At(0));
  Memset<device_type>(ctx, blob->mut_data_id(0), '\0',
                      blob->ByteSizeOfDataIdField());
  FOR_RANGE(size_t, i, 0, data_ids.size()) {
    CHECK_LE(data_ids[i].size(), JobDesc::Singleton()->SizeOfOneDataId());
    Memcpy<device_type>(ctx, blob->mut_data_id(i), data_ids[i].c_str(),
                        data_ids[i].size());
  }
}

template<DeviceType device_type>
void InitBlobWithBlobDesc(Blob* blob, const BlobDesc* blob_desc) {
  char* mem_ptr =
      static_cast<char*>(Malloc<device_type>(blob_desc->TotalByteSize()));
  blob->data_id_ptr_ = blob_desc->has_data_id() ? mem_ptr : nullptr;
  blob->dptr_ = mem_ptr + blob_desc->ByteSizeOfDataIdField();
  blob->blob_desc_ = blob_desc;
}

template<DeviceType device_type, typename T>
void InitBlobAndFillSameVal(DeviceCtx* ctx, Blob* blob,
                            const BlobDesc* blob_desc, float val) {
  InitBlobWithBlobDesc<device_type>(blob, blob_desc);
  FillConf fill_conf;
  fill_conf.mutable_constant_conf()->set_value(val);
  KernelUtil<device_type, T>::Fill(ctx, fill_conf, 0, blob);
}

template<DeviceType device_type, typename T>
void InitBlobAndFillRandomVal(DeviceCtx* ctx, Blob* blob,
                              const BlobDesc* blob_desc) {
  InitBlobWithBlobDesc<device_type>(blob, blob_desc);
  std::mt19937 random_seed_gen;
  FillConf fill_conf;
  fill_conf.mutable_gaussian_conf()->set_std(10);
  fill_conf.mutable_gaussian_conf()->set_mean(10);
  KernelUtil<device_type, T>::Fill(ctx, fill_conf, random_seed_gen(), blob);
}

template<DeviceType device_type, typename T>
class KTCommon final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KTCommon);
  KTCommon() = delete;

  static void CopyFromFloatVals(T* dst, const float* src, int64_t sz);

  static void BlobCmp(const Blob* lhs, const Blob* rhs);

  static void CheckResult(std::function<Blob*(const std::string&)> BnInOp2Blob,
                          const std::string& result,
                          const std::string& expected_result) {
    BlobCmp(BnInOp2Blob(result), BnInOp2Blob(expected_result));
  }

  static void CheckFillResult(const Blob* blob, const FillConf& fill_conf);
};

template<DeviceType device_type, typename T>
void InitBlobAndFillSpecifiedVal(Blob* blob, const BlobDesc* blob_desc,
                                 const std::vector<float> vals) {
  InitBlobWithBlobDesc<device_type>(blob, blob_desc);
  CHECK_EQ(vals.size(), blob_desc->shape().elem_cnt());
  KTCommon<device_type, T>::CopyFromFloatVals(blob->mut_dptr<T>(), &vals[0],
                                              vals.size());
}

}  // namespace test

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_OPKERNEL_TEST_COMMON_H_
