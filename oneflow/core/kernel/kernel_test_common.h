#ifndef ONEFLOW_CORE_KERNEL_KERNEL_TEST_COMMON_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_TEST_COMMON_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/device/async_cpu_stream.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/register/blob_desc.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename FloatingPointType>
class KernelTestCommon final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelTestCommon);
  KernelTestCommon() = delete;

  static Blob* CreateBlobWithVector(const BlobDesc* blob_desc,
                                    FloatingPointType* data_vec);

  static Blob* CreateBlobWithSameValue(const BlobDesc* blob_desc,
                                       FloatingPointType value);

  static Blob* CreateBlobWithRandomValue(const BlobDesc* blob_desc);

  static void BuildKernelCtx(KernelCtx* ctx);

  static void SyncStream(KernelCtx* ctx);

  static void BlobCmp(Blob* lhs, Blob* rhs);

  static void CheckResult(
      std::function<Blob*(const std::string&)> BnInOp2BlobPtr,
      const std::string& check, const std::string& expected);

  static void CheckFillResult(const Blob& check_blob,
                              const FillConf& fill_conf);
  static std::shared_ptr<BlobDesc> CreateBlobDesc(
      const std::vector<int64_t>& dim_vec, DataType dt) {
    std::shared_ptr<BlobDesc> blob_desc(new BlobDesc);
    blob_desc->mut_shape() = Shape(dim_vec);
    blob_desc->set_data_type(dt);
    blob_desc->set_has_data_id(false);
    return blob_desc;
  }
};

}  // namespace test
}  // namespace oneflow
#endif  // ONEFLOW_CORE_KERNEL_KERNEL_TEST_COMMON_H_
