#ifndef ONEFLOW_CORE_KERNEL_KERNEL_TEST_COMMON_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_TEST_COMMON_H_

#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

namespace test {

enum class FillType {
  kDoNotFill = 0,
  kConstant = 1,
  kUniform = 2,
  kGaussian = 3
};

template<DeviceType device_type, typename FloatingPointType>
class KernelTestCommon final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelTestCommon);
  KernelTestCommon() = delete;

  static Blob* CreateBlobWithVector(const std::vector<int64_t>& dim_vec,
                                    FloatingPointType* data_vec);

  static Blob* CreateBlobWithSameValue(const std::vector<int64_t>& dim_vec,
                                       FloatingPointType value);

  static Blob* CreateBlobWithRandomValue(const std::vector<int64_t>& dim_vec);

  static void BuildKernelCtx(KernelCtx* ctx);

  static void SyncStream(KernelCtx* ctx);

  static void BlobCmp(Blob* lhs, Blob* rhs);

  static void CheckResult(
      std::function<Blob*(const std::string&)> BnInOp2BlobPtr,
      const std::string& check, const std::string& expected);

  static void CheckDistribution(const Blob& check_blob,
                                const FillConf& fill_conf);
};

}  // namespace test
}  // namespace oneflow
#endif  // ONEFLOW_CORE_KERNEL_KERNEL_TEST_COMMON_H_
