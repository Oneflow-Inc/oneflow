#ifndef ONEFLOW_CORE_KERNEL_KERNEL_TEST_COMMON_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_TEST_COMMON_H_

#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

namespace test {

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
};

#define INSTANTIATE_CPU_KERNEL_TEST_COMMON_CLASS(classname) \
  char gInstantiationGuardCPU##classname;                   \
  template class classname<DeviceType::kCPU, float>;        \
  template class classname<DeviceType::kCPU, double>;
#define INSTANTIATE_GPU_KERNEL_TEST_COMMON_CLASS(classname) \
  char gInstantiationGuardGPU##classname;                   \
  template class classname<DeviceType::kGPU, float>;        \
  template class classname<DeviceType::kGPU, double>;

}  // namespace test
}  // namespace oneflow
#endif  // ONEFLOW_CORE_KERNEL_KERNEL_TEST_COMMON_H_
