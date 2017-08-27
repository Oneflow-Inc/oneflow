#ifndef ONEFLOW_CORE_KERNEL_KERNEL_TEST_COMMON_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_TEST_COMMON_H_

#include "oneflow/core/device/async_cpu_stream.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

namespace test {

template<DeviceType device_type>
Blob* CreateBlob(const BlobDesc*);

template<DeviceType device_type>
void BuildKernelCtx(KernelCtx* ctx);

template<DeviceType device_type>
void SyncStream(KernelCtx* ctx);

template<DeviceType device_type, typename T>
class KTCommon final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KTCommon);
  KTCommon() = delete;

  static Blob* CreateBlobWithSpecifiedVal(const BlobDesc*, T* val);

  static Blob* CreateBlobWithSameVal(const BlobDesc*, T val);

  static Blob* CreateBlobWithRandomVal(const BlobDesc*);

  static void BlobCmp(const Blob* lhs, const Blob* rhs);

  static void CheckResult(std::function<Blob*(const std::string&)> BnInOp2Blob,
                          const std::string& check,
                          const std::string& expected);

  static void CheckFillResult(const Blob* blob, const FillConf& fill_conf);
};

}  // namespace test
}  // namespace oneflow
#endif  // ONEFLOW_CORE_KERNEL_KERNEL_TEST_COMMON_H_
