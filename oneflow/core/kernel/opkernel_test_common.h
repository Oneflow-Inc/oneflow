#ifndef ONEFLOW_CORE_KERNEL_OPKERNEL_TEST_COMMON_H_
#define ONEFLOW_CORE_KERNEL_OPKERNEL_TEST_COMMON_H_

#include "oneflow/core/common/test_util.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

namespace test {

std::function<BlobDesc*(const std::string)> ConstructBn2BlobDescFunc(
    std::shared_ptr<Operator>);

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

  static Blob* CreateBlobWithSpecifiedVal(const BlobDesc* blob_desc,
                                          std::vector<T> val) {
    return CreateBlobWithSpecifiedValPtr(blob_desc, &(val[0]));
  }

  static Blob* CreateBlobWithSameVal(const BlobDesc* blob_desc, T val) {
    std::vector<T> val_vec(blob_desc->shape().elem_cnt(), val);
    return CreateBlobWithSpecifiedVal(blob_desc, val_vec);
  }

  static Blob* CreateBlobWithRandomVal(const BlobDesc* blob_desc) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0, 10);
    std::vector<T> val_vec(blob_desc->shape().elem_cnt());
    for (int64_t i = 0; i < blob_desc->shape().elem_cnt(); ++i) {
      val_vec[i] = static_cast<T>(dis(gen));
    }
    return CreateBlobWithSpecifiedVal(blob_desc, val_vec);
  }

  static void BlobCmp(const std::string& blob_name, const Blob* lhs,
                      const Blob* rhs);

  static void CheckResult(std::function<Blob*(const std::string&)> BnInOp2Blob,
                          const std::string& result,
                          const std::string& expected_result) {
    BlobCmp("", BnInOp2Blob(result), BnInOp2Blob(expected_result));
  }

  static void CheckInitializeResult(const Blob* blob,
                                    const InitializerConf& initializer_conf);

 private:
  static Blob* CreateBlobWithSpecifiedValPtr(const BlobDesc*, T* val);
};

}  // namespace test

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_OPKERNEL_TEST_COMMON_H_
