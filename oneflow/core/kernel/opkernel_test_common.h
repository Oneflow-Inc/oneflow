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

enum BlobInitMethod { kRandom, kSpecific };

struct BlobInitConf {
  BlobInitConf(float specific_num, BlobDesc* blob_desc) {
    this->init_method = kSpecific;
    this->specific_num = specific_num;
    this->blob_desc = blob_desc;
  }
  BlobInitConf(BlobDesc* blob_desc) {
    this->init_method = kRandom;
    this->specific_num = 0;
    this->blob_desc = blob_desc;
  }
  BlobInitMethod init_method;
  float specific_num;
  BlobDesc* blob_desc;
};

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

  static Blob* CreateBlobWithSpecifiedVal(const BlobDesc*, T* val);
  static Blob* CreateBlobWithSpecifiedVal(const BlobDesc* blob_desc,
                                          std::vector<T> val) {
    return CreateBlobWithSpecifiedVal(blob_desc, &(val[0]));
  }

  static std::function<Blob*(const std::string)> ConstructBnInOp2BlobFunc(
      HashMap<std::string, BlobInitConf>& bn2blob_init_conf) {
    auto bn2blob = new HashMap<std::string, Blob*>;
    for (auto blob_init_conf_pair : bn2blob_init_conf) {
      BlobInitConf& blob_init_conf = blob_init_conf_pair.second;
      if (blob_init_conf.init_method == kRandom) {
        (*bn2blob)[blob_init_conf_pair.first] =
            CreateBlobWithRandomVal(blob_init_conf.blob_desc);
      } else if (blob_init_conf.init_method == kSpecific) {
        (*bn2blob)[blob_init_conf_pair.first] =
            CreateBlobWithSameVal(blob_init_conf.blob_desc,
                                  static_cast<T>(blob_init_conf.specific_num));
      } else {
        UNEXPECTED_RUN();
      }
    }
    return [bn2blob](const std::string& bn) { return bn2blob->at(bn); };
  }

  static Blob* CreateBlobWithSameVal(const BlobDesc* blob_desc, T val) {
    T* val_vec = new T[blob_desc->shape().elem_cnt()];
    std::fill(val_vec, val_vec + blob_desc->shape().elem_cnt(), val);
    return CreateBlobWithSpecifiedVal(blob_desc, val_vec);
  }

  static Blob* CreateBlobWithRandomVal(const BlobDesc* blob_desc) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0, 10);
    T* val_vec = new T[blob_desc->shape().elem_cnt()];
    for (int64_t i = 0; i < blob_desc->shape().elem_cnt(); ++i) {
      val_vec[i] = static_cast<T>(dis(gen));
    }
    return CreateBlobWithSpecifiedVal(blob_desc, val_vec);
  }

  static void BlobCmp(const Blob* lhs, const Blob* rhs);

  static void CheckResult(std::function<Blob*(const std::string&)> BnInOp2Blob,
                          const std::string& result,
                          const std::string& expected_result) {
    BlobCmp(BnInOp2Blob(result), BnInOp2Blob(expected_result));
  }

  static void CheckFillResult(const Blob* blob, const FillConf& fill_conf);
};

}  // namespace test

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_OPKERNEL_TEST_COMMON_H_
