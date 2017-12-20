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

template<DeviceType device_type>
void SetBlobDataId(Blob*, const std::vector<std::string>&);

using RandomValConf = HashMap<std::string, BlobDesc*>;
using SameValConf = HashMap<std::string, std::tuple<float, BlobDesc*>>;
template<typename T>
using SpecifiedValConf =
    HashMap<std::string, std::tuple<std::vector<T>*, BlobDesc*>>;

template<DeviceType device_type, typename T>
class KTCommon final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KTCommon);
  KTCommon() = delete;

  static Blob* CreateBlobWithSpecifiedVal(const BlobDesc*, T* val);

  static Blob* CreateBlobWithSpecifiedVal(const BlobDesc* blob_desc,
                                          std::vector<T>& val) {
    return CreateBlobWithSpecifiedVal(blob_desc, &(val[0]));
  }

  static std::function<Blob*(const std::string)> ConstructBnInOp2BlobFunc(
      RandomValConf random_val_conf, SameValConf same_val_conf,
      SpecifiedValConf<T> specified_vals_conf) {
    auto bn2blob = new HashMap<std::string, Blob*>;
    BlobDesc* blob_desc = nullptr;
    for (auto blob_conf_pair : random_val_conf) {
      blob_desc = blob_conf_pair.second;
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<double> dis(0, 10);
      T* val_vec = new T[blob_desc->shape().elem_cnt()];
      for (int64_t i = 0; i < blob_desc->shape().elem_cnt(); ++i) {
        val_vec[i] = static_cast<T>(dis(gen));
      }
      (*bn2blob)[blob_conf_pair.first] =
          CreateBlobWithSpecifiedVal(blob_desc, val_vec);
    }
    for (auto blob_conf_pair : same_val_conf) {
      float val = 0.f;
      std::tie(val, blob_desc) = blob_conf_pair.second;
      T* val_vec = new T[blob_desc->shape().elem_cnt()];
      std::fill(val_vec, val_vec + blob_desc->shape().elem_cnt(),
                static_cast<T>(val));
      (*bn2blob)[blob_conf_pair.first] =
          CreateBlobWithSpecifiedVal(blob_desc, val_vec);
    }
    for (auto blob_conf_pair : specified_vals_conf) {
      std::vector<T>* specific_nums = nullptr;
      std::tie(specific_nums, blob_desc) = blob_conf_pair.second;
      (*bn2blob)[blob_conf_pair.first] =
          CreateBlobWithSpecifiedVal(blob_desc, *specific_nums);
    }
    return [bn2blob](const std::string& bn) { return bn2blob->at(bn); };
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
