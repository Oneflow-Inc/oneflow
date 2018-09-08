#ifndef ONEFLOW_CORE_KERNEL_SHARED_MODEL_DIFF_ADD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SHARED_MODEL_DIFF_ADD_KERNEL_H_

#include "oneflow/core/kernel/add_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class SharedModelDiffAddKernel : public AddKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SharedModelDiffAddKernel);
  SharedModelDiffAddKernel() = default;
  ~SharedModelDiffAddKernel() = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override {
    return this->op_conf().shared_model_diff_add_conf()
  }
};

}  // namespace oneflow

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSharedModelDiffAddConf, SharedModelDiffAddKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

#endif  // ONEFLOW_CORE_KERNEL_SHARED_MODEL_DIFF_ADD_KERNEL_H_
