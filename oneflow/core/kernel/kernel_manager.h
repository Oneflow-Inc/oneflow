#ifndef ONEFLOW_CORE_KERNEL_KERNEL_MANAGER_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_MANAGER_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class KernelMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelMgr);
  ~KernelMgr() = default;

  OF_SINGLETON(KernelMgr);

  const Kernel* GetKernelFromOpName(const std::string& op_name) {
    return op_name2kernel_ptr_.at(op_name).get();
  }

  void InitFromPlan(const Plan&);

 private:
  KernelMgr() = default;
  HashMap<std::string, std::unique_ptr<const Kernel>> op_name2kernel_ptr_;
};

void AddKernelCreator(OperatorConf::OpTypeCase, DeviceType,
                      std::function<Kernel*(const OperatorConf&)> creator);

void AddKernelCreator(OperatorConf::OpTypeCase, DeviceType,
                      Kernel* (*creator)(const OperatorConf&));

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_MANAGER_H_
