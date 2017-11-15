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
  KernelMgr() = delete;
  ~KernelMgr() = default;

  OF_SINGLETON(KernelMgr);

  const Kernel* GetKernelFromOpName(const std::string& op_name) {
    return op_name2kernel_ptr_.at(op_name).get();
  }

 private:
  KernelMgr(const Plan&);
  HashMap<std::string, std::unique_ptr<const Kernel>> op_name2kernel_ptr_;
};

using KernelCreator1 =
    std::function<Kernel*(const OperatorConf&, const OpContext&)>;
using KernelCreator2 = std::function<Kernel*(const OperatorConf&)>;
using KernelCreator3 = std::function<Kernel*(const OpContext&)>;
using KernelCreator4 = std::function<Kernel*()>;

void AddKernelCreator(OperatorConf::OpTypeCase, KernelCreator1);
void AddKernelCreator(OperatorConf::OpTypeCase, KernelCreator2);
void AddKernelCreator(OperatorConf::OpTypeCase, KernelCreator3);
void AddKernelCreator(OperatorConf::OpTypeCase, KernelCreator4);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_MANAGER_H_
