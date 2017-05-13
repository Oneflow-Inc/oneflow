#ifndef ONEFLOW_KERNEL_KERNEL_MANAGER_H_
#define ONEFLOW_KERNEL_KERNEL_MANAGER_H_

#include <utility>
#include <memory>
#include <string>
#include "common/util.h"
#include "common/proto_io.h"
#include "kernel/kernel.h"
#include "job/ofelf.pb.h"

namespace oneflow {

class KernelMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelMgr);
  ~KernelMgr() = default;

  static KernelMgr& Singleton() {
    static KernelMgr obj;
    return obj;
  }

  const Kernel* GetKernelFromOpName(const std::string& op_name) {
    return op_name2kernel_ptr_.at(op_name).get();
  }

  void InitFromOpProtos(const PbRpf<OperatorProto>& op_protos) {
    for (const OperatorProto& op_proto : op_protos) {
      const std::string& op_name = op_proto.op_conf().name();
      std::unique_ptr<Kernel> kernel_ptr = std::make_unique<Kernel>();
      kernel_ptr->InitFromOpProto(op_proto);
      CHECK(op_name2kernel_ptr_.emplace(op_name, std::move(kernel_ptr)).second);
    }
  }

 private:
  KernelMgr() = default;
  HashMap<const std::string, std::unique_ptr<Kernel>> op_name2kernel_ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_KERNEL_KERNEL_MANAGER_H_
