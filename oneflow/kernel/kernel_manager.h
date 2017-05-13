#ifndef ONEFLOW_KERNEL_KERNEL_MANAGER_H_
#define ONEFLOW_KERNEL_KERNEL_MANAGER_H_

#include <utility>
#include <memory>
#include <string>
#include "common/util.h"
#include "kernel/kernel.h"
#include "job/ofelf.pb.h"

namespace oneflow {

class KernelMgr {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelMgr);
  ~KernelMgr() = default;

  static KernelMgr& Singleton() {
    static KernelMgr obj;
    return obj;
  }

  const Kernel* OpName2Kernel(const std::string op_name) {
    CHECK_NE(op_name2Kernel_ptr_.find(op_name), op_name2Kernel_ptr_.end());
    return op_name2Kernel_ptr_.at(op_name).get();
  }

  using OpProtos = google::protobuf::RepeatedPtrField<oneflow::OperatorProto>&;
  void InitFromOpProtos(const OpProtos op_protos) {
    for (auto op_proto : op_protos) {
      const std::string op_name = op_proto.op_conf().name();
      std::unique_ptr<Kernel> kernel_ptr = std::make_unique<Kernel>();
      kernel_ptr->InitFromOpProto(op_proto);
      CHECK(op_name2Kernel_ptr_.emplace(op_name, std::move(kernel_ptr)).second);
    }
  }

 private:
  KernelMgr() = default;
  HashMap<const std::string, std::unique_ptr<Kernel>> op_name2Kernel_ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_KERNEL_KERNEL_MANAGER_H_
