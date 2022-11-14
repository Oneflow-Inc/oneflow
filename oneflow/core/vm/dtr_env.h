#include <vector>
#include <oneflow/core/vm/op_call_instruction_policy.h>

namespace oneflow {

namespace vm {
class TensorStorage;
}

namespace dtr {
class Env {
 public:
  double time_now() { return time_now_; }
  void add_time(double time) { time_now_ += time; }
  void remove_compute_op(vm::DtrOpCallInstructionPolicy* op) {
    ops.erase(std::remove(ops.begin(), ops.end(), op), ops.end());
  }
  vm::OpCallInstructionPolicy update_tensor_with_storage(vm::TensorStorage* storage, vm::OpCallInstructionPolicy* current_compute_op);

  std::vector<vm::DtrOpCallInstructionPolicy*> ops;

 private:
  double time_now_;
};
}  // namespace dtr
}  // namespace oneflow
