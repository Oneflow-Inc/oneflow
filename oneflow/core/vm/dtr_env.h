#include <vector>
#include <oneflow/core/vm/op_call_instruction_policy.h>

namespace oneflow {

namespace vm {
class TensorStorage;
}

namespace dtr {
class Env {
 public:
  Env() = default;
  ~Env();
  OF_DISALLOW_COPY_AND_MOVE(Env);
  double time_now() { return time_now_; }
  void add_time(double time) { time_now_ += time; }
  void remove_compute_op(vm::DtrOpCallInstructionPolicy* op) {
    ops.erase(std::remove(ops.begin(), ops.end(), op), ops.end());
  }
  vm::OpCallInstructionPolicy update_tensor_with_storage(vm::TensorStorage* storage, vm::OpCallInstructionPolicy* current_compute_op);

  std::vector<vm::DtrOpCallInstructionPolicy*> ops;

  void add_eviction_num(bool eager_eviction);

  int eager_eviction_num() const { return eager_eviction_num_; }
  int forced_eviction_num() const { return forced_eviction_num_; }

  void add_recomputation_num() { recomputation_num_++; }
  int recomputation_num() const { return recomputation_num_; }

  void clear_time() { time_now_ = 0; }

  std::set<vm::TensorStorage*> need_eager_eviction_storages;

 private:
  double time_now_ = 0;

  int eager_eviction_num_ = 0;
  int forced_eviction_num_ = 0;
  int recomputation_num_ = 0;
};
}  // namespace dtr
}  // namespace oneflow
