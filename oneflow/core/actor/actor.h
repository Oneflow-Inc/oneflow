#ifndef ONEFLOW_CORE_ACTOR_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_ACTOR_H_

#include <queue>
#include "oneflow/core/common/util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/common/task.pb.h"
#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/actor/actor_msg_bus.h"
#include "oneflow/core/register/register.h"
#include "oneflow/core/register/register_manager.h"

namespace oneflow {

class Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Actor);
  virtual ~Actor() = default;

  virtual void Init(const TaskProto& task_proto);
  virtual void ProcessMsg(const ActorMsg&) = 0;

  uint64_t actor_id() const { return actor_id_; }
 
 protected:
  struct ExecKernel {
    const Kernel* kernel;
    std::unordered_map<std::string, uint64_t> bn_in_op2regst_desc_id;
  };

  Actor() = default;
  void WardKernel(std::function<Regst*(uint64_t)> GetRegstFromRegstDescId);
  const std::vector<std::unique_ptr<Regst>>& produced_regst_vec() const {
    return produced_regst_vec_;
  }
  uint64_t GetRegstDescIdFromName(const std::string& name) const {
    return name2regst_desc_id_.at(name);
  }

 private:
  uint64_t actor_id_;
  KernelWardFunc ward_func_;
  std::vector<ExecKernel> exec_kernel_vec_;
  std::vector<std::unique_ptr<Regst>> produced_regst_vec_;
  HashMap<std::string, uint64_t> name2regst_desc_id_;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_ACTOR_H_
