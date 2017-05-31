#ifndef ONEFLOW_ACTOR_ACTOR_H_
#define ONEFLOW_ACTOR_ACTOR_H_

#include "common/util.h"
#include "kernel/kernel.h"
#include "common/task.pb.h"
#include "actor/actor_message.h"
#include "register/register.h"

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

 private:
  uint64_t actor_id_;
  KernelWardFunc ward_func_;
  std::vector<ExecKernel> exec_kernel_vec_;

};

} // namespace oneflow

#endif // ONEFLOW_ACTOR_ACTOR_H_
