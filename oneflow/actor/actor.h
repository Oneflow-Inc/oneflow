#ifndef ONEFLOW_ACTOR_ACTOR_H_
#define ONEFLOW_ACTOR_ACTOR_H_

#include "common/util.h"
#include "kernel/kernel.h"
#include "common/task.pb.h"
#include "actor/actor_message.pb.h"

namespace oneflow {


class Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Actor);
  virtual ~Actor() = default;

  virtual void Init(const TaskProto&);
  virtual void ProcessMsg(const ActorMsg&) = 0;

  uint64_t actor_id() const { return actor_id_; }
 
 protected:
  struct ExecKernel {
    const Kernel* kernel;
    std::unordered_map<std::string, uint64_t> bn_in_op2regst_desc_id_;
  };

  Actor() = default;
  void set_ward_func(KernelWardFunc val) { ward_func_ = val; }

 private:
  // Init at this
  uint64_t actor_id_;
  std::vector<ExecKernel> exec_kernel_vec_;
  std::unordered_map<uint64_t, std::string> regst_desc_id2regst_desc_name_;
  KernelWardFunc ward_func_;

};

} // namespace oneflow

#endif // ONEFLOW_ACTOR_ACTOR_H_
