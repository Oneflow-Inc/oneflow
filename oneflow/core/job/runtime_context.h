#ifndef ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_
#define ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_

#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

class RuntimeCtx final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RuntimeCtx);
  RuntimeCtx() = delete;
  ~RuntimeCtx() = default;

  OF_SINGLETON(RuntimeCtx);

  int64_t this_machine_id() const { return this_machine_id_; }
  bool IsThisMachineMaster() const { return this_machine_id_ == 0; }
  std::string GetThisCtrlAddr() const { return GetCtrlAddr(this_machine_id_); }
  std::string GetMasterCtrlAddr() const { return GetCtrlAddr(0); }
  std::string GetCtrlAddr(int64_t machine_id) const;

  BlockingCounter& mut_model_init_cnt() { return model_init_cnt_; }
  BlockingCounter& mut_running_actor_cnt() { return running_actor_cnt_; }
  BlockingCounter& mut_constructing_actor_cnt() {
    return constructing_actor_cnt_;
  }

 private:
  RuntimeCtx(const std::string& name);

  int64_t this_machine_id_;

  BlockingCounter model_init_cnt_;

  BlockingCounter running_actor_cnt_;
  BlockingCounter constructing_actor_cnt_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_
