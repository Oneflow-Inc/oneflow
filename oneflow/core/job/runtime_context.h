#ifndef ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_
#define ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_

#include "oneflow/core/common/thread_safe_counter.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/persistence/data_reader.h"

namespace oneflow {

class RuntimeCtx final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RuntimeCtx);
  ~RuntimeCtx() = default;

  OF_SINGLETON(RuntimeCtx);

  int64_t this_machine_id() const { return this_machine_id_; }

  void set_this_machine_name(const std::string& name);

  ThreadSafeCounter& mut_model_init_cnt() { return model_init_cnt_; }

  DataReader* GetDataReader(const std::string& name);
  void AddDataReader(const std::string& name, DataReader* data_reader);

  ThreadSafeCounter& mut_active_actor_cnt() { return active_actor_cnt_; }
  ThreadSafeCounter& mut_inactive_actor_cnt() { return inactive_actor_cnt_; }

 private:
  RuntimeCtx() { LOG(INFO) << "RuntimeCtx Init"; }

  int64_t this_machine_id_;
  std::string this_machine_name_;

  ThreadSafeCounter model_init_cnt_;

  HashMap<std::string, std::unique_ptr<DataReader>> data_reader_;

  ThreadSafeCounter active_actor_cnt_;
  ThreadSafeCounter inactive_actor_cnt_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_
