#ifndef ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_
#define ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/persistence/persistent_circular_line_reader.h"

namespace oneflow {

class RuntimeCtx final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RuntimeCtx);
  ~RuntimeCtx() = default;

  OF_SINGLETON(RuntimeCtx);

  int64_t this_machine_id() const { return this_machine_id_; }

  void set_this_machine_name(const std::string& name);

  void SetModelInitCnt(int32_t val);
  void OneModelInitDone();
  void WaitUnitlAllModelInitDone();

  PersistentCircularLineReader* GetDataReader() { return data_reader_.get(); }
  void InitDataReader(const std::string& filepath);

  void set_active_actor_cnt(int64_t val) {
    std::unique_lock<std::mutex> lck(active_actor_cnt_mtx_);
    active_actor_cnt_ = val;
  }
  void active_actor_cnt_minus1() {
    std::unique_lock<std::mutex> lck(active_actor_cnt_mtx_);
    active_actor_cnt_ -= 1;
    if (active_actor_cnt_ == 0) { active_actor_cnt_cond_.notify_all(); }
  }
  void WaitUntilNoActiveActor() {
    std::unique_lock<std::mutex> lck(active_actor_cnt_mtx_);
    active_actor_cnt_cond_.wait(lck,
                                [this]() { return active_actor_cnt_ == 0; });
  }

 private:
  RuntimeCtx() { LOG(INFO) << "RuntimeCtx Init"; }

  int64_t this_machine_id_;
  std::string this_machine_name_;

  int32_t model_init_cnt_;
  std::mutex model_init_cnt_mtx_;
  std::condition_variable model_init_cnt_cond_;
  std::unique_ptr<PersistentCircularLineReader> data_reader_;

  int64_t active_actor_cnt_;
  std::mutex active_actor_cnt_mtx_;
  std::condition_variable active_actor_cnt_cond_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_
