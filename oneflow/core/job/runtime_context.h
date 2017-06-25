#ifndef ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_
#define ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/id_manager.h"

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

 private:
  RuntimeCtx() = default;

  int64_t this_machine_id_;
  std::string this_machine_name_;

  int32_t model_init_cnt_;
  std::mutex model_init_cnt_mtx_;
  std::condition_variable model_init_cnt_cond_;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_
