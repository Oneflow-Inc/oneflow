#ifndef _TASK_FSM_FACTORY_H_
#define _TASK_FSM_FACTORY_H_
#include <glog/logging.h>
#include "task/task.h"
#include "task/fsm/task_fsm.h"
#include "task/fsm/task_boxing_fsm.h"
#include "task/fsm/task_common_fsm.h"
#include "task/fsm/task_compute_bp_fsm.h"
#include "task/fsm/task_compute_fp_test_fsm.h"
#include "task/fsm/task_compute_fp_train_fsm.h"
#include "task/fsm/task_compute_model_fsm.h"
#include "task/fsm/task_data_fsm.h"

namespace oneflow {
template <typename Dtype>
class TaskFSMFactory {
 public:
  static std::shared_ptr<TaskFSM<Dtype>> CreateFSM(Task<Dtype>* task);
};
}  // namespace oneflow
#endif  // _TASK_FSM_FACTORY_H_
