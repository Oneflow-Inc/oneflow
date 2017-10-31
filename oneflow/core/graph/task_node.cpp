#include "oneflow/core/graph/task_node.h"

namespace oneflow {

TaskNode::TaskNode() : machine_id_(-1), thrd_loc_id_(-1), task_id_(-1) {}

void TaskNode::set_machine_id(int64_t val) {
  machine_id_ = val;
  if (thrd_loc_id_ != -1) { UpdateTaskId(); }
}

void TaskNode::set_thrd_loc_id(int64_t val) {
  thrd_loc_id_ = val;
  if (machine_id_ != -1) { UpdateTaskId(); }
}

void TaskNode::UpdateTaskId() {
  CHECK_NE(machine_id_, -1);
  CHECK_NE(thrd_loc_id_, -1);
  task_id_ = IDMgr::Singleton()->NewTaskId(machine_id_, thrd_loc_id_);
}

std::string TaskNode::VisualStr() const {
  std::stringstream ss;
  ss << TodoTaskType_Name(GetTaskType()) << "\\n"
     << machine_id_ << "\\n"
     << thrd_loc_id_ << "\\n"
     << task_id_;
  return ss.str();
}

void TaskNode::NewProducedRegst(const std::string& name,
                                int32_t min_register_num,
                                int32_t max_register_num) {
  auto regst = std::make_shared<RegstDesc>();
  regst->set_producer(this);
  regst->set_min_register_num(min_register_num);
  regst->set_max_register_num(max_register_num);
  CHECK(produced_regsts_.emplace(name, regst).second);
}

}  // namespace oneflow
