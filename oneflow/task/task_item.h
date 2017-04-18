#ifndef _TASK_ITEM_H_
#define _TASK_ITEM_H_
#include <memory>
#include <vector>
#include <cstdint>
#include "runtime/event_message.h"

namespace oneflow {
// TaskItem describes a work item which a task will execute. It is used to
// exchange information between TaskSM, TaskOp and TaskConsequence.
class TaskItem {
 public:
  TaskItem() {}
  ~TaskItem() {}

  void set_task_id(int32_t task_id) { task_id_ = task_id; }
  void set_data_id(int64_t data_id) { data_id_ = data_id; }
  void add_src_register(int64_t register_id) {
    src_register_ids_.push_back(register_id);
    register_ids_.push_back(register_id);
  }
  void add_dst_register(int64_t register_id) {
    dst_register_ids_.push_back(register_id);
    register_ids_.push_back(register_id);
  }
  //void add_consumer_task_id(int32_t task_id) {
  //  consumer_task_ids_.push_back(task_id);
  //}
  void add_msg(MsgPtr msg) {
    msgs_.push_back(msg);
  }

  const std::vector<MsgPtr>& msgs() const {
    return msgs_;
  }
  int32_t task_id() const { return task_id_; }
  int64_t data_id() const { return data_id_; }
  const std::vector<int64_t>& register_ids() const {
    return register_ids_;
  }
  const std::vector<int64_t>& src_register_ids() const {
    return src_register_ids_;
  }
  const std::vector<int64_t>& dst_register_ids() const {
    return dst_register_ids_;
  }
  //const std::vector<int32_t>& consumer_task_ids() const {
  //  return consumer_task_ids_;
  //}

 private:
  // NOTE(v-kayin):
  // TaskItem just need to keep src_register_ids_ and dst_register_id_ for
  // execution, msgs_ for callback. These is enough for TaskOp and
  // TaskConsequence.
  // Callback messages are set by TaskSM(easier to decide what kind of messages
  // should be sent).

  int32_t task_id_{ -1 };                  // The task id of current task
  int64_t data_id_{ -1 };                  // On which data this item works
  std::vector<int64_t> register_ids_;
  std::vector<int64_t> src_register_ids_;  // The source register ids
  std::vector<int64_t> dst_register_ids_;          // One destination register id
  // std::vector<int32_t> consumer_task_ids_; // The task ids of consumers who are
                                           // interested in the dst_register_id_

  std::vector<MsgPtr> msgs_;
};
}  // namespace oneflow
#endif  // _TASK_ITEM_H_
