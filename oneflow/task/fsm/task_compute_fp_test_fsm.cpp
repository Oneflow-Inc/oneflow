#include "task/fsm/task_compute_fp_test_fsm.h"
#include <vector>
#include <utility>
#include <cstdint>
#include "common/common.h"
#include "task/task_item.h"
#include "task/task_context.h"
#include "task/node_manager.h"
#include "task/job_manager.h"
#include "runtime/event_message.h"

namespace oneflow {
template <typename Dtype>
TaskComputeFPTestFSM<Dtype>::TaskComputeFPTestFSM(Task<Dtype>* task) :
  TaskFSM<Dtype>(task) {
  CHECK_EQ(consumed_group_ids_.size(), 1);
  CHECK_EQ(produced_group_ids_.size(), 1);
  auto consumers
    = task->task_dag()->GetConsumersOfGroupId(produced_group_ids_[0]);
  CHECK_EQ(consumers.size(), 1);
  consumer_id_ = consumers[0];

  // config Tetris
  // 1. set ready source column, initial empty
  group_id_to_col_id_[consumed_group_ids_[0]] = source_col_id_;
  std::shared_ptr<TetrisFIFOColumn<int64_t>> source(new
    TetrisFIFOColumn<int64_t>(source_col_id_));
  tetris_.Add(source);


  // 2. set produced reference counting column, namely available dst,
  // initial full
  auto produced_group_id = produced_group_ids_[0];
  group_id_to_col_id_[produced_group_id] = produced_group_id;
  std::shared_ptr<TetrisRefCountColumn<int64_t>> ref_count(new
    TetrisRefCountColumn<int64_t>(produced_group_id, 1));

  // fill full
  auto register_ids
    = task->task_context()->register_ids_in_group(produced_group_id);
  for (auto register_id : register_ids) {
    ref_count->Push(register_id);
  }

  tetris_.Add(ref_count);
}

template <typename Dtype>
TaskComputeFPTestFSM<Dtype>::~TaskComputeFPTestFSM() {}

template <typename Dtype>
void TaskComputeFPTestFSM<Dtype>::ProcessMessage(MsgPtr msg) {
  if (msg->is_model()) {
    model_register_id_ = msg->register_id();
  } else {
    auto col_id = register_id_to_col_id(msg->register_id());
    tetris_.Push(col_id, msg->register_id(), msg->data_id());
  }
}

template <typename Dtype>
bool TaskComputeFPTestFSM<Dtype>::HasTaskItem() const {
  return tetris_.Ready() && (model_register_id_ > -1);
}

template <typename Dtype>
TaskItem* TaskComputeFPTestFSM<Dtype>::GetTaskItem() {
  // CHECK(HasAction());
  // The info required for TaskItem
  int32_t task_id = task_->task_id();
  int64_t data_id;
  std::vector<int64_t> source_ids;
  int64_t dst_id;

  // Set these info
  auto tetris_items = tetris_.Pop();
  auto source_item_it = tetris_items.find(source_col_id_);
  CHECK(source_item_it != tetris_items.end());
  CHECK_EQ(source_item_it->second.size(), 1);
  source_ids.push_back(source_item_it->second[0].first);
  data_id = source_item_it->second[0].second;
  source_ids.push_back(model_register_id_);

  auto ref_item_it
    = tetris_items.find(group_id_to_col_id_[produced_group_ids_[0]]);
  CHECK(ref_item_it != tetris_items.end());
  CHECK_EQ(ref_item_it->second.size(), 1);
  dst_id = ref_item_it->second[0].first;

  // Set TaskItem
  // Will be released when action completes
  TaskItem* task_item = new TaskItem();
  task_item->set_task_id(task_id);
  task_item->set_data_id(data_id);

  // 1. For execution
  for (auto source : source_ids) {
    task_item->add_src_register(source);
  }
  task_item->add_dst_register(dst_id);

  // 2. For callback messages
  // 2.1 All actors need to send Produced msg to consumers, unless no consumers.
  MsgPtr produced_msg(new EventMessage(
    task_id, consumer_id_, data_id, dst_id, MessageType::kProduced));
  task_item->add_msg(produced_msg);

  // 2.2 Consumed msgs
  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  auto source_id = source_ids[0];
  auto consumed_to_task_id = id_map->task_id_from_register_id(source_id);
  MsgPtr consumed_msg(new EventMessage(
    task_id, consumed_to_task_id, data_id, source_id, MessageType::kConsumed));
  task_item->add_msg(consumed_msg);

  return task_item;
}

INSTANTIATE_CLASS(TaskComputeFPTestFSM);
}  // namespace oneflow
