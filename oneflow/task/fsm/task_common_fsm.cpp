#include "task/fsm/task_common_fsm.h"
#include <vector>
#include <utility>
#include <cstdint>
#include "common/common.h"
#include "task/task_item.h"
#include "task/task_context.h"
#include "task/node_manager.h"
#include "task/job_manager.h"
#include "dag/task_dag.h"
#include "thread/event_message.h"

namespace caffe {
// Common FSM for all path, including kCopyTask, kNetTask
template <typename Dtype>
TaskCommonFSM<Dtype>::TaskCommonFSM(Task<Dtype>* task) : TaskFSM<Dtype>(task) {
  // config Tetris
  // two columns, both of them are TetrisFIFOColumn
  CHECK_EQ(consumed_group_ids_.size(), 1);
  CHECK_EQ(produced_group_ids_.size(), 1);
  auto consumers
    = task->task_dag()->GetConsumersOfGroupId(produced_group_ids_[0]);
  CHECK_EQ(consumers.size(), 1);
  consumer_id_ = consumers[0];

  // 1. set ready source column, initial empty
  group_id_to_col_id_[consumed_group_ids_[0]] = source_col_id_;
  std::shared_ptr<TetrisFIFOColumn<int64_t>> source(new
    TetrisFIFOColumn<int64_t>(source_col_id_));
  tetris_.Add(source);

  // 2. set produced reference counting column, namely available dst,
  // initial full
  auto produced_group_id = produced_group_ids_[0];
  std::shared_ptr<TetrisRefCountColumn<int64_t>> ref_count(new
    TetrisRefCountColumn<int64_t>(produced_group_id, 1));
  group_id_to_col_id_[produced_group_id] = produced_group_id;

  // fill full
  auto register_ids
    = task->task_context()->register_ids_in_group(produced_group_id);
  for (auto register_id : register_ids) {
    ref_count->Push(register_id);
  }

  tetris_.Add(ref_count);
}

template <typename Dtype>
TaskCommonFSM<Dtype>::~TaskCommonFSM() {}

template <typename Dtype>
TaskItem* TaskCommonFSM<Dtype>::GetTaskItem() {
  // CHECK(HasAction());
  // The info required for TaskItem
  int32_t task_id = task_->task_id();
  int64_t data_id;
  int64_t source_id;
  int64_t dst_id;

  // Set these info
  auto tetris_items = tetris_.Pop();
  auto source_item_it = tetris_items.find(source_col_id_);
  CHECK(source_item_it != tetris_items.end());
  CHECK_EQ(source_item_it->second.size(), 1);
  source_id = source_item_it->second[0].first;
  data_id = source_item_it->second[0].second;

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
  task_item->add_src_register(source_id);
  task_item->add_dst_register(dst_id);

  // 2. For callback messages
  // 2.1 All actors need to send Produced msg to consumers, unless no consumers.
  MsgPtr produced_msg(new EventMessage(
    task_id, consumer_id_, data_id, dst_id, MessageType::kProduced));
  task_item->add_msg(produced_msg);


  // 2.2 Consumed msgs
  auto& id_map = caffe::TheOne<Dtype>::id_map();
  auto consumed_to_task_id = id_map->task_id_from_register_id(source_id);
  MsgPtr consumed_msg(new EventMessage(
    task_id, consumed_to_task_id, data_id, source_id, MessageType::kConsumed));
  task_item->add_msg(consumed_msg);

  return task_item;
}

INSTANTIATE_CLASS(TaskCommonFSM);
}  // namespace caffe
