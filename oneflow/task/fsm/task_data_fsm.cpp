#include "task/fsm/task_data_fsm.h"
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
template <typename Dtype>
TaskDataFSM<Dtype>::TaskDataFSM(Task<Dtype>* task) :
  TaskFSM<Dtype>(task),
  piece_id_(-1) {
  // Ensure: thread_id == device_id
  //auto& id_map = caffe::TheOne<Dtype>::id_map();
  //device_id_ = id_map->thread_id_from_task_id(task_->task_id());
  device_id_ = 0;
  CHECK_EQ(consumed_group_ids_.size(), 0);
  CHECK_EQ(produced_group_ids_.size(), 1);
  auto consumers
    = task->task_dag()->GetConsumersOfGroupId(produced_group_ids_[0]);
  CHECK_EQ(consumers.size(), 1);
  consumer_id_ = consumers[0];

  auto produced_group_id = produced_group_ids_[0];
  group_id_to_col_id_[produced_group_id] = produced_group_id;

  // config Tetris
  // Single column, uses SimpleFIFOColumn
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
TaskDataFSM<Dtype>::~TaskDataFSM() {}

template <typename Dtype>
int64_t TaskDataFSM<Dtype>::new_data_id() {
  piece_id_++;
  auto& id_map = caffe::TheOne<Dtype>::id_map();
  return id_map->data_id_from_device_and_piece(device_id_, piece_id_);
}

template <typename Dtype>
TaskItem* TaskDataFSM<Dtype>::GetTaskItem() {
  //CHECK(HasAction());
  // The info required for TaskItem
  int32_t task_id = task_->task_id();
  int64_t data_id = new_data_id();
  int64_t dst_id;

  // Set these info
  auto tetris_items = tetris_.Pop();
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
  task_item->add_dst_register(dst_id);

  // 2. For callback messages
  // 2.1 All actors need to send Produced msg to consumers, unless no consumers.
  MsgPtr produced_msg(new EventMessage(
    task_id, consumer_id_, data_id, dst_id, MessageType::kProduced));
  task_item->add_msg(produced_msg);
  // Data Task do not need send consumed msg.

  return task_item;
}

INSTANTIATE_CLASS(TaskDataFSM);
}  // namespace caffe
