#include "task/fsm/task_compute_bp_fsm.h"
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
TaskComputeBPFSM<Dtype>::TaskComputeBPFSM(Task<Dtype>* task) :
  TaskFSM<Dtype>(task) {
  // config Tetris
  // 1. set ready source column, initial empty
  // For BackwardComputeTask, producer_num will be 3 (BackwardComputeTask
  // count twice, 1 for model, 1 for activation)
  CHECK_EQ(consumed_group_ids_.size(), 3);
  CHECK_EQ(produced_group_ids_.size(), 2);
  auto& node_manager = caffe::TheOne<Dtype>::node_manager();
  auto& id_map = caffe::TheOne<Dtype>::id_map();

  for (auto consumed_group_id : consumed_group_ids_) {
    group_id_to_col_id_[consumed_group_id] = source_col_id_;
    // set |fp_compute_task_id_|
    auto task_id = id_map->task_id_from_group_id(consumed_group_id);
    if (node_manager->GetTaskById(task_id)->task_type()
      == TaskType::kComputeTask) {
      fp_compute_task_id_ = task_id;
      break;
    }
  }

  std::shared_ptr<TetrisIDColumn<int64_t>> source(new
    TetrisIDColumn<int64_t>(source_col_id_, 3));
  tetris_.Add(source);

  // 2. set produced reference counting column
  // two TetrisFIFOColumns (for activation diff and weight gradient diff,
  // it is possible that the activation diff register does not have consumer)
  for (auto produced_group_id : produced_group_ids_) {
    group_id_to_col_id_[produced_group_id] = produced_group_id;

    auto consumers =
      task->task_dag()->GetConsumersOfGroupId(produced_group_id);
    CHECK_EQ(consumers.size(), 1);
    group_id_to_consumer_id_[produced_group_id] = consumers[0];

    std::shared_ptr<TetrisRefCountColumn<int64_t>> ref_count(new
      TetrisRefCountColumn<int64_t>(produced_group_id, 1));

    // fill full
    auto& register_ids
      = task->task_context()->register_ids_in_group(produced_group_id);
    for (auto register_id : register_ids) {
      ref_count->Push(register_id);
    }

    tetris_.Add(ref_count);
  }
}

template <typename Dtype>
TaskComputeBPFSM<Dtype>::~TaskComputeBPFSM() {}

template <typename Dtype>
TaskItem* TaskComputeBPFSM<Dtype>::GetTaskItem() {
  // CHECK(HasAction());
  // The info required for TaskItem
  int32_t task_id = task_->task_id();
  int64_t data_id;
  std::vector<int64_t> source_ids;
  std::vector<int64_t> dst_ids;

  // Set these info
  auto tetris_items = tetris_.Pop();
  auto source_item_it = tetris_items.find(source_col_id_);
  CHECK(source_item_it != tetris_items.end());
  for (auto reg_dataid_pair : source_item_it->second) {
    source_ids.push_back(reg_dataid_pair.first);
    data_id = reg_dataid_pair.second;
  }
  for (auto produced_group_id : produced_group_ids_) {
    auto dst_item_it =
      tetris_items.find(group_id_to_col_id_[produced_group_id]);
    CHECK(dst_item_it != tetris_items.end());
    CHECK_EQ(dst_item_it->second.size(), 1);
    dst_ids.push_back(dst_item_it->second[0].first);
  }

  // Set TaskItem
  // Will be released when action completes
  TaskItem* task_item = new TaskItem();
  task_item->set_task_id(task_id);
  task_item->set_data_id(data_id);

  auto& id_map = caffe::TheOne<Dtype>::id_map();
  auto& node_manager = caffe::TheOne<Dtype>::node_manager();
  // For execution
  for (auto source : source_ids) {
    task_item->add_src_register(source);

    // Consumed msgs
    bool is_model = node_manager->is_in_model_path(source);
    MsgPtr msg(new EventMessage(
      task_id, id_map->task_id_from_register_id(source),
      data_id, source, MessageType::kConsumed));
    if (is_model) {
      msg->set_to_task_id(fp_compute_task_id_);
      msg->set_is_model(true);
    }
    task_item->add_msg(msg);
  }

  for (auto dst : dst_ids) {
    task_item->add_dst_register(dst);

    // send Produced msg to consumers
    auto group_id = id_map->group_id_from_register_id(dst);
    auto to_task_id = group_id_to_consumer_id_[group_id];
    MsgPtr msg(new EventMessage(
      task_id, to_task_id, data_id, dst, MessageType::kProduced));
  }

  return task_item;
}

INSTANTIATE_CLASS(TaskComputeBPFSM);
}  // namespace caffe
