#include "task/fsm/task_compute_model_fsm.h"
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
/*
  TaskComputeModelFSM deal with 2 kinds of operation:
  1, add diff to gradient
  2, add gradient to model

  For now, we assume all diff arrive in data-id order, that is we create a
  single CUDA stream for one task.
*/

template <typename Dtype>
TaskComputeModelFSM<Dtype>::TaskComputeModelFSM(Task<Dtype>* task) :
TaskFSM<Dtype>(task), added_diff_num_(0), model_inited_(0) {
  CHECK_EQ(consumed_group_ids_.size(), 1);
  CHECK_EQ(produced_group_ids_.size(), 2);
  // TODO(v-kayin): set batch size
  diff_num_ = 1;

  // assume produced_group_ids_ contain: gradient group id, then model group id
  auto gradient_group_id = produced_group_ids_[0];
  auto& gradient_register_ids
    = task->task_context()->register_ids_in_group(gradient_group_id);
  CHECK_EQ(gradient_register_ids.size(), 1);
  gradient_register_id_ = gradient_register_ids[0];

  // set produced model reference counting column in tetris
  auto produced_group_id = produced_group_ids_[1];
  group_id_to_col_id_[produced_group_id] = produced_group_id;
  consumer_ids_ = task->task_dag()->
    GetConsumersOfGroupId(produced_group_id);

  auto consumer_num = consumer_ids_.size();  // how many?
  std::shared_ptr<TetrisRefCountColumn<int64_t>> ref_count(new
    TetrisRefCountColumn<int64_t>(produced_group_id, consumer_num));
  // fill full
  auto& register_ids
    = task->task_context()->register_ids_in_group(produced_group_id);
  for (auto register_id : register_ids) {
    for (auto consumer_id : consumer_ids_) {
      ref_count->Push(register_id);
    }
  }
}

template <typename Dtype>
TaskComputeModelFSM<Dtype>::~TaskComputeModelFSM() {}

template <typename Dtype>
void TaskComputeModelFSM<Dtype>::ProcessMessage(MsgPtr msg) {
  if (msg->message_type() == MessageType::kProduced) {
    if (!msg->is_model()) {
      // diff "produced" message
      diff_ids_.push({ msg->register_id(), msg->data_id() });
    } else {
      // how about init model here with LoadPath?
      newest_model_register_id_ = msg->register_id();
      model_inited_ = 1;
    }
  } else if (msg->message_type() == MessageType::kConsumed) {
    // model "consumed" message
    auto col_id = register_id_to_col_id(msg->register_id());
    tetris_.Push(col_id, msg->register_id(), msg->data_id());
  }
}

template <typename Dtype>
bool TaskComputeModelFSM<Dtype>::HasTaskItem() const {
  return !diff_ids_.empty();
}

template <typename Dtype>
TaskItem* TaskComputeModelFSM<Dtype>::GetTaskItem() {
  // CHECK(HasAction());
  int32_t task_id = task_->task_id();
  int64_t data_id = diff_ids_.front().second;
  int64_t diff_id = diff_ids_.front().first;
  diff_ids_.pop();

  TaskItem* task_item = new TaskItem();
  task_item->set_task_id(task_id);
  // No matter whether add gradient to model, data_id in TaskItem will always
  // be diff version
  task_item->set_data_id(data_id);

  // register order in task_item:
  // diff-reg-id, gradient-reg-id,(old-model-reg-id, new-model-reg-id)

  // 1, add diff to gradient
  task_item->add_src_register(diff_id);
  added_diff_num_++;
  task_item->add_dst_register(gradient_register_id_);
  // Diff consumed msg
  auto& id_map = oneflow::TheOne<Dtype>::id_map();
  auto diff_task_id = id_map->task_id_from_register_id(diff_id);
  MsgPtr diff_consumed_msg(new EventMessage(
    task_id, diff_task_id, data_id, diff_id, MessageType::kConsumed));
  task_item->add_msg(diff_consumed_msg);

  // 2, add gradient to model
  if (added_diff_num_ >= diff_num_ && tetris_.Ready()) {
    CHECK_NE(model_inited_, 0);
    task_item->add_src_register(newest_model_register_id_);

    auto tetris_items = tetris_.Pop();
    auto ref_item_it
      = tetris_items.find(group_id_to_col_id_[produced_group_ids_[1]]);
    CHECK(ref_item_it != tetris_items.end());
    CHECK_EQ(ref_item_it->second.size(), 1);
    auto model_dst = ref_item_it->second[0].first;

    task_item->add_dst_register(model_dst);

    if (model_inited_ == 1) {
      // Let the initial model register become available
      auto col_id = group_id_to_col_id_[produced_group_ids_[1]];
      for (auto consumer_id : consumer_ids_) {
        tetris_.Push(col_id, newest_model_register_id_, 0);
      }
      model_inited_ = 2;
    }
    newest_model_register_id_ = model_dst;

    // New model produced message
    for (auto consumer_id : consumer_ids_) {
      MsgPtr msg(new EventMessage(
        task_id, consumer_id, data_id, model_dst, MessageType::kProduced));
      task_item->add_msg(msg);
    }
  }

  return task_item;
}

INSTANTIATE_CLASS(TaskComputeModelFSM);
}  // namespace oneflow
