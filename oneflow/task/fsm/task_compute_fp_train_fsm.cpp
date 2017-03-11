#include "task/fsm/task_compute_fp_train_fsm.h"
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
TaskComputeFPTrainFSM<Dtype>::TaskComputeFPTrainFSM(Task<Dtype>* task) :
  TaskFSM<Dtype>(task) {
  /*
   * kComputeTask:
   * Forward: (train)
   * Src side: TetrisSSPColumn (includes input data register and model
   * register)
   * Dst side: TetrisIDColumn (includes activation, requires reference
   * counting, usually has two consumers, activation to downstream copyd2h,
   * backward correspondence. If the forward compute task is the last one,
   * it does not have copyd2h)
   */
  CHECK_EQ(consumed_group_ids_.size(), 2);
  CHECK_EQ(produced_group_ids_.size(), 1);

  // config Tetris
  // 1. set ready source column, initial empty
  for (auto consumed_group_id : consumed_group_ids_) {
    group_id_to_col_id_[consumed_group_id] = source_col_id_;
  }
  // TODO(v-kayin): get SSP staleness from config
  int32_t staleness = 1;
  std::shared_ptr<TetrisSSPColumn<int64_t, Dtype>> source(new
    TetrisSSPColumn<int64_t, Dtype>(source_col_id_, staleness,
    task->task_id()));
  tetris_.Add(source);


  // 2. set produced reference counting column, namely available dst,
  // initial full
  auto produced_group_id = produced_group_ids_[0];
  group_id_to_col_id_[produced_group_id] = produced_group_id;
  consumer_ids_ = task->task_dag()->
    GetConsumersOfGroupId(produced_group_id);

  auto consumer_num = consumer_ids_.size();
  CHECK_EQ(consumer_num, 2);
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

  // set |bp_compute_task_id_|
  auto& node_manager = caffe::TheOne<Dtype>::node_manager();
  for (auto consumer_id : consumer_ids_) {
    if (node_manager->GetTaskById(consumer_id)->task_type()
      == TaskType::kComputeTask) {
      bp_compute_task_id_ = consumer_id;
      break;
    }
  }

  tetris_.Add(ref_count);
}

template <typename Dtype>
TaskComputeFPTrainFSM<Dtype>::~TaskComputeFPTrainFSM() {}

template <typename Dtype>
TaskItem* TaskComputeFPTrainFSM<Dtype>::GetTaskItem() {
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
  CHECK_EQ(source_item_it->second.size(), 2);
  source_ids.push_back(source_item_it->second[0].first);
  data_id = source_item_it->second[0].second;
  source_ids.push_back(source_item_it->second[1].first);

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
  for (auto consumer_id : consumer_ids_) {
    MsgPtr msg(new EventMessage(
      task_id, consumer_id, data_id, dst_id, MessageType::kProduced));
    task_item->add_msg(msg);
  }
  // ForwardComputeTask send model to BackwardComputeTask
  auto model_register = source_ids[1];
  MsgPtr model_msg(new EventMessage(
    task_id, bp_compute_task_id_, data_id,
    model_register, MessageType::kProduced));
  model_msg->set_is_model(true);
  task_item->add_msg(model_msg);

  // 2.2 Consumed msgs
  auto& id_map = caffe::TheOne<Dtype>::id_map();
  auto source_id = source_ids[0];
  auto consumed_to_task_id = id_map->task_id_from_register_id(source_id);
  MsgPtr consumed_msg(new EventMessage(
    task_id, consumed_to_task_id, data_id, source_id, MessageType::kConsumed));
  task_item->add_msg(consumed_msg);

  return task_item;
}

INSTANTIATE_CLASS(TaskComputeFPTrainFSM);
}  // namespace caffe
