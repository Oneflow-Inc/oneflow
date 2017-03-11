#include "task/fsm/task_fsm.h"
#include <vector>
#include <utility>
#include "common/common.h"
#include "task/task_item.h"
#include "task/task_context.h"
#include "task/node_manager.h"
#include "task/job_manager.h"
#include "dag/task_dag.h"
#include "thread/event_message.h"
#include "thread/comm_bus.h"

namespace caffe {
  // NOTE(v-kayin): Only consider DataPath
template <typename Dtype>
TaskFSM<Dtype>::TaskFSM(Task<Dtype>* task,
  std::shared_ptr<TaskFSMInfo> task_fsm_info) :
  task_(task),
  task_fsm_info_(task_fsm_info),
  source_col_name_("source"),
  ref_col_name_("ref_count"),
  forward_compute_train_(false),
  bp_compute_task_id_(-1),
  backward_compute_train_(false),
  fp_compute_task_id_(-1) {
  // Ensure: thread_id == device_id
  auto& id_map = caffe::TheOne<Dtype>::id_map();
  device_id_ = id_map->thread_id_from_task_id(task_->task_id());

  auto producers = task_fsm_info_->Producers();
  auto consumers = task_fsm_info_->Consumers();

  // config Tetris
  // 1. set ready source column, initial empty
  auto producer_num = producers.size();
  if (task_->task_type() == TaskType::kComputeTask
    && task_->task_dag()->is_forward()) {
    // TODO(v-kayin): get SSP staleness from config,
    // for phase = TEST, staleness should be -1
    int32_t staleness = 1;
    if (true) {  // phase = TEST
      staleness = -1;
    } else {
      forward_compute_train_ = true;
      auto& job_manager = caffe::TheOne<Dtype>::job_manager();
      for (auto consumer_id : consumers) {
        auto& consumer_dag = job_manager->GetTaskDag(consumer_id);
        if (consumer_dag->task_type() == TaskType::kComputeTask) {
          bp_compute_task_id_ = consumer_id;
          break;
        }
      }
      CHECK_NE(bp_compute_task_id_, -1)
        << "TaskFSM error: bp_compute_task_id shouble be set";
    }
    std::shared_ptr<SimpleTetrisSSPColumn<int64_t>> source(new
      SimpleTetrisSSPColumn<int64_t>(source_col_name_, staleness));
    tetris_.Add(source);
  } else {
    if (task_->task_type() == TaskType::kComputeTask) {
      // For BackwardComputeTask, producer_num will be 3 (BackwardComputeTask
      // count twice, 1 for model, 1 for activation)
      producer_num++;
      backward_compute_train_ = true;
      auto& job_manager = caffe::TheOne<Dtype>::job_manager();
      for (auto producer_id : producers) {
        auto& producer_dag = job_manager->GetTaskDag(producer_id);
        if (producer_dag->task_type() == TaskType::kComputeTask) {
          fp_compute_task_id_ = producer_id;
          break;
        }
      }
      CHECK_NE(fp_compute_task_id_, -1)
        << "TaskFSM error: fp_compute_task_id shouble be set";
    }
    // if producer_num<1 then no need for source_col
    if (producer_num == 1) {
      std::shared_ptr<SimpleTetrisFIFOColumn<int64_t>> source(new
        SimpleTetrisFIFOColumn<int64_t>(source_col_name_));
      tetris_.Add(source);
    } else if (producer_num > 1) {
      std::shared_ptr<SimpleTetrisIDColumn<int64_t>> source(new
        SimpleTetrisIDColumn<int64_t>(source_col_name_, producer_num));
      tetris_.Add(source);
    }
  }

  // 2. set produced reference counting column, namely available dst,
  // initial full
  auto consumer_num = consumers.size();
  if (consumer_num == 1) {
    std::shared_ptr<SimpleTetrisFIFOColumn<int64_t>> ref_count(new
      SimpleTetrisFIFOColumn<int64_t>(ref_col_name_));
    tetris_.Add(ref_count);
  } else {
    std::shared_ptr<SimpleTetrisIDColumn<int64_t>> ref_count(new
      SimpleTetrisIDColumn<int64_t>(ref_col_name_, consumer_num));
    tetris_.Add(ref_count);
  }
  // Fill ref_count column full initially
  // FIXME(v-kayin): not sure about the way to get dst register ids, ie. Copy task
  // TODO(jiyuan): get produced group ids, get register ids of each group
  //auto& register_ids = task_->task_context()->GetDataRegisterIds();
  //for (auto register_id : register_ids) {
  //  // Fill all dst register_id element in column full with consumer task ids
  //  for (auto consumer_id : consumers) {
  //    tetris_.Push(ref_col_name_, (int64_t)consumer_id, register_id);
  //  }
  //}
}

template <typename Dtype>
TaskFSM<Dtype>::~TaskFSM() {}

template <typename Dtype>
void TaskFSM<Dtype>::ConsumeMessage(std::shared_ptr<EventMessage> msg) {
  if (msg->message_type() == MessageType::kProduced) {
    // deal with "produced" message
    tetris_.Push(source_col_name_, msg->register_id(), msg->data_id());
  } else if (msg->message_type() == MessageType::kConsumed) {
    // deal with "consumed" message
    if (msg->is_model()) {
      tetris_.Push(source_col_name_, msg->register_id(), msg->data_id());
    } else {
      tetris_.Push(ref_col_name_, msg->from_task_id(), msg->register_id());
    }
  }
}

template <typename Dtype>
bool TaskFSM<Dtype>::HasAction() const {
  return tetris_.HasReady();
}

template <typename Dtype>
TaskItem* TaskFSM<Dtype>::GetTaskItem() {
  CHECK(HasAction());
  auto producers = task_fsm_info_->Producers();
  auto consumers = task_fsm_info_->Consumers();
  // The info required for TaskItem
  int32_t task_id = task_->task_id();
  int64_t data_id;
  std::vector<int64_t> ready_sources;
  int64_t dst_id;

  // Set these info
  auto tetris_items = tetris_.Pop();
  if (task_->task_type() == TaskType::kDataTask) {
    data_id = new_data_id();
  } else {
    auto source_item_it = tetris_items.find(source_col_name_);
    CHECK(source_item_it != tetris_items.end());
    bool data_id_tag = true;
    for (auto reg_dataid_pair : source_item_it->second) {
      ready_sources.push_back(reg_dataid_pair.first);
      if (data_id_tag) {
        data_id = reg_dataid_pair.second;
        data_id_tag = false;
      }
    }
  }
  auto ref_item_it = tetris_items.find(ref_col_name_);
  CHECK(ref_item_it != tetris_items.end());
  dst_id = ref_item_it->second[0].second;

  // Set TaskItem
  // Will be released when action completes
  TaskItem* task_item = new TaskItem();
  task_item->set_task_id(task_id);
  task_item->set_data_id(data_id);

  // 1. For execution
  for (auto source : ready_sources) {
    task_item->add_src_register(source);
  }
  task_item->set_dst_register_id(dst_id);

  // 2. For callback messages
  // 2.1 All actors need to send Produced msg to consumers, unless no consumers.
  for (auto consumer_id : consumers) {
    std::shared_ptr<EventMessage> msg(new EventMessage());
    msg->set_data_id(data_id);
    msg->set_from_task_id(task_id);
    msg->set_message_type(MessageType::kProduced);
    msg->set_register_id(dst_id);
    msg->set_to_task_id(consumer_id);
    task_item->add_msg(msg);
  }
  // ForwardComputeTask send model to BackwardComputeTask
  if (forward_compute_train_) {
    auto model_register = ready_sources[1];
    std::shared_ptr<EventMessage> msg(new EventMessage());
    msg->set_data_id(data_id);
    msg->set_from_task_id(task_id);
    msg->set_message_type(MessageType::kProduced);
    msg->set_register_id(model_register);
    msg->set_to_task_id(bp_compute_task_id_);
    msg->set_is_model(true);
    task_item->add_msg(msg);
  }

  // 2.2 Consumed msgs
  auto& id_map = caffe::TheOne<Dtype>::id_map();
  if (!backward_compute_train_) {
    // ALL Tasks own registers now. We can get "to_task_id" from "register_id"
    // directly, except model register for backward_compute
    for (auto source : ready_sources) {
      std::shared_ptr<EventMessage> msg(new EventMessage());
      msg->set_data_id(data_id);
      msg->set_from_task_id(task_id);
      msg->set_message_type(MessageType::kConsumed);
      msg->set_register_id(source);
      msg->set_to_task_id(id_map->task_id_from_register_id(source));
      task_item->add_msg(msg);
    }
  } else {
    for (auto source : ready_sources) {
      // TODO(v-kayin): source register, model or data?
      bool is_model = true;
      std::shared_ptr<EventMessage> msg(new EventMessage());
      msg->set_data_id(data_id);
      msg->set_from_task_id(task_id);
      msg->set_message_type(MessageType::kConsumed);
      msg->set_register_id(source);
      if (is_model) {
        msg->set_to_task_id(fp_compute_task_id_);
        msg->set_is_model(true);
      } else {
        msg->set_to_task_id(id_map->task_id_from_register_id(source));
      }
      task_item->add_msg(msg);
    }
  }

  return task_item;
}
template <typename Dtype>
int64_t TaskFSM<Dtype>::new_data_id() {
  // TODO(v-kayin): data id: device_id, batch_id, mini batch id
  piece_id_++;
  auto& id_map = caffe::TheOne<Dtype>::id_map();
  return id_map->data_id_from_device_and_piece(device_id_, piece_id_);
}

INSTANTIATE_CLASS(TaskFSM);
}  // namespace caffe
