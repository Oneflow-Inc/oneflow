#include "runtime/comm_bus.h"

namespace oneflow {

CommBus::CommBus(int32_t thread_num_each_client)
  : thread_num_each_client_(thread_num_each_client) {
}

CommBus::~CommBus() {
}

void CommBus::Init() {
  network_ = GetNdspiRDMAInstance();  // Used for inter-node messaging
  for (int32_t i = 0; i < thread_num_each_client_; ++i) {
    queues_.emplace_back(new MtQueue<MsgPtr>());
  }
}

void CommBus::SendMessage(MsgPtr msg) {
  // Get destination machine and thread's id from msg
  // auto& id_map = caffe::TheOne<Dtype>::id_map();
  std::shared_ptr<IDMap> id_map(new IDMap());  //FIXME(jiyuan)
  int32_t from_task_id = msg->from_task_id();
  int32_t to_task_id = msg->to_task_id();
  int32_t src_machine_id = id_map->machine_id_from_task_id(from_task_id);
  int32_t dst_machine_id = id_map->machine_id_from_task_id(to_task_id);
  if (src_machine_id == dst_machine_id) {
    int32_t thread_local_id = id_map->thread_local_id_from_task_id(to_task_id);
    SendIntraNodeMessage(thread_local_id, msg);
  } else {
    SendInterNodeMessage(src_machine_id, dst_machine_id, msg);
  }
}

void CommBus::SendIntraNodeMessage(int32_t dst_thread_local_id,
  MsgPtr msg) {
    queues_[dst_thread_local_id]->Push(msg);
}

void CommBus::SendInterNodeMessage(int32_t src_machine_id,
  int32_t dst_machine_id, MsgPtr msg) {
  NetworkMessage net_msg;
  net_msg.type = NetworkMessageType::MSG_TYPE_REQUEST_ACK;
  net_msg.src_machine_id = src_machine_id;
  net_msg.dst_machine_id = dst_machine_id;
  net_msg.event_msg = *msg;
  network_->Send(net_msg);
}

MessageQueue CommBus::GetQueue(int32_t thread_local_id) const {
  CHECK_GE(thread_local_id, 0);
  CHECK_LT(thread_local_id, static_cast<int32_t>(queues_.size()));
  return queues_[thread_local_id];
}

}