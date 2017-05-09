#ifndef ONEFLOW_RUNTIME_COMM_BUS_H_
#define ONEFLOW_RUNTIME_COMM_BUS_H_
#include <glog/logging.h>
#include <vector>
#include <memory>
#include "context/id_map.h"
#include "runtime/event_message.h"
#include "runtime/mt_queue.h"
#include "network/network.h"
#include "network/network_message.h"

namespace oneflow {
using MessageQueue = std::shared_ptr<MtQueue<MsgPtr>>;

class CommBus {
 public:
    explicit CommBus(int32_t thread_num_each_client);
    ~CommBus();

    void Init();
    void SendMessage(MsgPtr msg);
    MessageQueue GetQueue(int32_t thread_local_id) const;

 private:
    // There is a message queue for each thread on this machine
    std::vector<MessageQueue> queues_;
    int32_t thread_num_each_client_;
    Network *network_;  // For inter-node messaging

    void SendIntraNodeMessage(
      int32_t dst_thread_local_id, MsgPtr msg);

    void SendInterNodeMessage(int32_t src_machine_id, int32_t dst_machine_id,
      MsgPtr msg);

    CommBus(const CommBus& other) = delete;
    CommBus& operator=(const CommBus& other) = delete;
};

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
  net_msg.src_rank = src_machine_id;
  net_msg.dst_rank = dst_machine_id;
  net_msg.event_msg = *msg;
  network_->Send(net_msg);
}

MessageQueue CommBus::GetQueue(int32_t thread_local_id) const {
  CHECK_GE(thread_local_id, 0);
  CHECK_LT(thread_local_id, static_cast<int32_t>(queues_.size()));
  return queues_[thread_local_id];
}

}  // namespace oneflow
#endif  // ONEFLOW_RUNTIME_COMM_BUS_H_
