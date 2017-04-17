#ifndef _COMM_BUS_H_
#define _COMM_BUS_H_
#include <glog/logging.h>
#include <vector>
#include <memory>
#include "thread/event_message.h"
#include "thread/mt_queue.h"
#include "net/network.h"
#include "net/network_message.h"

namespace oneflow {
using MessageQueue = std::shared_ptr<MtQueue<MsgPtr>>;

template <typename Dtype>
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

template <typename Dtype>
CommBus<Dtype>::CommBus(int32_t thread_num_each_client)
  : thread_num_each_client_(thread_num_each_client) {
}

template <typename Dtype>
CommBus<Dtype>::~CommBus() {
}

template <typename Dtype>
void CommBus<Dtype>::Init() {
  network_ = GetNdspiRDMAInstance();  // Used for inter-node messaging
  for (int32_t i = 0; i < thread_num_each_client_; ++i) {
    queues_.emplace_back(new MtQueue<MsgPtr>());
  }
}

template <typename Dtype>
void CommBus<Dtype>::SendMessage(MsgPtr msg) {
  // Get destination machine and thread's id from msg
  /*
  auto&& id_map = oneflow::TheOne<Dtype>::id_map();
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
  */
}

template <typename Dtype>
void CommBus<Dtype>::SendIntraNodeMessage(int32_t dst_thread_local_id,
  MsgPtr msg) {
    queues_[dst_thread_local_id]->Push(msg);
}

template <typename Dtype>
void CommBus<Dtype>::SendInterNodeMessage(int32_t src_machine_id,
  int32_t dst_machine_id, MsgPtr msg) {
  NetworkMessage net_msg;
  net_msg.type = NetworkMessageType::MSG_TYPE_REQUEST_ACK;
  net_msg.src_rank = src_machine_id;
  net_msg.dst_rank = dst_machine_id;
  net_msg.event_msg = *msg;
  network_->Send(net_msg);
}

template <typename Dtype>
MessageQueue CommBus<Dtype>::GetQueue(int32_t thread_local_id) const {
  CHECK_GE(thread_local_id, 0);
  CHECK_LT(thread_local_id, static_cast<int32_t>(queues_.size()));
  return queues_[thread_local_id];
}

}  // namespace oneflow
#endif  // _COMM_BUS_H_
