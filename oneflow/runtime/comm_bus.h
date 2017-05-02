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

}  // namespace oneflow
#endif  // ONEFLOW_RUNTIME_COMM_BUS_H_
