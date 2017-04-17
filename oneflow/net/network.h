#ifndef THOR_NET_NETWORK_H_
#define THOR_NET_NETWORK_H_

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include "thread/event_message.h"
#include "net/network_topology.h"

namespace oneflow {

struct NetworkMessage;

struct NetworkResult;

struct MemoryDescriptor;

class NetworkMemory;

/*
Network defines interface that thor uses for machine-level communication

Init must be called before any operation

The Network interface provide asynchronous communication with neighbors in 
network topo graph. The Network can send msg to others.
 
Two kinds of communication can be done. One is send Network Message, the other
is write a local memory to a remote machines. For the write call, the memory must
be registerd before writing.

Inquiring sending status and polling incoming message, call

NetworkResult result;
while (true) {
  if (Poll(&result)) {
    switch(result->type) {
      case NET_READ_OK:    // fetch blob ok
      case NET_RECEIVE_OK: // receive new message
      case NET_SEND_OK:    // send control message ok
    }
  }
}
*/

// Not thread-safe
class Network {
public:
  // Init the network environment and connect with each other based on the 
  // network topo. Suppose net_topo is a connected graph
  virtual void Init(int32_t my_rank, const NetworkTopology& net_topo) = 0;

  virtual void Finalize() = 0;

  // Barrier all nodes in network topo
  // Should be called only after there is no message on the fly
  // User should make sure all the request has finished
  virtual void Barrier() = 0;

  virtual NetworkMemory* NewNetworkMemory() = 0;

  // |msg| is owned by the caller and can be released once |Send| returns, even 
  // though the actual transmission of the |msg| content has not occurred.
  virtual bool Send(const NetworkMessage& msg) = 0;

  // We assume a following working procedure:
  // (1) Issue a READ verb to RDMA service by |Read|, which also connects a 
  // time_stamp with the completion event of this READ command;
  // (2) Connect the time_stamp of the last READ request with a kProduced 
  // event_message by |RegisterEventMessage|;
  // (3) Poll the completion event of READ with |Poll|;
  // Ensure that there is exactly one step (2) after each step (1), that is,
  // |Read|->|RegisterEventMessage|->|Read|->|RegisterEventMessage|.
  // The calling sequence
  // |Read|->|Read|->|RegisterEventMessage|->|RegisterEventMessage| is not 
  // allowed.
  // Ensure the step (3) is after step (2). Note that we don't assume the 
  // order of completion event occurrence and the step (2). It is possible that
  // completion event occurs before or after step (2). We just assume step (3) 
  // occurs after step (2). This is naturally guaranteed by the fact that we 
  // only use a single net_thread to process all the network routines at a node.
  virtual void Read(MemoryDescriptor* src, NetworkMemory* dst) = 0;

  virtual void RegisterEventMessage(MsgPtr event_msg) = 0;

  // Poll a result from completion queue if have. Return true if get result, 
  // false otherwise.
  // |*result| is owned by the caller.
  virtual bool Poll(NetworkResult* result) = 0;

  // Wait all on the air message(Send request) finished
  // virtual void WaitAll() = 0;
}; 
Network* GetNdspiRDMAInstance();
}  // namespace oneflow

#endif  // THOR_NET_NETWORK_H_
