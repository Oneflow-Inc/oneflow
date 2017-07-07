#ifndef ONEFLOW_CORE_NETWORK_NETWORK_H_
#define ONEFLOW_CORE_NETWORK_NETWORK_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/network/network_message.h"
#include "oneflow/core/network/network_memory.h"
#include "oneflow/core/network/network_topology.h"

namespace oneflow {

// Not thread-safe
class Network {
 public:
  Network() = default;
  virtual ~Network() = default;
  // Init the network environment and connect with each other based on the
  // network topo. Suppose net_topo is a connected graph
  virtual void Init(int64_t my_machine_id, const NetworkTopology& net_topo) = 0;

  virtual void Finalize() = 0;

  // Barrier all nodes in network topo
  // Should be called only after there is no message on the fly
  // User should make sure all the request has finished
  virtual void Barrier() = 0;

  virtual NetworkMemory* NewNetworkMemory() = 0;

  // |msg| is owned by the caller and can be released once |Send| returns, even
  // though the actual transmission of the |msg| content has not occurred.
  virtual void Send(const NetworkMessage& msg) = 0;

  // We assume a following working procedure:
  // (1) Issue a READ verb to RDMA service by |Read|, which also connects a
  // time_stamp with the completion event of this READ command;
  // (2) Connect the time_stamp of the last READ request with a kProduced
  // event_message by |EnrollActorMessage|;
  // (3) Poll the completion event of READ with |Poll|;
  // Ensure that there is exactly one step (2) after each step (1), that is,
  // |Read|->|EnrollActorMessage|->|Read|->|EnrollActorMessage|.
  // The calling sequence
  // |Read|->|Read|->|EnrollActorMessage|->|EnrollActorMessage| is not
  // allowed.
  // Ensure the step (3) is after step (2). Note that we don't assume the
  // order of completion event occurrence and the step (2). It is possible that
  // completion event occurs before or after step (2). We just assume step (3)
  // occurs after step (2). This is naturally guaranteed by the fact that we
  // only use a single net_thread to process all the network routines at a node.
  virtual void Read(const MemoryDescriptor& src, NetworkMemory* dst) = 0;

  virtual void EnrollActorMessage(MsgPtr actor_msg) = 0;

  // Poll a result from completion queue if have. Return true if get result,
  // false otherwise.
  // |*result| is owned by the caller.
  virtual bool Poll(NetworkResult* result) = 0;

  // Wait all on the air message(Send request) finished
  // virtual void WaitAll() = 0;
};

Network* GetRdmaInstance();
}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_NETWORK_H_
