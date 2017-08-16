#ifndef ONEFLOW_CORE_NETWORK_NETWORK_H_
#define ONEFLOW_CORE_NETWORK_NETWORK_H_

#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/network/network_memory.h"
#include "oneflow/core/network/network_message.h"
#include "oneflow/core/network/network_topology.h"

namespace oneflow {

class Network {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Network);
  virtual ~Network() = default;

  // Init the network environment and connect with each other based on the
  // network topo. Suppose net_topo is a connected graph
  virtual void Init(int64_t my_machine_id, const NetworkTopology& net_topo) = 0;

  virtual NetworkMemory* RegisterMemory(void* dptr, size_t len) = 0;

  // |msg| is owned by the caller and can be released once |Send| returns, even
  // though the actual transmission of the |msg| content has not occurred.
  virtual void SendMsg(const NetworkMessage& msg) = 0;
  virtual void SetCallbackForReceivedActorMsg(
      std::function<void()> callback) = 0;

  virtual void Read(
      const MemoryDescriptor& remote_memory_descriptor,
      NetworkMemory* local_memory, std::function<void()> callback) = 0;

  // Poll a result from completion queue if have. Return true if get result,
  // false otherwise.
  // |*result| is owned by the caller.
  virtual bool Poll(NetworkResult* result) = 0;

  // Barrier all nodes in network topo
  // Should be called only after there is no message on the fly
  // User should make sure all the request has finished
  virtual void Barrier() = 0;

 protected:
  Network() = default;
};

Network* GetRdmaInstance();

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_NETWORK_H_
