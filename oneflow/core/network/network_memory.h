#ifndef ONEFLOW_CORE_NETWORK_NETWORK_MEMORY_H_
#define ONEFLOW_CORE_NETWORK_NETWORK_MEMORY_H_
#include <cstdint>

namespace oneflow {

struct MemoryDescriptor {
  // To avoid involve windows header for the following typedef
  // typedef unsigned long long UINT64
  // typedef unsigned int       UINT32
  uint64_t machine_id;
  uint64_t address;
  uint32_t remote_token;
};

class NetworkMemory {
 public:
  virtual ~NetworkMemory() = default;

  // Id: id of this memory, -1 means the id is trivial
  //
  // For Write used Memory, the id must be assigned
  //
  // We need the id in write semantic to identify which memory is written
  // and when we polled one write finish event from completion queue, we can
  // know which write request it is.
  //
  // For Send used Memory, the id means nothing
  //
  // User will never care about the memory send message uses, it is handled by
  // network layer
  //
  // Memory can be used as src memory in Write only after its previous Write
  // requests are all finished, since the id will be re
  void Reset(void* memory, size_t size, int64_t id = -1) {
    memory_ = memory;
    size_ = size;
    id_ = id;
  }

  int64_t& mutable_id() { return id_; }
  bool registered() const { return registered_; }
  const MemoryDescriptor& memory_discriptor() const { return descriptor_; }

  // Register memory for network transportation
  virtual void Register() = 0;
  // Unregister memory after finishing network transportation
  virtual void Unregister() = 0;
  virtual void* sge() = 0;  // XXX(shiyuan) delete const

 protected:
  MemoryDescriptor descriptor_;
  int64_t id_;           // memory id, set as register id
  void* memory_;         // Not owned
  size_t size_;
  bool registered_ = false;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_NETWORK_MEMORY_H_
