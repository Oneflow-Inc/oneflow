#ifndef ONEFLOW_CORE_NETWORK_NETWORK_MEMORY_H_
#define ONEFLOW_CORE_NETWORK_NETWORK_MEMORY_H_

namespace oneflow {

struct MemoryDescriptor {
  int64_t machine_id;
  uint64_t address;
  uint32_t remote_token;
};

class NetworkMemory {
 public:
  NetworkMemory() = default;
  virtual ~NetworkMemory() = default;

  void Reset(void* memory, size_t size) {
    memory_ = memory;
    size_ = size;
  }

  bool registered() const { return registered_; }
  const MemoryDescriptor& memory_discriptor() const { return descriptor_; }

  // Register memory for network transportation
  virtual void Register() = 0;
  // Unregister memory after finishing network transportation
  virtual void Unregister() = 0;
  virtual void* sge() = 0;

 protected:
  MemoryDescriptor descriptor_;
  void* memory_;  // Not owned
  size_t size_;
  bool registered_ = false;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_NETWORK_MEMORY_H_
