#ifndef ONEFLOW_CORE_MEMORY_MEMORY_DEVICE_H_
#define ONEFLOW_CORE_MEMORY_MEMORY_DEVICE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/memory/memory_case.pb.h"

namespace oneflow {

enum MemoryType {
  kHostMemory,
  kDeviceMemory,
};

class MemoryDeviceMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemoryDeviceMgr);
  ~MemoryDeviceMgr() = default;
  OF_SINGLETON(MemoryDeviceMgr);

  //  Getters
  size_t host_mem_size() const { return host_mem_size_; }
  size_t dev_mem_size() const { return dev_mem_size_; }

 private:
  MemoryDeviceMgr();
  size_t GetThisMachineHostMemSize() const;
  size_t GetThisMachineDeviceMemSize() const;

  size_t host_mem_size_;
  size_t dev_mem_size_;
};

class MemoryDevice final {
 public:
  explicit MemoryDevice(uint32_t machine_id, const MemoryCase& mem_case);
  MemoryDevice(const MemoryDevice& mem_device) = default;
  size_t Size() const;

  uint32_t machine_id() const { return machine_id_; }
  const MemoryType& memory_type() const { return memory_type_; }
  uint32_t device_id() const { return device_id_; }

  bool operator==(const MemoryDevice& mem_dev) const {
    return mem_dev.memory_type() == memory_type()
           && mem_dev.device_id() == device_id()
           && mem_dev.machine_id() == machine_id();
  }

 private:
  uint32_t machine_id_;
  MemoryType memory_type_;
  uint16_t device_id_;
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::MemoryDevice> final {
  hash() = default;
  std::size_t operator()(const oneflow::MemoryDevice& mem_device) const {
    return (static_cast<int16_t>(mem_device.memory_type()) * UINT16_MAX
            + mem_device.device_id())
               * UINT32_MAX
           + mem_device.machine_id();
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_MEMORY_MEMORY_DEVICE_H_
