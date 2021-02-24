#include "oneflow/core/device/stream_index.h"

namespace oneflow {

StreamIndexGenerator* NewStreamIndexGenerator(DeviceType device_type) {
  return NewObj<int, StreamIndexGenerator>(device_type);
}

StreamIndexGeneratorManager::generator_key_t StreamIndexGeneratorManager::MakeGeneratorKey(
    ProcessId process_id, DeviceId device_id) const {
  return (static_cast<generator_key_t>(process_id) << DeviceId::kBits)
         | static_cast<generator_key_t>(device_id);
}

StreamIndexGenerator* StreamIndexGeneratorManager::GetGenerator(ProcessId process_id,
                                                                DeviceId device_id) {
  generator_key_t key = MakeGeneratorKey(process_id, device_id);
  auto iter = generators_.emplace(key, NewStreamIndexGenerator(device_id.device_type())).first;
  return iter->second.get();
}

}  // namespace oneflow
