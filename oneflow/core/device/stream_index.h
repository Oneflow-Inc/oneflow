#ifndef ONEFLOW_CORE_DEVICE_STREAM_INDEX_H_
#define ONEFLOW_CORE_DEVICE_STREAM_INDEX_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/id_util.h"

namespace oneflow {

using stream_index_t = uint32_t;

class StreamIndexGenerator {
 public:
  virtual stream_index_t GenerateComputeStreamIndex() = 0;
  virtual stream_index_t GenerateH2DStreamIndex() = 0;
  virtual stream_index_t GenerateD2HStreamIndex() = 0;

  virtual bool IsComputeStreamIndex(stream_index_t index) const = 0;
  virtual bool IsH2DStreamIndex(stream_index_t index) const = 0;
  virtual bool IsD2HStreamIndex(stream_index_t index) const = 0;
};

class StreamIndexGeneratorManager final {
 public:
  using generator_key_t = uint32_t;

  StreamIndexGenerator* GetGenerator(ProcessId process_id, DeviceId device_id);

 private:
  static_assert(ProcessId::kBits + DeviceId::kBits <= std::numeric_limits<generator_key_t>::digits,
                "generator_key_t is illegal");

  generator_key_t MakeGeneratorKey(ProcessId process_id, DeviceId device_id) const;

  HashMap<generator_key_t, std::unique_ptr<StreamIndexGenerator>> generators_;
};

StreamIndexGenerator* NewStreamIndexGenerator(DeviceType device_type);

#define REGISTER_STREAM_INDEX_GENERATOR(device_type_v, stream_index_generator_class) \
  REGISTER_CLASS(int, device_type_v, StreamIndexGenerator, stream_index_generator_class)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_STREAM_INDEX_H_
