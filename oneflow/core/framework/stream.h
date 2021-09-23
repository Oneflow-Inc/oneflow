#ifndef ONEFLOW_CORE_FRAMEWORK_STREAM_H_
#define ONEFLOW_CORE_FRAMEWORK_STREAM_H_

#include <memory>
#include <string>
#include <unordered_set>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stream_descriptor.h"

namespace oneflow {

class LocalDepObject;

class Stream final {
 public:
  explicit Stream(const Stream&) = default;
  explicit Stream(Stream&&) = default;
  ~Stream() = default;
  Stream& operator=(const Stream&) = delete;

  // new stream by default device.
  static Maybe<Symbol<Stream>> (*NewByDefaultDevice)(const std::string& stream_type_name);

  // new stream by default stream_type_name.
  static Maybe<Symbol<Stream>> (*NewByDefaultName)(Symbol<Device> device);

  static Maybe<Symbol<Stream>> (*New)(const std::string& stream_type_name, Symbol<Device> device);

  const StreamDescriptor& stream_descriptor() const { return *stream_descriptor_; }
  const Symbol<Device>& device() const { return device_; }

  bool operator==(const Stream& other) const {
    return this->stream_descriptor_ == other.stream_descriptor_ && this->device_ == other.device_;
  }

  size_t CalcHashValue() const {
    return std::hash<const StreamDescriptor*>()(stream_descriptor_)
        ^ std::hash<Symbol<Device>>()(device_);
  }

 private:
  Stream(const StreamDescriptor* stream_descriptor, Symbol<Device> device)
      : stream_descriptor_(stream_descriptor), device_(device) {}

  static Maybe<Symbol<Stream>> RawNew(const StreamDescriptor* stream_descriptor, Symbol<Device> device);

  Maybe<void> Init();

  // hash key fields.
  const StreamDescriptor* stream_descriptor_;
  Symbol<Device> device_;

  Optional<LocalDepObject*> transport_local_dep_object_;
  LocalDepObject* schedule_local_dep_object_;
};

}

namespace std {
template<>
struct hash<oneflow::Stream> final {
  size_t operator()(const oneflow::Stream& stream) const { return stream.CalcHashValue(); }
};
}

#endif  // ONEFLOW_CORE_FRAMEWORK_STREAM_H_
