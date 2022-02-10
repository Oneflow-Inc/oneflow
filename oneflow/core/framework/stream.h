#ifndef ONEFLOW_CORE_FRAMEWORK_STREAM_H_
#define ONEFLOW_CORE_FRAMEWORK_STREAM_H_

#include <hash>
#include "oneflow/core/common/stream_role.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/framework/device.h"

namespace oneflow {

namespace vm {
class MirroredObject;
}
using LocalDepObject = vm::MirroredObject;

class Stream final {
 public:
  Stream(const Stream&) = delete;
  Stream(Stream&&) = delete;
  ~Stream() = default;

  bool operator==(const Stream& that) const {
    return this->device() == that.device() && this->stream_role() == that.stream_role();
  }
  bool operator!=(const Stream& that) const { return !(*this == that); }

  Stream(Symbol<Device> device, StreamRole stream_role);

  static Symbol<Stream> (*New)(Symbol<Device> device, StreamRole stream_role);

  Symbol<Device> device() const { return device_; }
  StreamRole stream_role() const { return stream_role_; }

  LocalDepObject* mut_schedule_local_dep_object() const { return schedule_local_dep_object_; }
  const Optional<LocalDepObject*>& mut_transport_local_dep_object() const {
    return transport_local_dep_object_;
  }

 private:
  Symbol<Device> device_;
  StreamRole stream_role_;

  LocalDepObject* schedule_local_dep_object_;
  Optional<LocalDepObject*> transport_local_dep_object_;
};

}

namespace std {
template<>
struct hash<oneflow::Stream> final {
  size_t operator()(const oneflow::Stream& stream) const {
    return std::hash<oneflow::Device>()(stream.device())
          ^ std::hash<int>()(static_cast<int>(stream.stream_role())); 
  }
};

}  // namespace std
#endif  // ONEFLOW_CORE_FRAMEWORK_STREAM_H_
