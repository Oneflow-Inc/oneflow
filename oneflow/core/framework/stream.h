#ifndef ONEFLOW_CORE_FRAMEWORK_STREAM_H_
#define ONEFLOW_CORE_FRAMEWORK_STREAM_H_

#include <hash>
#include "oneflow/core/common/stream_role.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/device.h"

namespace oneflow {

class Stream final {
 public:
  Stream(const Stream&) = delete;
  Stream(Stream&&) = delete;
  ~Stream() = default;

  bool operator==(const Stream& that) const {
    return this->device() == that.device() && this->stream_role() == that.stream_role();
  }
  bool operator!=(const Stream& that) const { return !(*this == that); }

  Stream(Symbol<Device> device, StreamRole stream_role)
      : device_(device), stream_role_(stream_role) {}

  static Symbol<Stream> (*New)(Symbol<Device> device, StreamRole stream_role);

  Symbol<Device> device() const { return device_; }
  StreamRole stream_role() const { return stream_role_; }

 private:

  Symbol<Device> device_;
  StreamRole stream_role_;
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
