#include "oneflow/core/persistence/normal_persistent_in_streams.h"

namespace oneflow {

int32_t NormalPersistentInStreams::ReadLine(std::string* l) {
  DoRead([&]() { return streams_[cur_stream_id_]->ReadLine(l); });
}

int32_t NormalPersistentInStreams::Read(char* s, size_t n) {
  DoRead([&]() { return streams_[cur_stream_id_]->Read(s, n); });
}

int32_t NormalPersistentInStreams::DoRead(std::function<int32_t()> handler) {
  if (cur_stream_id_ == streams_.size()) return -1;
  if (handler() == -1) {
    ++cur_stream_id_;
    if (cur_stream_id_ == streams_.size()) {
      return -1;
    } else {
      return handler();
    }
  } else {
    return 0;
  }
}

}  // namespace oneflow
