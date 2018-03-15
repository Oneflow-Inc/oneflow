#include "oneflow/core/record/record.h"

namespace oneflow {

template<>
bool ReadRecord(PersistentInStream* in_stream, OFRecord* record) {
  size_t record_size;
  if (in_stream->Read(reinterpret_cast<char*>(&record_size), sizeof(size_t))) {
    std::unique_ptr<std::vector<char>> buffer =
        of_make_unique<std::vector<char>>(record_size);
    CHECK_EQ(in_stream->Read(buffer->data(), record_size), 0);
    record->ParseFromArray(buffer->data(), record_size);
    return true;
  } else {
    return false;
  }
}

}  // namespace oneflow
