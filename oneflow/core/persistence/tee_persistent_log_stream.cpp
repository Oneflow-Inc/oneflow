#include "oneflow/core/persistence/tee_persistent_log_stream.h"

namespace oneflow {

TeePersistentLogStream::TeePersistentLogStream(
    std::vector<std::unique_ptr<PersistentOutStream>>&& branches)
    : LogStream(), branches_(std::move(branches)){};

TeePersistentLogStream::~TeePersistentLogStream() { Flush(); }

void TeePersistentLogStream::Flush() {
  for (const auto& branch : branches_) { branch->Flush(); }
};

LogStream& TeePersistentLogStream::Write(const char* s, size_t n) {
  for (const auto& branch : branches_) { branch->Write(s, n); }
  return *this;
};

}  // namespace oneflow
