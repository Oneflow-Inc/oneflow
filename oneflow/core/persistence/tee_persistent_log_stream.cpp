#include "oneflow/core/persistence/tee_persistent_log_stream.h"

namespace oneflow {

TeePersistentLogStream::TeePersistentLogStream(
    std::vector<std::unique_ptr<PersistentOutStream>>&& branches)
    : branches_(std::move(branches)){};

TeePersistentLogStream::~TeePersistentLogStream() { Flush(); }

void TeePersistentLogStream::Flush() {
  for (auto& branch : branches_) { branch->Flush(); }
};

LogStream& TeePersistentLogStream::Write(const char* s, size_t n) {
  for (auto& branch : branches_) { branch->Write(s, n); }
  return *this;
};

}  // namespace oneflow
