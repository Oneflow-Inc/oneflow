#include "oneflow/core/persistence/tee_persistent_out_stream.h"

namespace oneflow {

TeePersistentOutStream::TeePersistentOutStream(
    std::vector<std::unique_ptr<PersistentOutStream>> branches)
    : branches_(std::move(branches)){};

TeePersistentOutStream::~TeePersistentOutStream() { Flush(); }

void TeePersistentOutStream::Flush() {
  for (auto& branch : branches_) { branch->Flush(); }
};

OutStream& TeePersistentOutStream::Write(const char* s, size_t n) {
  for (auto& branch : branches_) { branch->Write(s, n); }
  return *this;
};

}  // namespace oneflow
