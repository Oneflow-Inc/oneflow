#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/common/str_util.h"
#include <google/protobuf/text_format.h>

namespace oneflow {

TeePersistentLogStream::TeePersistentLogStream(const std::string& path) {
  destinations_.emplace_back(LocalFS(), FLAGS_log_dir);
  branches_.reserve(destinations_.size());
  for (const auto& destination : destinations_) {
    branches_.emplace_back(std::make_unique<PersistentOutStream>(
        destination.mut_file_system(), JoinPath(destination.base_dir(), path)));
  }
}

TeePersistentLogStream::~TeePersistentLogStream() { Flush(); }

std::unique_ptr<TeePersistentLogStream> TeePersistentLogStream::Create(const std::string& path) {
  auto stream_ptr = new TeePersistentLogStream(path);
  return std::unique_ptr<TeePersistentLogStream>(stream_ptr);
}

void TeePersistentLogStream::Flush() {
  for (const auto& branch : branches_) { branch->Flush(); }
};

void TeePersistentLogStream::Write(const char* s, size_t n) {
  for (const auto& branch : branches_) { branch->Write(s, n); }
};

void TeePersistentLogStream::Write(const std::string& str) { this->Write(str.data(), str.size()); }

void TeePersistentLogStream::Write(const PbMessage& proto) {
  std::string output;
  google::protobuf::TextFormat::PrintToString(proto, &output);
  this->Write(output);
}

}  // namespace oneflow
