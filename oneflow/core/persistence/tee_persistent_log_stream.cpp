#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/common/str_util.h"
#include <google/protobuf/text_format.h>

namespace oneflow {

TeePersistentLogStream::TeePersistentLogStream(const std::string& path) {
  destinations_.emplace_back(LocalFS(), LogDir());
  branches_.reserve(destinations_.size());
  for (const auto& destination : destinations_) {
    branches_.emplace_back(std::make_unique<PersistentOutStream>(
        destination.mut_file_system(), JoinPath(destination.base_dir(), path)));
  }
}

TeePersistentLogStream::~TeePersistentLogStream() { Flush(); }

void TeePersistentLogStream::Flush() {
  for (const auto& branch : branches_) { branch->Flush(); }
};

void TeePersistentLogStream::Write(const char* s, size_t n) {
  for (const auto& branch : branches_) { branch->Write(s, n); }
};

void SaveProtoAsTextFile(const PbMessage& proto, const std::string& path) {
  std::string output;
  google::protobuf::TextFormat::PrintToString(proto, &output);
  TeePersistentLogStream log_stream(path);
  log_stream << output;
}

}  // namespace oneflow
