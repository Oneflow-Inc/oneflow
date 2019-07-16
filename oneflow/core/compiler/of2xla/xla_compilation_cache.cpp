#include "oneflow/core/compiler/of2xla/xla_compilation_cache.h"

namespace oneflow {
namespace mola {

bool Signature::operator==(const Signature &other) const {
  return this->name == other.name &&
         this->device_ordinal == other.device_ordinal &&
         this->entry_shapes == other.entry_shapes;
}

Signature ComputeSignature(const std::string &name,
                           const int device_ordinal,
                           const std::vector<Blob *> &entry_blobs) {
  Signature signature;
  signature.name = name;
  signature.device_ordinal = device_ordinal;
  signature.entry_shapes.resize(entry_blobs.size());
  for (int i = 0; i < entry_blobs.size(); ++i) {
    signature.entry_shapes[i] = entry_blobs[i]->shape();
  }

  return std::move(signature);
}

CompilationResult *XlaCompilationCache::GetRecord(
    const Signature &signature) const {
  CompilationResult *record = nullptr;
  // std::shared_lock<std::shared_mutex> lock(mutex_);
  std::lock_guard<std::mutex> lock(mutex_);
  const auto &it = records_.find(signature);
  if (it != records_.end()) {
    record = it->second.get();
  }
  return record;
}

void XlaCompilationCache::Record(
    const Signature &signature,
    const std::shared_ptr<CompilationResult> &result) {
  // std::unique_lock<std::shared_mutex> lock(mutex_);
  std::lock_guard<std::mutex> lock(mutex_);
  records_.emplace(signature, result);
}

void XlaCompilationCache::Release() {
  std::unordered_map<Signature, std::shared_ptr<CompilationResult>>
      empty_records;
  records_.swap(empty_records);
}

}  // namespace mola
}  // namespace oneflow

namespace std {
size_t hash<oneflow::mola::Signature>::operator()(
    const oneflow::mola::Signature &signature) const {
  size_t hash_val = std::hash<std::string>()(signature.name) ^
                    std::hash<int>()(signature.device_ordinal);
  for (const auto &shape : signature.entry_shapes) {
    hash_val ^= std::hash<std::string>()(shape.ToString());
  }
  return hash_val;
}
}  // namespace std
