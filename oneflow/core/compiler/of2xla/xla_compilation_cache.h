#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_COMPILATION_CACHE_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_COMPILATION_CACHE_H_

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <functional>

#include "oneflow/core/common/shape.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/compiler/of2xla/xla_compiler.h"

namespace oneflow {
namespace mola {
struct Signature;
}  // namespace mola
}  // namespace oneflow

namespace std {
template <>
struct hash<oneflow::mola::Signature> {
  size_t operator()(const oneflow::mola::Signature &signature) const; 
};
}  // namespace std

namespace oneflow {
namespace mola {

struct Signature {
  // Builder name
  std::string name;
  // Device ordinal
  int device_ordinal;

  // Signature will lose efficacy if the entry shapes changed, then it always
  // leads to recompile the program
  std::vector<Shape> entry_shapes;

  bool operator==(const Signature &other) const;
};

Signature ComputeSignature(const std::string &name, const int device_ordinal,
                           const std::vector<Blob *> &entry_blobs);

class XlaCompilationCache {
 public:
  CompilationResult *GetRecord(const Signature &signature) const;
  
  void Record(const Signature &signature,
              const std::shared_ptr<CompilationResult> &result);

  void Release();

 private:
  // static std::shared_mutex mutex_;
  mutable std::mutex mutex_;
  std::unordered_map<Signature, std::shared_ptr<CompilationResult>> records_;
};

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_COMPILATION_CACHE_H_
