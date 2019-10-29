#ifndef ONEFLOW_XRT_XLA_COMPILATION_CACHE_H_
#define ONEFLOW_XRT_XLA_COMPILATION_CACHE_H_

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "oneflow/xrt/executable.h"
#include "oneflow/xrt/parameter.h"
#include "oneflow/xrt/shape.h"
#include "oneflow/xrt/utility/stl.h"

namespace oneflow {
namespace xrt {

struct Signature {
  // Builder name
  std::string name;
  // Device ordinal
  int device_ordinal;
  // It will Lose efficacy if the entry shapes has been changed.
  std::vector<XrtShape> entry_shapes;
};

bool operator==(const Signature &lhs, const Signature &rhs);

struct SignatureHash {
  size_t operator()(const Signature &signature) const;
};

Signature ComputeSignature(const std::string &name, const int device_ordinal,
                           const std::vector<xrt::Parameter> &entry_params);

class CompilationCache {
 public:
  Executable *GetRecord(const Signature &signature) const;

  void Record(const Signature &signature,
              const std::shared_ptr<Executable> &result);

  void Release();

 private:
  // static std::shared_mutex mutex_;
  mutable std::mutex mutex_;
  util::Map<Signature, std::shared_ptr<Executable>, SignatureHash> records_;
};

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_COMPILATION_CACHE_H_
