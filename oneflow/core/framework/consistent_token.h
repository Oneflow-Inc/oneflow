#ifndef ONEFLOW_CORE_FRAMEWORK_CONSISTENT_TOKEN_H_
#define ONEFLOW_CORE_FRAMEWORK_CONSISTENT_TOKEN_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/type_traits.h"

namespace oneflow {

class ParallelDesc;
class ConsistentToken;

template<>
struct IsScalarType<ConsistentToken> final {
  static const bool value = true;
};

class ConsistentToken final {
 public: 
  ConsistentToken(int32_t major, int32_t minor) : major_(major), minor_(minor) {}
  ConsistentToken(const ConsistentToken&) = default;
  ConsistentToken(ConsistentToken&) = default;
  ~ConsistentToken(ConsistentToken&) = default;

  int32_t major() const { return major_; }
  int32_t minor() const { return minor_; }
  operator int64_t() const { return static_cast<int64_t>(major_) << 32 + minor_; }

  ConsistentToken& operator++() {
    ++minor_;
    return *this;
  }

 private:
  int32_t major_;
  int32_t minor_;
};

static_assert(sizeof(ConsistentToken) == sizeof(int64_t), "");

Maybe<void> InitCurrentConsistentToken(Symbol<ParallelDesc> parallel_desc);
Maybe<ConsistentToken> GetCurrentConsistentToken(Symbol<ParallelDesc> parallel_desc);
Maybe<ConsistentToken> GetAutoIncrementalConsistentToken(Symbol<ParallelDesc> parallel_desc);

}

#endif  // ONEFLOW_CORE_FRAMEWORK_CONSISTENT_TOKEN_H_
