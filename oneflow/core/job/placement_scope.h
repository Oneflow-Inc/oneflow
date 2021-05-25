#ifndef ONEFLOW_CORE_JOB_PLACEMENT_SCOPE_H_
#define ONEFLOW_CORE_JOB_PLACEMENT_SCOPE_H_

#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

class OperatorConf;

class PlacementScope final {
 public:
  PlacementScope(
      Symbol<ParallelDesc> device_parallel_desc, Symbol<ParallelDesc> host_parallel_desc)
    : device_parallel_desc_(device_parallel_desc), host_parallel_desc_(host_parallel_desc) {}

  size_t hash_value() const {
    const auto& hash_functor = std::hash<Symbol<ParallelDesc>>();
    return hash_functor(device_parallel_desc_) ^ hash_functor(host_parallel_desc_);
  }

  bool operator==(const PlacementScope& other) const {
    return this->device_parallel_desc_ == other.device_parallel_desc_
      && this->host_parallel_desc_ == other.host_parallel_desc_;
  }

  Symbol<ParallelDesc> device_parallel_desc() const { return device_parallel_desc_; }
  Symbol<ParallelDesc> host_parallel_desc() const { return host_parallel_desc_; }

  Maybe<Symbol<ParallelDesc>> GetParallelDesc(const OperatorConf& op_conf) const;

 private:
  Symbol<ParallelDesc> device_parallel_desc_;
  Symbol<ParallelDesc> host_parallel_desc_;
};

}

namespace std {

template<>
struct hash<oneflow::PlacementScope> final {
  size_t operator()(const oneflow::PlacementScope& val) const { return val.hash_value(); }
};

}

#endif  // ONEFLOW_CORE_JOB_PLACEMENT_SCOPE_H_
