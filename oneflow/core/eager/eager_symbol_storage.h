#ifndef ONEFLOW_CORE_EAGER_EAGER_SYMBOL_STORAGE_H_
#define ONEFLOW_CORE_EAGER_EAGER_SYMBOL_STORAGE_H_

#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

class Scope;
class ScopeProto;

class JobDesc;
class JobConfigProto;

namespace vm {

template<>
struct ConstructArgType4Symbol<JobDesc> final {
  using type = JobConfigProto;
};

template<>
struct ConstructArgType4Symbol<Scope> final {
  using type = ScopeProto;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_EAGER_SYMBOL_STORAGE_H_
