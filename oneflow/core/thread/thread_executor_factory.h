#ifndef ONEFLOW_CORE_THREAD_THREAD_EXECUTOR_FACTORY_H_
#define ONEFLOW_CORE_THREAD_THREAD_EXECUTOR_FACTORY_H_

#include "oneflow/maybe/variant.h"
#include "oneflow/core/thread/thread_executor.h"

namespace oneflow {
namespace thread {

constexpr bool IsTbbEnabled() {
#ifdef WITH_TBB
  return true;
#else
  return false;
#endif
}

constexpr bool IsOmpEnabled() {
#ifdef WITH_Omp
  return true;
#else
  return false;
#endif
}

enum class ExecutorType {
  kSeq,
  kOf,
  kTbb,
  kOmp,
};

class ExecutorFactory {
 public:
  static Maybe<thread::ExecutorBase> Create(ExecutorType type);
  static Maybe<thread::ExecutorBase> Create(const std::string& type);
};

}  // namespace thread
}  // namespace oneflow
#endif  // ONEFLOW_CORE_THREAD_THREAD_EXECUTOR_FACTORY_H_
