
#include <fmt/core.h>
#include <unordered_map>
#include "oneflow/core/thread/thread_executor_factory.h"
#include "oneflow/core/thread/thread_executor.h"

namespace oneflow {
namespace thread {

namespace {

template<typename T>
ExecutorFactory::ProductType CreateExecutor() {
  return ExecutorFactory::ProductType(std::make_unique<thread::ExecutorBase<T>>(T()));
}

}  // namespace

Maybe<ExecutorFactory::ProductType> ExecutorFactory::Create(ExecutorType type) {
  if (type == ExecutorType::kOf) { return CreateExecutor<thread::OfExecutor>(); }

  const auto format_error_msg = [](const auto& name, const auto& option) {
    return fmt::format("{} is not enabled, you should compile oneflow with "
                       "`-DCPU_THREADING_RUNTIMES={}`",
                       name, option);
  };

  if (type == ExecutorType::kTbb) {
    if (!IsTbbEnabled()) { return Error::RuntimeError() << format_error_msg("OneTBB", "TBB"); }
    return CreateExecutor<thread::TbbExecutor>();
  }
  if (type == ExecutorType::kOmp) {
    if (!IsOmpEnabled()) { return Error::RuntimeError() << format_error_msg("OpenMP", "OMP"); }
    return CreateExecutor<thread::OmpExecutor>();
  }
  return CreateExecutor<thread::SeqExecutor>();
}

Maybe<ExecutorFactory::ProductType> ExecutorFactory::Create(const std::string& type) {
  std::unordered_map<std::string, ExecutorType> types = {
      {"SEQ", ExecutorType::kSeq},
      {"OF", ExecutorType::kOf},
      {"TBB", ExecutorType::kTbb},
      {"SEQ", ExecutorType::kOmp},
  };
  if (types.find(type) == types.end()) {
    return Error::RuntimeError() << fmt::format("Not supportted cpu threading runtime: {}", type);
  }
  return Create(types[type]);
}

}  // namespace thread
}  // namespace oneflow