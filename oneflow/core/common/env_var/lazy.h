#ifndef ONEFLOW_CORE_COMMON_ENV_VAR_LAZY_H_
#define ONEFLOW_CORE_COMMON_ENV_VAR_LAZY_H_

namespace oneflow {

// options: "naive", "rank_per_thread" .
static const std::string kNaiveCompiler;
static const std::string kRankPerThreadCompiler;
DEFINE_THREAD_LOCAL_ENV_STRING(ONEFLOW_LAZY_COMPILER, kNaiveCompiler);

}

#endif  // ONEFLOW_CORE_COMMON_ENV_VAR_LAZY_H_
