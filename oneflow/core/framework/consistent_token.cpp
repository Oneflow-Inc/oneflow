#include <memory>
#include "oneflow/core/framework/consistent_token.h"
#include "oneflow/core/thread/thread_unique_tag.h"

namespace oneflow {

namespace {

std::unique_ptr<ConsistentToken>& MutThreadLocalConsistentId(Symbol<ParallelDesc> parallel_desc) {
  static thread_local HashMap<Symbol<ParallelDesc>, std::unique_ptr<ConsistentToken>> parallel_desc2consistent_id;
  return parallel_desc2consistent_id[parallel_desc];
}

}

Maybe<void> InitCurrentConsistentToken(Symbol<ParallelDesc> parallel_desc) {
  auto& token = MutThreadLocalConsistentId(parallel_desc);
  CHECK_OR_RETURN(!static_cast<bool>(token));
  if (!parallel_desc->containing_current_rank()) { return Maybe<void>::Ok(); }
  JUST(GetThisThreadUniqueTag());
  TODO_THEN_RETURN();
  return Maybe<void>::Ok();
}

Maybe<ConsistentToken> GetCurrentConsistentToken(Symbol<ParallelDesc> parallel_desc) {
  JUST(GetThisThreadUniqueTag());
  if (!parallel_desc->containing_current_rank()) { return ConsistentToken(0, 0); }
  const auto& token = MutThreadLocalConsistentId(parallel_desc);
  CHECK_OR_RETURN(static_cast<bool>(token));
  return *token;
}

Maybe<ConsistentToken> GetAutoIncrementalConsistentToken(Symbol<ParallelDesc> parallel_desc) {
  JUST(GetThisThreadUniqueTag());
  CHECK_OR_RETURN(parallel_desc->containing_current_rank());
  const auto& token = MutThreadLocalConsistentId(parallel_desc);
  CHECK_OR_RETURN(static_cast<bool>(token));
  return ++*token;
}

}
