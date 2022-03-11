#include <memory>
#include "oneflow/core/framework/multi_client_session_context.h"
#include "oneflow/core/job/env_global_objects_scope.h"

#ifndef ONEFLOW_API_CPP_ENV_IMPL_H_
#define ONEFLOW_API_CPP_ENV_IMPL_H_

namespace oneflow_api {
namespace of = oneflow;
class OneFlowEnv {
  public:
  OF_DISALLOW_COPY(OneFlowEnv);
  OneFlowEnv();
  ~OneFlowEnv();
  std::shared_ptr<of::MultiClientSessionContext> GetSessionCtx() {
	  return session_ctx_;
  }
private:
  std::shared_ptr<of::EnvGlobalObjectsScope> env_ctx_;
  std::shared_ptr<of::MultiClientSessionContext> session_ctx_;
};
}  // namespace oneflow_api

#endif  // ONEFLOW_API_CPP_ENV_IMPL_H_