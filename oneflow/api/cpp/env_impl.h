/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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
  std::shared_ptr<of::MultiClientSessionContext> GetSessionCtx() { return session_ctx_; }

 private:
  std::shared_ptr<of::EnvGlobalObjectsScope> env_ctx_;
  std::shared_ptr<of::MultiClientSessionContext> session_ctx_;
};
}  // namespace oneflow_api

#endif  // ONEFLOW_API_CPP_ENV_IMPL_H_
