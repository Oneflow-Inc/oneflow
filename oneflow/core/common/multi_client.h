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
#ifndef ONEFLOW_CORE_COMMON_MULTICLIENT_H_
#define ONEFLOW_CORE_COMMON_MULTICLIENT_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

inline Optional<bool>* IsMultiClientPtr() { return Global<Optional<bool>, MultiClient>::Get(); }

inline Maybe<bool> IsMultiClient() { return JUST(*Global<Optional<bool>, MultiClient>::Get()); }

inline Maybe<void> SetIsMultiClient(bool is_multi_client) {
  CHECK_NOTNULL_OR_RETURN(IsMultiClientPtr());
  *IsMultiClientPtr() = is_multi_client;
  return Maybe<void>::Ok();
}
}  // namespace oneflow

#endif
