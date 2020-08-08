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
#ifndef ONEFLOW_CORE_COMMON_CALLBACK_MSG_H_
#define ONEFLOW_CORE_COMMON_CALLBACK_MSG_H_

#include <functional>
#include "oneflow/core/object_msg/object_msg.h"

namespace oneflow {

// clang-format off
OBJECT_MSG_BEGIN(CallbackMsg);
  // methods
  OF_PUBLIC void __Init__() {}
  OF_PUBLIC void __Init__(const std::function<void()>& callback) { *mut_callback() = callback; }

  // fields
  OBJECT_MSG_DEFINE_STRUCT(std::function<void()>, callback);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(callback_link);
OBJECT_MSG_END(CallbackMsg);
// clang-format on

using CallbackMsgListPtr = OBJECT_MSG_LIST_PTR(CallbackMsg, callback_link);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_CALLBACK_MSG_H_
