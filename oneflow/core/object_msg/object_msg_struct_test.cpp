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
#include "oneflow/core/object_msg/object_msg_struct.h"
#include "oneflow/core/common/callback.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

namespace {

TEST(OBJECT_MSG_DEFINE_STRUCT, basic) {
  bool flag = false;
  auto foo = ObjectMsgPtr<CallbackMsg>::New();
  *foo->mut_callback() = [&flag]() { flag = true; };
  foo->callback()();
  ASSERT_TRUE(flag);
}

}  // namespace

}  // namespace test

}  // namespace oneflow
