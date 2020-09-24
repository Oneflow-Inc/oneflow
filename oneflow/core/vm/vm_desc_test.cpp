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
// include sstream first to avoid some compiling error
// caused by the following trick
// reference: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=65899
#include <sstream>
#define private public
#include "oneflow/core/vm/vm_desc.msg.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/object_msg/object_msg_reflection.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/test_util.h"
#include "oneflow/core/common/cached_object_msg_allocator.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {
namespace vm {

namespace test {

TEST(VmDesc, ToDot) {
  std::string dot_str = ObjectMsgListReflection<VmDesc>().ToDot("VmDesc");
  // std::cout << std::endl;
  // std::cout << dot_str << std::endl;
  // std::cout << std::endl;
}

}  // namespace test

}  // namespace vm
}  // namespace oneflow
