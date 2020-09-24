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
#include "oneflow/core/object_msg/object_msg.h"
#include "oneflow/core/vm/stream_type_id.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

namespace test {

TEST(StreamTypeId, logical_compare) {
  StreamTypeId stream_type_id0;
  stream_type_id0.__Init__(LookupStreamType4TypeIndex<ControlStreamType>(),
                           InterpretType::kCompute);
  StreamTypeId stream_type_id1;
  stream_type_id1.__Init__(LookupStreamType4TypeIndex<ControlStreamType>(),
                           InterpretType::kCompute);

  ASSERT_EQ(&stream_type_id0.stream_type(), &stream_type_id1.stream_type());
  ASSERT_EQ(stream_type_id0.interpret_type(), stream_type_id1.interpret_type());
  ASSERT_EQ(std::memcmp(&stream_type_id0, &stream_type_id1, sizeof(StreamTypeId)), 0);
  ASSERT_EQ(stream_type_id0 == stream_type_id1, true);
  ASSERT_EQ(stream_type_id0 != stream_type_id1, false);
  ASSERT_EQ(stream_type_id0 <= stream_type_id1, true);
  ASSERT_EQ(stream_type_id0 < stream_type_id1, false);
  ASSERT_EQ(stream_type_id0 >= stream_type_id1, true);
  ASSERT_EQ(stream_type_id0 > stream_type_id1, false);
  LookupInferStreamTypeId(stream_type_id0);
}

// clang-format off
OBJECT_MSG_BEGIN(StreamTypeIdItem);
  OBJECT_MSG_DEFINE_MAP_KEY(StreamTypeId, stream_type_id);
OBJECT_MSG_END(StreamTypeIdItem);
// clang-format on
using StreamTypeIdSet = OBJECT_MSG_MAP(StreamTypeIdItem, stream_type_id);

TEST(StreamTypeId, map_key) {
  auto stream_type_id0 = ObjectMsgPtr<StreamTypeIdItem>::New();
  stream_type_id0->mut_stream_type_id()->__Init__(LookupStreamType4TypeIndex<ControlStreamType>(),
                                                  InterpretType::kCompute);
  auto stream_type_id1 = ObjectMsgPtr<StreamTypeIdItem>::New();
  stream_type_id1->mut_stream_type_id()->__Init__(LookupStreamType4TypeIndex<ControlStreamType>(),
                                                  InterpretType::kCompute);
  StreamTypeIdSet stream_type_id_set;
  ASSERT_TRUE(stream_type_id_set.Insert(stream_type_id0.Mutable()).second);
  ASSERT_TRUE(!stream_type_id_set.Insert(stream_type_id1.Mutable()).second);
  ASSERT_EQ(stream_type_id_set.size(), 1);
}

}  // namespace test

}  // namespace vm
}  // namespace oneflow
