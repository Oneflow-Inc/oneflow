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
#include "oneflow/core/intrusive/intrusive.h"
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

class StreamTypeIdItem final : public intrusive::Base {
 public:
  // Getters
  const StreamTypeId& stream_type_id() const { return stream_type_id_.key().Get(); }
  // Setters
  StreamTypeId* mut_stream_type_id() { return stream_type_id_.mut_key()->Mutable(); }

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  StreamTypeIdItem() : intrusive_ref_(), stream_type_id_() {}
  intrusive::Ref intrusive_ref_;

 public:
  // skiplist hooks
  intrusive::SkipListHook<FlatMsg<StreamTypeId>, 20> stream_type_id_;
};
using StreamTypeIdSet = intrusive::SkipList<INTRUSIVE_FIELD(StreamTypeIdItem, stream_type_id_)>;

TEST(StreamTypeId, map_key) {
  auto stream_type_id0 = intrusive::make_shared<StreamTypeIdItem>();
  stream_type_id0->mut_stream_type_id()->__Init__(LookupStreamType4TypeIndex<ControlStreamType>(),
                                                  InterpretType::kCompute);
  auto stream_type_id1 = intrusive::make_shared<StreamTypeIdItem>();
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
