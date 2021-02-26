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
#include <set>
#include <cstdlib>
#include "oneflow/core/common/cfg_reflection_test.cfg.h"
#include "oneflow/core/common/cfg_reflection_test.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/cfg.h"

namespace oneflow {
namespace test {

TEST(std_hash_cfg, required_int) {
  std::srand(0);
  std::set<std::size_t> hash_values;
  static const int kNumMax = 10000;
  const auto& hash = std::hash<cfg::ReflectionTestFoo>();
  for (int i = 0; i < kNumMax; ++i) {
    cfg::ReflectionTestFoo foo;
    foo.set_required_int32(std::rand());
    hash_values.insert(hash(foo));
  }
  ASSERT_GT(hash_values.size(), int(kNumMax / 1.2));
}

TEST(std_hash_cfg, optional_string) {
  std::srand(0);
  std::set<std::size_t> hash_values;
  static const int kNumMax = 10000;
  const auto& hash = std::hash<cfg::ReflectionTestFoo>();
  for (int i = 0; i < kNumMax; ++i) {
    cfg::ReflectionTestFoo foo;
    foo.set_optional_string(std::to_string(std::rand()));
    hash_values.insert(hash(foo));
  }
  ASSERT_GT(hash_values.size(), int(kNumMax / 1.2));
}

TEST(std_hash_cfg, required_msg) {
  std::srand(0);
  std::set<std::size_t> hash_values;
  static const int kNumMax = 10000;
  const auto& hash = std::hash<cfg::ReflectionTestBar>();
  for (int i = 0; i < kNumMax; ++i) {
    cfg::ReflectionTestBar bar;
    cfg::ReflectionTestFoo* foo = bar.mutable_required_foo();
    foo->set_required_int32(std::rand());
    hash_values.insert(hash(bar));
  }
  ASSERT_GT(hash_values.size(), int(kNumMax / 1.2));
}

TEST(std_hash_cfg, repeated_msg) {
  std::srand(0);
  std::set<std::size_t> hash_values;
  static const int kNumMax = 10000;
  const auto& hash = std::hash<cfg::ReflectionTestBar>();
  for (int i = 0; i < kNumMax; ++i) {
    cfg::ReflectionTestBar bar;
    for (int j = 0; j < 10; ++j) {
      bar.mutable_repeated_foo()->Add()->set_required_int32(std::rand());
    }
    hash_values.insert(hash(bar));
  }
  ASSERT_GT(hash_values.size(), int(kNumMax / 1.2));
}

TEST(std_hash_cfg, map_msg) {
  std::srand(0);
  std::set<std::size_t> hash_values;
  static const int kNumMax = 10000;
  const auto& hash = std::hash<cfg::ReflectionTestBar>();
  for (int i = 0; i < kNumMax; ++i) {
    cfg::ReflectionTestBar bar;
    for (int j = 0; j < 10; ++j) {
      (*bar.mutable_map_foo())[i].set_optional_string(std::to_string(std::rand()));
    }
    hash_values.insert(hash(bar));
  }
  ASSERT_GT(hash_values.size(), int(kNumMax / 1.2));
}

TEST(CfgReflection, FieldDefined_required_not_set) {
  {
    ReflectionTestFoo foo;
    cfg::ReflectionTestFoo cfg_foo(foo);
    ASSERT_EQ(FieldDefinedInPbMessage(foo, "required_int32"),
              cfg_foo.FieldDefined4FieldName("required_int32"));
  }
  {
    ReflectionTestBar bar;
    cfg::ReflectionTestBar cfg_bar(bar);
    ASSERT_EQ(FieldDefinedInPbMessage(bar, "required_foo"),
              cfg_bar.FieldDefined4FieldName("required_foo"));
  }
}

TEST(CfgReflection, FieldDefined_required_set) {
  {
    ReflectionTestFoo foo;
    foo.set_required_int32(0);
    cfg::ReflectionTestFoo cfg_foo(foo);
    ASSERT_EQ(FieldDefinedInPbMessage(foo, "required_int32"),
              cfg_foo.FieldDefined4FieldName("required_int32"));
  }
  {
    ReflectionTestBar bar;
    bar.mutable_required_foo();
    cfg::ReflectionTestBar cfg_bar(bar);
    ASSERT_EQ(FieldDefinedInPbMessage(bar, "required_foo"),
              cfg_bar.FieldDefined4FieldName("required_foo"));
  }
}

TEST(CfgReflection, FieldDefined_optional_not_set) {
  {
    ReflectionTestFoo foo;
    cfg::ReflectionTestFoo cfg_foo(foo);
    ASSERT_EQ(FieldDefinedInPbMessage(foo, "optional_string"),
              cfg_foo.FieldDefined4FieldName("optional_string"));
  }
  {
    ReflectionTestBar bar;
    cfg::ReflectionTestBar cfg_bar(bar);
    ASSERT_EQ(FieldDefinedInPbMessage(bar, "optional_foo"),
              cfg_bar.FieldDefined4FieldName("optional_foo"));
  }
}

TEST(CfgReflection, FieldDefined_optional_set) {
  {
    ReflectionTestFoo foo;
    foo.set_optional_string("");
    cfg::ReflectionTestFoo cfg_foo(foo);
    ASSERT_EQ(FieldDefinedInPbMessage(foo, "optional_string"),
              cfg_foo.FieldDefined4FieldName("optional_string"));
  }
  {
    ReflectionTestBar bar;
    bar.mutable_optional_foo();
    cfg::ReflectionTestBar cfg_bar(bar);
    ASSERT_EQ(FieldDefinedInPbMessage(bar, "optional_foo"),
              cfg_bar.FieldDefined4FieldName("optional_foo"));
  }
}

TEST(CfgReflection, FieldDefined_repeated_not_set) {
  {
    ReflectionTestFoo foo;
    cfg::ReflectionTestFoo cfg_foo(foo);
    ASSERT_EQ(FieldDefinedInPbMessage(foo, "repeated_int32"),
              cfg_foo.FieldDefined4FieldName("repeated_int32"));
  }
  {
    ReflectionTestBar bar;
    cfg::ReflectionTestBar cfg_bar(bar);
    ASSERT_EQ(FieldDefinedInPbMessage(bar, "repeated_foo"),
              cfg_bar.FieldDefined4FieldName("repeated_foo"));
  }
}

TEST(CfgReflection, FieldDefined_repeated_set) {
  {
    ReflectionTestFoo foo;
    foo.add_repeated_int32(0);
    cfg::ReflectionTestFoo cfg_foo(foo);
    ASSERT_EQ(FieldDefinedInPbMessage(foo, "repeated_int32"),
              cfg_foo.FieldDefined4FieldName("repeated_int32"));
  }
  {
    ReflectionTestBar bar;
    bar.add_repeated_foo();
    cfg::ReflectionTestBar cfg_bar(bar);
    ASSERT_EQ(FieldDefinedInPbMessage(bar, "repeated_foo"),
              cfg_bar.FieldDefined4FieldName("repeated_foo"));
  }
}

TEST(CfgReflection, FieldDefined_map_not_set) {
  {
    ReflectionTestFoo foo;
    cfg::ReflectionTestFoo cfg_foo(foo);
    ASSERT_EQ(FieldDefinedInPbMessage(foo, "map_int32"),
              cfg_foo.FieldDefined4FieldName("map_int32"));
  }
  {
    ReflectionTestBar bar;
    cfg::ReflectionTestBar cfg_bar(bar);
    ASSERT_EQ(FieldDefinedInPbMessage(bar, "map_foo"), cfg_bar.FieldDefined4FieldName("map_foo"));
  }
}

TEST(CfgReflection, FieldDefined_map_set) {
  {
    ReflectionTestFoo foo;
    (*foo.mutable_map_int32())[0] = 0;
    cfg::ReflectionTestFoo cfg_foo(foo);
    ASSERT_EQ(FieldDefinedInPbMessage(foo, "map_int32"),
              cfg_foo.FieldDefined4FieldName("map_int32"));
  }
  {
    ReflectionTestBar bar;
    (*bar.mutable_map_foo())[0] = ReflectionTestFoo();
    cfg::ReflectionTestBar cfg_bar(bar);
    ASSERT_EQ(FieldDefinedInPbMessage(bar, "map_foo"), cfg_bar.FieldDefined4FieldName("map_foo"));
  }
}

TEST(CfgReflection, FieldDefined_oneof_not_set) {
  {
    ReflectionTestFoo foo;
    cfg::ReflectionTestFoo cfg_foo(foo);
    ASSERT_EQ(FieldDefinedInPbMessage(foo, "oneof_int32"),
              cfg_foo.FieldDefined4FieldName("oneof_int32"));
  }
  {
    ReflectionTestBar bar;
    cfg::ReflectionTestBar cfg_bar(bar);
    ASSERT_EQ(FieldDefinedInPbMessage(bar, "oneof_foo"),
              cfg_bar.FieldDefined4FieldName("oneof_foo"));
    ASSERT_TRUE(cfg_bar.FieldDefined4FieldName<cfg::Message>("oneof_foo"));
    ASSERT_TRUE(cfg_bar.FieldDefined4FieldName<cfg::ReflectionTestFoo>("oneof_foo"));
  }
}

TEST(CfgReflection, FieldDefined_oneof_set) {
  {
    ReflectionTestFoo foo;
    foo.set_oneof_int32(0);
    cfg::ReflectionTestFoo cfg_foo(foo);
    ASSERT_EQ(FieldDefinedInPbMessage(foo, "oneof_int32"),
              cfg_foo.FieldDefined4FieldName("oneof_int32"));
  }
  {
    ReflectionTestBar bar;
    bar.mutable_oneof_foo();
    cfg::ReflectionTestBar cfg_bar(bar);
    ASSERT_EQ(FieldDefinedInPbMessage(bar, "oneof_foo"),
              cfg_bar.FieldDefined4FieldName("oneof_foo"));
  }
}

TEST(CfgReflection, FieldPtr) {
  {
    cfg::ReflectionTestFoo cfg_foo;
    cfg_foo.set_required_int32(0);
    ASSERT_EQ(&cfg_foo.required_int32(), cfg_foo.FieldPtr4FieldName<int32_t>("required_int32"));
    cfg_foo.set_optional_string("");
    ASSERT_EQ(&cfg_foo.optional_string(),
              cfg_foo.FieldPtr4FieldName<std::string>("optional_string"));
    cfg_foo.add_repeated_int32(0);
    using RepeatedField = cfg::_RepeatedField_<int32_t>;
    ASSERT_EQ(dynamic_cast<const RepeatedField*>(&cfg_foo.repeated_int32()),
              cfg_foo.FieldPtr4FieldName<RepeatedField>("repeated_int32"));
    (*cfg_foo.mutable_map_int32())[0] = 0;
    using MapField = cfg::_MapField_<int32_t, int32_t>;
    ASSERT_EQ(dynamic_cast<const MapField*>(&cfg_foo.map_int32()),
              cfg_foo.FieldPtr4FieldName<MapField>("map_int32"));
    cfg_foo.set_oneof_int32(0);
    ASSERT_EQ(&cfg_foo.oneof_int32(), cfg_foo.FieldPtr4FieldName<int32_t>("oneof_int32"));
  }
  {
    cfg::ReflectionTestBar cfg_bar;
    cfg_bar.mutable_required_foo();
    ASSERT_EQ(&cfg_bar.required_foo(),
              cfg_bar.FieldPtr4FieldName<cfg::ReflectionTestFoo>("required_foo"));
    cfg_bar.mutable_optional_foo();
    ASSERT_EQ(&cfg_bar.optional_foo(),
              cfg_bar.FieldPtr4FieldName<cfg::ReflectionTestFoo>("optional_foo"));
    cfg_bar.add_repeated_foo();
    using RepeatedField = cfg::_RepeatedField_<cfg::ReflectionTestFoo>;
    ASSERT_EQ(dynamic_cast<const RepeatedField*>(&cfg_bar.repeated_foo()),
              cfg_bar.FieldPtr4FieldName<RepeatedField>("repeated_foo"));
    (*cfg_bar.mutable_map_foo())[0] = cfg::ReflectionTestFoo();
    using MapField = cfg::_MapField_<int32_t, cfg::ReflectionTestFoo>;
    ASSERT_EQ(dynamic_cast<const MapField*>(&cfg_bar.map_foo()),
              cfg_bar.FieldPtr4FieldName<MapField>("map_foo"));
    cfg_bar.mutable_oneof_foo();
    ASSERT_EQ(&cfg_bar.oneof_foo(),
              cfg_bar.FieldPtr4FieldName<cfg::ReflectionTestFoo>("oneof_foo"));
  }
}

TEST(CfgReflection, MutableFieldPtr) {
  {
    cfg::ReflectionTestFoo cfg_foo;
    cfg_foo.set_required_int32(0);
    ASSERT_EQ(cfg_foo.mutable_required_int32(),
              cfg_foo.MutableFieldPtr4FieldName<int32_t>("required_int32"));
    cfg_foo.set_optional_string("");
    ASSERT_EQ(cfg_foo.mutable_optional_string(),
              cfg_foo.MutableFieldPtr4FieldName<std::string>("optional_string"));
    cfg_foo.add_repeated_int32(0);
    using RepeatedField = cfg::_RepeatedField_<int32_t>;
    ASSERT_EQ(dynamic_cast<RepeatedField*>(cfg_foo.mutable_repeated_int32()),
              cfg_foo.MutableFieldPtr4FieldName<RepeatedField>("repeated_int32"));
    (*cfg_foo.mutable_map_int32())[0] = 0;
    using MapField = cfg::_MapField_<int32_t, int32_t>;
    ASSERT_EQ(dynamic_cast<MapField*>(cfg_foo.mutable_map_int32()),
              cfg_foo.MutableFieldPtr4FieldName<MapField>("map_int32"));
    cfg_foo.set_oneof_int32(0);
    ASSERT_EQ(cfg_foo.mutable_oneof_int32(),
              cfg_foo.MutableFieldPtr4FieldName<int32_t>("oneof_int32"));
  }
  {
    cfg::ReflectionTestBar cfg_bar;
    cfg_bar.mutable_required_foo();
    ASSERT_EQ(cfg_bar.mutable_required_foo(),
              cfg_bar.MutableFieldPtr4FieldName<cfg::ReflectionTestFoo>("required_foo"));
    cfg_bar.mutable_optional_foo();
    ASSERT_EQ(cfg_bar.mutable_optional_foo(),
              cfg_bar.MutableFieldPtr4FieldName<cfg::ReflectionTestFoo>("optional_foo"));
    cfg_bar.add_repeated_foo();
    using RepeatedField = cfg::_RepeatedField_<cfg::ReflectionTestFoo>;
    ASSERT_EQ(dynamic_cast<RepeatedField*>(cfg_bar.mutable_repeated_foo()),
              cfg_bar.MutableFieldPtr4FieldName<RepeatedField>("repeated_foo"));
    (*cfg_bar.mutable_map_foo())[0] = cfg::ReflectionTestFoo();
    using MapField = cfg::_MapField_<int32_t, cfg::ReflectionTestFoo>;
    ASSERT_EQ(dynamic_cast<MapField*>(cfg_bar.mutable_map_foo()),
              cfg_bar.MutableFieldPtr4FieldName<MapField>("map_foo"));
    cfg_bar.mutable_oneof_foo();
    ASSERT_EQ(cfg_bar.mutable_oneof_foo(),
              cfg_bar.MutableFieldPtr4FieldName<cfg::ReflectionTestFoo>("oneof_foo"));
  }
}

TEST(CfgReflection, GetValFromCfgMessage) {
  ReflectionTestFoo foo;
  foo.set_required_int32(8888);
  cfg::ReflectionTestFoo cfg_foo(foo);
  ASSERT_EQ(GetValFromCfgMessage<int32_t>(cfg_foo, "required_int32"), foo.required_int32());
}

TEST(CfgReflection, GetCfgRfFromCfgMessage) {
  ReflectionTestFoo foo;
  foo.add_repeated_int32(8888);
  cfg::ReflectionTestFoo cfg_foo(foo);
  ASSERT_EQ(GetCfgRfFromCfgMessage<int32_t>(cfg_foo, "repeated_int32").size(),
            GetPbRfFromPbMessage<int32_t>(foo, "repeated_int32").size());
  ASSERT_EQ(GetCfgRfFromCfgMessage<int32_t>(cfg_foo, "repeated_int32").Get(0),
            GetPbRfFromPbMessage<int32_t>(foo, "repeated_int32").Get(0));
}

TEST(CfgReflection, GetCfgRpfFromCfgMessage) {
  ReflectionTestBar bar;
  bar.mutable_repeated_foo()->Add();
  cfg::ReflectionTestBar cfg_bar(bar);
  ASSERT_EQ(GetCfgRpfFromCfgMessage<cfg::ReflectionTestFoo>(cfg_bar, "repeated_foo").size(), 1);
}

TEST(CfgReflection, MutCfgRpfFromCfgMessage) {
  ReflectionTestBar bar;
  bar.mutable_repeated_foo()->Add();
  cfg::ReflectionTestBar cfg_bar(bar);
  ASSERT_EQ(MutCfgRpfFromCfgMessage<cfg::ReflectionTestFoo>(&cfg_bar, "repeated_foo")->size(), 1);
}

TEST(CfgReflection, SetValInCfgMessage) {
  ReflectionTestFoo foo;
  SetValInPbMessage<int32_t>(&foo, "required_int32", 8888);
  cfg::ReflectionTestFoo cfg_foo;
  SetValInCfgMessage<int32_t>(&cfg_foo, "required_int32", 8888);
  ASSERT_EQ(cfg_foo.required_int32(), foo.required_int32());
  ASSERT_EQ(cfg_foo.required_int32(), 8888);
}

TEST(CfgReflection, GetMessageInCfgMessage) {
  ReflectionTestBar bar;
  bar.mutable_required_foo()->set_required_int32(8888);
  cfg::ReflectionTestBar cfg_bar(bar);
  ASSERT_EQ(
      GetValFromPbMessage<int32_t>(GetMessageInPbMessage(bar, "required_foo"), "required_int32"),
      GetValFromCfgMessage<int32_t>(GetMessageInCfgMessage(cfg_bar, "required_foo"),
                                    "required_int32"));
}

TEST(CfgReflection, MutableMessageInCfgMessage) {
  ReflectionTestBar bar;
  SetValInPbMessage<int32_t>(MutableMessageInPbMessage(&bar, "required_foo"), "required_int32",
                             8888);
  cfg::ReflectionTestBar cfg_bar;
  SetValInCfgMessage<int32_t>(MutableMessageInCfgMessage(&cfg_bar, "required_foo"),
                              "required_int32", 8888);
  ASSERT_EQ(cfg_bar.required_foo().required_int32(), bar.required_foo().required_int32());
  ASSERT_EQ(cfg_bar.required_foo().required_int32(), 8888);
}

TEST(CfgReflection, GetStrValInCfgFdOrCfgRpf) {
  cfg::ReflectionTestFoo cfg_foo;
  cfg_foo.set_optional_string("8888");
  cfg_foo.add_repeated_string("8888");
  ASSERT_EQ("8888", GetStrValInCfgFdOrCfgRpf(cfg_foo, "optional_string"));
  ASSERT_EQ("8888", GetStrValInCfgFdOrCfgRpf(cfg_foo, "repeated_string_0"));
}

TEST(CfgReflection, HasStrFieldInCfgFdOrCfgRpf) {
  cfg::ReflectionTestFoo cfg_foo;
  cfg_foo.set_optional_string("8888");
  cfg_foo.add_repeated_string("8888");
  ASSERT_TRUE(HasStrFieldInCfgFdOrCfgRpf(cfg_foo, "optional_string"));
  ASSERT_FALSE(HasStrFieldInCfgFdOrCfgRpf(cfg_foo, "undefined_field"));
  ASSERT_TRUE(HasStrFieldInCfgFdOrCfgRpf(cfg_foo, "repeated_string_0"));
  ASSERT_FALSE(HasStrFieldInCfgFdOrCfgRpf(cfg_foo, "repeated_string_1"));
}

TEST(CfgReflection, ReplaceStrValInCfgFdOrCfgRpf) {
  cfg::ReflectionTestFoo cfg_foo;
  cfg_foo.set_optional_string("8888");
  cfg_foo.add_repeated_string("8888");
  ASSERT_EQ("8888", ReplaceStrValInCfgFdOrCfgRpf(&cfg_foo, "optional_string", "4444"));
  ASSERT_EQ("8888", ReplaceStrValInCfgFdOrCfgRpf(&cfg_foo, "repeated_string_0", "4444"));
  ASSERT_EQ("4444", ReplaceStrValInCfgFdOrCfgRpf(&cfg_foo, "optional_string", "8888"));
  ASSERT_EQ("4444", ReplaceStrValInCfgFdOrCfgRpf(&cfg_foo, "repeated_string_0", "8888"));
}

TEST(CfgReflection, AddValInCfgRf) {
  cfg::ReflectionTestFoo cfg_foo;
  AddValInCfgRf<int32_t>(&cfg_foo, "repeated_int32", 8888);
  ASSERT_EQ(cfg_foo.repeated_int32().size(), 1);
  ASSERT_EQ(cfg_foo.repeated_int32().Get(0), 8888);
}

TEST(Cfg, ResizeRepeated) {
  cfg::ReflectionTestBar bar;
  auto* first = bar.mutable_repeated_foo()->Add();
  for (int i = 0; i < 100; ++i) { bar.mutable_repeated_foo(); }
  ASSERT_TRUE(bar.mutable_repeated_foo(0) == first);
}

}  // namespace test
}  // namespace oneflow
