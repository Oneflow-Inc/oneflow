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
#include "oneflow/core/common/cfg_reflection_test.cfg.h"
#include "oneflow/core/common/cfg_reflection_test.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {
namespace test {

TEST(CfgReflection, HasField_required_not_set) {
  {
    ReflectionTestFoo foo;
    cfg::ReflectionTestFoo cfg_foo(foo);
    ASSERT_EQ(HasFieldInPbMessage(foo, "required_int32"),
              cfg_foo.HasField4FieldName("required_int32"));
  }
  {
    ReflectionTestBar bar;
    cfg::ReflectionTestBar cfg_bar(bar);
    ASSERT_EQ(HasFieldInPbMessage(bar, "required_foo"), cfg_bar.HasField4FieldName("required_foo"));
  }
}

TEST(CfgReflection, HasField_required_set) {
  {
    ReflectionTestFoo foo;
    foo.set_required_int32(0);
    cfg::ReflectionTestFoo cfg_foo(foo);
    ASSERT_EQ(HasFieldInPbMessage(foo, "required_int32"),
              cfg_foo.HasField4FieldName("required_int32"));
  }
  {
    ReflectionTestBar bar;
    bar.mutable_required_foo();
    cfg::ReflectionTestBar cfg_bar(bar);
    ASSERT_EQ(HasFieldInPbMessage(bar, "required_foo"), cfg_bar.HasField4FieldName("required_foo"));
  }
}

TEST(CfgReflection, HasField_optional_not_set) {
  {
    ReflectionTestFoo foo;
    cfg::ReflectionTestFoo cfg_foo(foo);
    ASSERT_EQ(HasFieldInPbMessage(foo, "optional_string"),
              cfg_foo.HasField4FieldName("optional_string"));
  }
  {
    ReflectionTestBar bar;
    cfg::ReflectionTestBar cfg_bar(bar);
    ASSERT_EQ(HasFieldInPbMessage(bar, "optional_foo"), cfg_bar.HasField4FieldName("optional_foo"));
  }
}

TEST(CfgReflection, HasField_optional_set) {
  {
    ReflectionTestFoo foo;
    foo.set_optional_string("");
    cfg::ReflectionTestFoo cfg_foo(foo);
    ASSERT_EQ(HasFieldInPbMessage(foo, "optional_string"),
              cfg_foo.HasField4FieldName("optional_string"));
  }
  {
    ReflectionTestBar bar;
    bar.mutable_optional_foo();
    cfg::ReflectionTestBar cfg_bar(bar);
    ASSERT_EQ(HasFieldInPbMessage(bar, "optional_foo"), cfg_bar.HasField4FieldName("optional_foo"));
  }
}

TEST(CfgReflection, HasField_repeated_not_set) {
  {
    ReflectionTestFoo foo;
    cfg::ReflectionTestFoo cfg_foo(foo);
    ASSERT_EQ(HasFieldInPbMessage(foo, "repeated_int32"),
              cfg_foo.HasField4FieldName("repeated_int32"));
  }
  {
    ReflectionTestBar bar;
    cfg::ReflectionTestBar cfg_bar(bar);
    ASSERT_EQ(HasFieldInPbMessage(bar, "repeated_foo"), cfg_bar.HasField4FieldName("repeated_foo"));
  }
}

TEST(CfgReflection, HasField_repeated_set) {
  {
    ReflectionTestFoo foo;
    foo.add_repeated_int32(0);
    cfg::ReflectionTestFoo cfg_foo(foo);
    ASSERT_EQ(HasFieldInPbMessage(foo, "repeated_int32"),
              cfg_foo.HasField4FieldName("repeated_int32"));
  }
  {
    ReflectionTestBar bar;
    bar.add_repeated_foo();
    cfg::ReflectionTestBar cfg_bar(bar);
    ASSERT_EQ(HasFieldInPbMessage(bar, "repeated_foo"), cfg_bar.HasField4FieldName("repeated_foo"));
  }
}

TEST(CfgReflection, HasField_map_not_set) {
  {
    ReflectionTestFoo foo;
    cfg::ReflectionTestFoo cfg_foo(foo);
    ASSERT_EQ(HasFieldInPbMessage(foo, "map_int32"), cfg_foo.HasField4FieldName("map_int32"));
  }
  {
    ReflectionTestBar bar;
    cfg::ReflectionTestBar cfg_bar(bar);
    ASSERT_EQ(HasFieldInPbMessage(bar, "map_foo"), cfg_bar.HasField4FieldName("map_foo"));
  }
}

TEST(CfgReflection, HasField_map_set) {
  {
    ReflectionTestFoo foo;
    (*foo.mutable_map_int32())[0] = 0;
    cfg::ReflectionTestFoo cfg_foo(foo);
    ASSERT_EQ(HasFieldInPbMessage(foo, "map_int32"), cfg_foo.HasField4FieldName("map_int32"));
  }
  {
    ReflectionTestBar bar;
    (*bar.mutable_map_foo())[0] = ReflectionTestFoo();
    cfg::ReflectionTestBar cfg_bar(bar);
    ASSERT_EQ(HasFieldInPbMessage(bar, "map_foo"), cfg_bar.HasField4FieldName("map_foo"));
  }
}

TEST(CfgReflection, HasField_oneof_not_set) {
  {
    ReflectionTestFoo foo;
    cfg::ReflectionTestFoo cfg_foo(foo);
    ASSERT_EQ(HasFieldInPbMessage(foo, "oneof_int32"), cfg_foo.HasField4FieldName("oneof_int32"));
  }
  {
    ReflectionTestBar bar;
    cfg::ReflectionTestBar cfg_bar(bar);
    ASSERT_EQ(HasFieldInPbMessage(bar, "oneof_foo"), cfg_bar.HasField4FieldName("oneof_foo"));
  }
}

TEST(CfgReflection, HasField_oneof_set) {
  {
    ReflectionTestFoo foo;
    foo.set_oneof_int32(0);
    cfg::ReflectionTestFoo cfg_foo(foo);
    ASSERT_EQ(HasFieldInPbMessage(foo, "oneof_int32"), cfg_foo.HasField4FieldName("oneof_int32"));
  }
  {
    ReflectionTestBar bar;
    bar.mutable_oneof_foo();
    cfg::ReflectionTestBar cfg_bar(bar);
    ASSERT_EQ(HasFieldInPbMessage(bar, "oneof_foo"), cfg_bar.HasField4FieldName("oneof_foo"));
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

}  // namespace test
}  // namespace oneflow
