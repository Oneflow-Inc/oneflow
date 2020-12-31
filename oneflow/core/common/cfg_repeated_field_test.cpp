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
#include "oneflow/core/common/cfg.h"
namespace oneflow {
namespace test {

TEST(CfgReflection, FieldDefined_repeated_field_not_set) {
  {
    ReflectionTestFoo foo;
    cfg::ReflectionTestFoo cfg_foo(foo);
    const oneflow::cfg::_RepeatedField_<int>& repeated_int32 = oneflow::cfg::_RepeatedField_<int>();
    static_assert(std::is_same<decltype(repeated_int32), decltype(cfg_foo.repeated_int32())>::value,
                  "Not a oneflow::cfg::_RepeatedField_<int> type!");
  }
  {
    ReflectionTestBar bar;
    cfg::ReflectionTestBar cfg_bar(bar);
    const oneflow::cfg::_RepeatedField_<cfg::ReflectionTestFoo>& repeated_foo =
        oneflow::cfg::_RepeatedField_<cfg::ReflectionTestFoo>();
    static_assert(std::is_same<decltype(repeated_foo), decltype(cfg_bar.repeated_foo())>::value,
                  "Not a oneflow::cfg::_RepeatedField_<cfg::ReflectionTestFoo> type!");
  }
}

TEST(CfgReflection, FieldDefined_repeated_field_set) {
  {
    ReflectionTestFoo foo;
    foo.add_repeated_int32(0);
    cfg::ReflectionTestFoo cfg_foo(foo);
    const oneflow::cfg::_RepeatedField_<int>& repeated_int32 = oneflow::cfg::_RepeatedField_<int>();
    static_assert(std::is_same<decltype(repeated_int32), decltype(cfg_foo.repeated_int32())>::value,
                  "Not a oneflow::cfg::_RepeatedField_<int> type!");
  }
  {
    ReflectionTestBar bar;
    bar.add_repeated_foo();
    cfg::ReflectionTestBar cfg_bar(bar);
    const oneflow::cfg::_RepeatedField_<cfg::ReflectionTestFoo>& repeated_foo =
        oneflow::cfg::_RepeatedField_<cfg::ReflectionTestFoo>();
    static_assert(std::is_same<decltype(repeated_foo), decltype(cfg_bar.repeated_foo())>::value,
                  "Not a oneflow::cfg::_RepeatedField_<cfg::ReflectionTestFoo> type!");
  }
}

}  // namespace test
}  // namespace oneflow
