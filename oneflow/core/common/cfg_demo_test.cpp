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
#include "oneflow/core/common/demo.cfg.h"
#include "oneflow/core/common/demo.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/cfg.h"
#include <iostream>
using namespace std;
namespace oneflow {
namespace test {

void CfgPassByValue(oneflow::cfg::Foo foo) {
  ASSERT_EQ(foo.freeze(), true);
  ASSERT_EQ(foo.bar().freeze(), true);
  ASSERT_EQ(foo.bars().freeze(), true);
  ASSERT_EQ(foo.bar().doo().freeze(), true);
  ASSERT_EQ(foo.bar().doo().noo().freeze(), true);
  ASSERT_EQ(foo.bar().doo().noos().freeze(), true);
  ASSERT_EQ(foo.bar().moo().freeze(), true);
  ASSERT_EQ(foo.map_bar().freeze(), true);
  ASSERT_EQ(foo.bar().doo().map_noo().freeze(), true);
  ASSERT_EQ(foo.bar().doo().of_noo().freeze(), true);
  ASSERT_EQ(foo.bar().moo().of_noo2().freeze(), true);
}

TEST(Demo, freeze_root_node) {
  cfg::Foo foo;
  cfg::Bar bar;

  foo.set_name("foo");
  foo.mutable_bar()->set_name("bar");
  foo.mutable_bar()->mutable_doo()->set_name("doo");
  foo.mutable_bar()->mutable_doo()->mutable_of_noo()->set_name("of_noo");
  foo.mutable_bar()->mutable_moo()->set_name("moo");
  foo.mutable_bar()->mutable_doo()->mutable_noo()->set_name("noo");
  foo.mutable_bar()->mutable_moo()->mutable_of_noo2()->set_name("of_noo");
  foo.set_freeze();

  ASSERT_EQ(foo.freeze(), true);
  ASSERT_EQ(foo.bar().freeze(), true);
  ASSERT_EQ(foo.bars().freeze(), true);
  ASSERT_EQ(foo.bar().doo().freeze(), true);
  ASSERT_EQ(foo.bar().doo().noo().freeze(), true);
  ASSERT_EQ(foo.bar().doo().noos().freeze(), true);
  ASSERT_EQ(foo.bar().moo().freeze(), true);
  ASSERT_EQ(foo.map_bar().freeze(), true);
  ASSERT_EQ(foo.bar().doo().map_noo().freeze(), true);
  ASSERT_EQ(foo.bar().doo().of_noo().freeze(), true);
  ASSERT_EQ(foo.bar().moo().of_noo2().freeze(), true);
}

TEST(Demo, pass_by_value) {
  cfg::Foo foo;
  cfg::Bar bar;

  foo.set_name("foo");
  foo.mutable_bar()->set_name("bar");
  foo.mutable_bar()->mutable_doo()->set_name("doo");
  foo.mutable_bar()->mutable_doo()->mutable_of_noo()->set_name("of_noo");
  foo.mutable_bar()->mutable_moo()->set_name("moo");
  foo.mutable_bar()->mutable_doo()->mutable_noo()->set_name("noo");
  foo.mutable_bar()->mutable_moo()->mutable_of_noo2()->set_name("of_noo");
  foo.set_freeze();
  CfgPassByValue(foo);
}

TEST(Demo, freeze_mid_node) {
  cfg::Foo foo;
  cfg::Bar bar;

  foo.set_name("foo");
  foo.mutable_bar()->set_name("bar");
  foo.mutable_bar()->mutable_doo()->set_name("doo");
  foo.mutable_bar()->mutable_doo()->mutable_of_noo()->set_name("of_noo");
  foo.mutable_bar()->mutable_moo()->set_name("moo");
  foo.mutable_bar()->mutable_doo()->mutable_noo()->set_name("noo");
  foo.mutable_bar()->mutable_moo()->mutable_of_noo2()->set_name("of_noo");
  foo.mutable_bar()->mutable_doo()->set_freeze();

  ASSERT_EQ(foo.freeze(), false);
  ASSERT_EQ(foo.bar().freeze(), false);
  ASSERT_EQ(foo.bars().freeze(), false);
  ASSERT_EQ(foo.bar().doo().freeze(), true);
  ASSERT_EQ(foo.bar().doo().noo().freeze(), true);
  ASSERT_EQ(foo.bar().doo().noos().freeze(), true);
  ASSERT_EQ(foo.bar().moo().freeze(), false);
  ASSERT_EQ(foo.map_bar().freeze(), false);
  ASSERT_EQ(foo.bar().doo().map_noo().freeze(), true);
  ASSERT_EQ(foo.bar().doo().of_noo().freeze(), true);
  ASSERT_EQ(foo.bar().moo().of_noo2().freeze(), false);
}

}  // namespace test
}  // namespace oneflow
