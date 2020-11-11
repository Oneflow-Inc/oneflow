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
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/flex.h"

namespace oneflow {
namespace test {

FLEX_DEF(Location, builder) {
  return builder.Struct()
      .Required<int32_t>("x")
      .Required<int32_t>("y")
      .Optional<std::string>("description", "Home")
      .Build();
}

void TestField(const StructFlexDef* location) {
  ASSERT_TRUE(location != nullptr);
  ASSERT_TRUE(location->Field4FieldName("x").flex_def()
              == FlexDefBuilderTrait<int32_t>::GetFlexDef());
  ASSERT_TRUE(location->Field4FieldName("y").flex_def()
              == FlexDefBuilderTrait<int32_t>::GetFlexDef());
  ASSERT_TRUE(location->Field4FieldName("description").flex_def()
              == FlexDefBuilderTrait<std::string>::GetFlexDef());
  ASSERT_EQ(location->Field4FieldName("description").default_val()->GetString(), "Home");
}

TEST(FlexDef, field_type) {
  const auto* ptr = Location::GetFlexDef().get();
  const auto* location = dynamic_cast<const StructFlexDef*>(ptr);
  TestField(location);
}

FLEX_DEF(GeoObject, builder) {
  return builder.Struct()
      .Required<Location>("location")
      .Optional<std::string>("name", "undefined")
      .Build();
}

TEST(FlexDef, nested_field_type) {
  const auto* geo_type = dynamic_cast<const StructFlexDef*>(GeoObject::GetFlexDef().get());
  const auto* ptr = geo_type->Field4FieldName("location").flex_def().get();
  const auto* location = dynamic_cast<const StructFlexDef*>(ptr);
  TestField(location);
}

TEST(FlexDef, dynamic_build) {
  std::shared_ptr<const FlexDef> Location = FlexDefBuilder()
                                                .Struct()
                                                .Required<int32_t>("x")
                                                .Required<int32_t>("y")
                                                .Optional<std::string>("description", "Home")
                                                .Build();
  std::shared_ptr<const FlexDef> GeoObject = FlexDefBuilder()
                                                 .Struct()
                                                 .Required(Location, "location")
                                                 .Optional<std::string>("name", "undefined")
                                                 .Build();
  const auto* geo_type = dynamic_cast<const StructFlexDef*>(GeoObject.get());
  const auto* ptr = geo_type->Field4FieldName("location").flex_def().get();
  const auto* location = dynamic_cast<const StructFlexDef*>(ptr);
  TestField(location);
}

TEST(FlexValue, basic) {
  MutFlexValue location = Location::NewMutFlexValue();
  location.Set<int32_t>("x", 30);
  location.Set<int32_t>("y", 40);
  ASSERT_EQ(location.Get<int32_t>("x"), 30);
  ASSERT_EQ(location.Get<int32_t>("y"), 40);
  ASSERT_EQ(location.Get<std::string>("description"), "Home");
}

FLEX_DEF(Ids, builder) { return builder.Struct().List<int32_t>("ids").Build(); }

TEST(List, ids) {
  MutFlexValue ids = Ids::NewMutFlexValue();
  ASSERT_EQ(ids.GetList("ids").size(), 0);
  MutListFlexValue list_int32 = ids.MutableList("ids");
  list_int32.Add<int32_t>(0);
  list_int32.Add<int32_t>(1);
  // also valid if compatible
  list_int32.Add<int16_t>(2);
  // deduced type is valid if compatible
  list_int32.Add(3);
  ASSERT_EQ(list_int32.size(), 4);
  for (int64_t i = 0; i < 4; ++i) { ASSERT_EQ(list_int32.Get<int32_t>(i), i); }
}

FLEX_DEF(Strings, builder) { return builder.Struct().List<std::string>("strings").Build(); }

TEST(List, Strings) {
  MutFlexValue strings = Strings::NewMutFlexValue();
  MutListFlexValue list_string = strings.MutableList("strings");
  list_string.Add<std::string>("0");
  list_string.Add<std::string>("1");
  // also valid if compatible
  list_string.Add<std::string>("2");
  // deduced type is valid if compatible
  list_string.Add<std::string>("3");
  ASSERT_EQ(list_string.size(), 4);
  for (int64_t i = 0; i < 4; ++i) { ASSERT_EQ(list_string.Get<std::string>(i), std::to_string(i)); }
}

TEST(FlexValue, nested) {
  MutFlexValue geo_obj = GeoObject::NewMutFlexValue();
  ASSERT_EQ(geo_obj.Get("location").Get<std::string>("description"), "Home");
  MutFlexValue location = geo_obj.Mutable("location");
  location.Set<int32_t>("x", 30);
  location.Set<int32_t>("y", 40);
  location.Set<std::string>("description", "Company");
  ASSERT_EQ(location.Get<int32_t>("x"), 30);
  ASSERT_EQ(location.Get<int32_t>("y"), 40);
  ASSERT_EQ(geo_obj.Get("location").Get<std::string>("description"), "Company");
}

TEST(FlexValue, dynamic_nested) {
  std::shared_ptr<const FlexDef> Location = FlexDefBuilder()
                                                .Struct()
                                                .Required<int32_t>("x")
                                                .Required<int32_t>("y")
                                                .Optional<std::string>("description", "Home")
                                                .Build();
  std::shared_ptr<const FlexDef> GeoObject = FlexDefBuilder()
                                                 .Struct()
                                                 .Required(Location, "location")
                                                 .Optional<std::string>("name", "undefined")
                                                 .Build();
  MutFlexValue geo_obj = NewMutFlexValue(GeoObject);
  ASSERT_EQ(geo_obj.Get("location").Get<std::string>("description"), "Home");
  MutFlexValue location = geo_obj.Mutable("location");
  location.Set<int32_t>("x", 30);
  location.Set<int32_t>("y", 40);
  location.Set<std::string>("description", "Company");
  ASSERT_EQ(location.Get<int32_t>("x"), 30);
  ASSERT_EQ(location.Get<int32_t>("y"), 40);
  ASSERT_EQ(geo_obj.Get("location").Get<std::string>("description"), "Company");
}

FLEX_DEF(BinaryTree, builder) {
  return builder.Struct()
      .Optional<BinaryTree>("left")
      .Optional<BinaryTree>("right")
      .Optional<int32_t>("weight", 1)
      .Build();
}

TEST(FlexDef, defined_or_has) {
  MutFlexValue tree = BinaryTree::NewMutFlexValue();
  ASSERT_TRUE(tree.Defined("weight"));
  ASSERT_TRUE(!tree.Defined("undefined-field"));
  ASSERT_TRUE(!tree.Has("weight"));
  tree.Set<int32_t>("weight", 3);
  ASSERT_TRUE(tree.Has("weight"));
}

TEST(FlexDef, recursive) {
  MutFlexValue tree = BinaryTree::NewMutFlexValue();
  ASSERT_EQ(tree.Get<int32_t>("weight"), 1);
  ASSERT_TRUE(!tree.Has("left"));
  ASSERT_TRUE(!tree.Has("right"));
  ASSERT_EQ(tree.Get("left").Get<int32_t>("weight"), 1);
  ASSERT_EQ(tree.Get("right").Get<int32_t>("weight"), 1);
  tree.Mutable("left").Set<int32_t>("weight", 2);
  ASSERT_TRUE(tree.Has("left"));
  ASSERT_EQ(tree.Get("left").Get<int32_t>("weight"), 2);
}

DECLARE_FLEX_DEF(RedTree);
DECLARE_FLEX_DEF(BlackTree);

DEFINE_FLEX_DEF(RedTree, builder) {
  return builder.Struct()
      .Optional<BlackTree>("left")
      .Optional<BlackTree>("right")
      .Optional<int32_t>("weight", 1)
      .Build();
}
DEFINE_FLEX_DEF(BlackTree, builder) {
  return builder.Struct()
      .Optional<RedTree>("left")
      .Optional<RedTree>("right")
      .Optional<int32_t>("weight", 1)
      .Build();
}

TEST(FlexDef, recursive_two_flex_def) {
  MutFlexValue tree = BlackTree::NewMutFlexValue();
  ASSERT_EQ(tree.Get<int32_t>("weight"), 1);
  ASSERT_TRUE(!tree.Has("left"));
  ASSERT_TRUE(!tree.Has("right"));
  ASSERT_EQ(tree.Get("left").Get<int32_t>("weight"), 1);
  ASSERT_EQ(tree.Get("right").Get<int32_t>("weight"), 1);
  tree.Mutable("left").Set<int32_t>("weight", 2);
  ASSERT_TRUE(tree.Has("left"));
  ASSERT_EQ(tree.Get("left").Get<int32_t>("weight"), 2);
}

}  // namespace test
}  // namespace oneflow
