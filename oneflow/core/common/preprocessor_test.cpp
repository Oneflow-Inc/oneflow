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
#include <gtest/gtest.h>
#include <unordered_map>
#include "oneflow/core/common/data_type.h"

namespace oneflow {

TEST(PP_SEQ, internal_seq_size) {
#define SEQ (1)(2)(3)
  ASSERT_EQ(OF_PP_SEQ_SIZE(SEQ), 3);
#undef SEQ
}

TEST(PP_SEQ, internal_big_seq_size) {
#define SEQ                                                                                        \
  (0)(1)(2)(3)(4)(5)(6)(7)(8)(9)(10)(11)(12)(13)(14)(15)(16)(17)(18)(19)(20)(21)(22)(23)(24)(25)(  \
      26)(27)(28)(29)(30)(31)(32)(33)(34)(35)(36)(37)(38)(39)(40)(41)(42)(43)(44)(45)(46)(47)(48)( \
      49)(50)(51)(52)(53)(54)(55)(56)(57)(58)(59)(60)(61)(62)(63)
  ASSERT_EQ(OF_PP_SEQ_SIZE(SEQ), 64);
#undef SEQ
}

TEST(PP_SEQ, internal_for_each) {
#define SEQ (1)(2)(3)(4)
#define MAKE_PAIR(x) {x, x},
  std::unordered_map<int, int> identity = {OF_PP_INTERNAL_SEQ_FOR_EACH_ATOMIC(MAKE_PAIR, _, SEQ)};
#undef MAKE_PAIR
#undef SEQ
  for (int i = 1; i <= 4; ++i) { ASSERT_EQ(i, identity[i]); }
}

TEST(PP_TUPLE, internal_is_tuple_empty) {
  ASSERT_EQ(OF_PP_INTERNAL_IS_TUPLE_EMPTY(()), 1);
  ASSERT_EQ(OF_PP_INTERNAL_IS_TUPLE_EMPTY((1)), 0);
  ASSERT_EQ(OF_PP_INTERNAL_IS_TUPLE_EMPTY((1, 2)), 0);
}

TEST(PP_TUPLE, internal_tuple_size) {
  ASSERT_EQ(OF_PP_INTERNAL_TUPLE_SIZE(()), 0);
  ASSERT_EQ(OF_PP_INTERNAL_TUPLE_SIZE((1)), 1);
  ASSERT_EQ(OF_PP_INTERNAL_TUPLE_SIZE((1, 2)), 2);
  ASSERT_EQ(OF_PP_INTERNAL_TUPLE_SIZE((1, 2, 3)), 3);
  ASSERT_EQ(OF_PP_INTERNAL_TUPLE_SIZE((1, 2, 3, 4)), 4);
  ASSERT_EQ(OF_PP_INTERNAL_TUPLE_SIZE((1, 2, 3, 4, 5)), 5);
}

TEST(PP_SEQ, internal_seq_product) {
#define SEQ (0)(1)
  std::string expanded(OF_PP_STRINGIZE(OF_PP_INTERNAL_SEQ_PRODUCT(SEQ, SEQ)));
#undef SEQ
  ASSERT_TRUE((expanded == "((0, 0)) ((1, 0)) ((0, 1)) ((1, 1))")
              || (expanded == "((0, 0)) ((1, 0))  ((0, 1)) ((1, 1))"));
}

TEST(PP_SEQ, internal_different_seq_product) {
#define SEQ1 (0)(1)
#define SEQ2 (a)(b)
  std::string expanded(OF_PP_STRINGIZE(OF_PP_INTERNAL_SEQ_PRODUCT(SEQ1, SEQ2)));
#undef SEQ1
#undef SEQ2
  ASSERT_TRUE((expanded == "((0, a)) ((1, a)) ((0, b)) ((1, b))")
              || (expanded == "((0, a)) ((1, a))  ((0, b)) ((1, b))"));
}

TEST(PP_SEQ, internal_seq_product_for_each) {
#define SEQ (0)(1)
#define MAKE_ENTRY(x, y) {OF_PP_STRINGIZE(OF_PP_CAT(x, y)), x || y},
  std::unordered_map<std::string, bool> or_table = {
      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, OF_PP_INTERNAL_SEQ_PRODUCT(SEQ, SEQ))};
#undef MAKE_ENTRY
#undef SEQ
  ASSERT_EQ(or_table["00"], false);
  ASSERT_EQ(or_table["01"], true);
  ASSERT_EQ(or_table["10"], true);
  ASSERT_EQ(or_table["11"], true);
}

TEST(PP, stringize) {
  ASSERT_EQ(OF_PP_STRINGIZE(foo), "foo");
  ASSERT_EQ(OF_PP_STRINGIZE(bar), "bar");
}

TEST(PP, concate) {
  ASSERT_EQ(OF_PP_CAT(OF_PP_, STRINGIZE)(foo), "foo");
  ASSERT_EQ(OF_PP_CAT(OF_PP_, STRINGIZE)(bar), "bar");
}

TEST(PP_SEQ, make_tuple_seq) { ASSERT_EQ(OF_PP_STRINGIZE(OF_PP_MAKE_TUPLE_SEQ(1, 2)), "((1, 2))"); }

TEST(PP_SEQ, for_each_tuple) {
#define SEQ ((1, 1))((2, 2))((3, 3))((4, 4))
#define MAKE_ENTRY(x, y) {x, y},
  std::unordered_map<int, int> identity = {OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, SEQ)};
#undef MAKE_ENTRY
#undef SEQ
  for (int i = 1; i <= 4; ++i) { ASSERT_EQ(i, identity[i]); }
}

TEST(PP_SEQ, outter_for_each_tuple) {
#define SEQ ((1, 1))((2, 2))((3, 3))((4, 4))
#define MAKE_ENTRY(x, y) {x, y},
  std::unordered_map<int, int> identity = {OF_PP_OUTTER_FOR_EACH_TUPLE(MAKE_ENTRY, SEQ)};
#undef MAKE_ENTRY
#undef SEQ
  for (int i = 1; i <= 4; ++i) { ASSERT_EQ(i, identity[i]); }
}

TEST(PP_SEQ, nested_for_each_tuple) {
#define SEQ ((0))((1))((2))((3))
#define MAKE_INNER(x) x,
#define MAKE_OUTTER(x) {OF_PP_FOR_EACH_TUPLE(MAKE_INNER, SEQ)},
  std::vector<std::vector<int>> table = {OF_PP_OUTTER_FOR_EACH_TUPLE(MAKE_OUTTER, SEQ)};
#undef MAKE_OUTTER
#undef MAKE_INNER
#undef SEQ
  ASSERT_EQ(table.size(), 4);
  for (int i = 0; i < 4; ++i) {
    ASSERT_EQ(table[i].size(), 4);
    for (int j = 0; j < 4; ++j) { ASSERT_EQ(j, table[i][j]); }
  }
}

TEST(PP_SEQ, seq_product_for_each) {
#define SEQ (0)(1)
#define MAKE_ENTRY(x, y) {OF_PP_STRINGIZE(OF_PP_CAT(x, y)), x || y},
  std::unordered_map<std::string, bool> or_table = {
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, SEQ, SEQ)};
#undef MAKE_ENTRY
#undef SEQ
  ASSERT_EQ(or_table["00"], false);
  ASSERT_EQ(or_table["01"], true);
  ASSERT_EQ(or_table["10"], true);
  ASSERT_EQ(or_table["11"], true);
}

}  // namespace oneflow
