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
#include "oneflow/core/common/fixed_vector.h"
#include "gtest/gtest.h"
#include <functional>

namespace oneflow {

namespace test {

using FixedVec = fixed_vector<int, 32>;

TEST(fixed_vector, constructor_0) {
  FixedVec a(8);
  ASSERT_EQ(a.size(), 8);
}

TEST(fixed_vector, constructor_1) {
  int value = 30;
  FixedVec a(8, value);
  ASSERT_TRUE(std::all_of(a.begin(), a.end(), [value](const int x) { return x == value; }));
}

TEST(fixed_vector, constructor_2) {
  std::vector<int> vec{1, 2, 3, 4};
  FixedVec a(vec.begin(), vec.end());
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, constructor_3) {
  std::vector<int> vec{1, 2, 3, 4};
  FixedVec b(vec.begin(), vec.end());
  FixedVec a(b);
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, constructor_4) {
  std::vector<int> vec{1, 2, 3, 4};
  FixedVec b(vec.begin(), vec.end());
  FixedVec a(std::move(b));
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, constructor_5) {
  std::vector<int> vec{1, 2, 3, 4};
  FixedVec a{1, 2, 3, 4};
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, operator_assign_0) {
  std::vector<int> vec{1, 2, 3, 4};
  FixedVec b(vec.begin(), vec.end());
  FixedVec a;
  a = b;
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, operator_assign_1) {
  std::vector<int> vec{1, 2, 3, 4};
  FixedVec b(vec.begin(), vec.end());
  FixedVec a;
  a = std::move(b);
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, operator_assign_2) {
  std::vector<int> vec{1, 2, 3, 4};
  FixedVec a;
  a = {1, 2, 3, 4};
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, assign_0) {
  int value = 30;
  FixedVec a;
  a.assign(8, value);
  ASSERT_TRUE(std::all_of(a.begin(), a.end(), [value](const int x) { return x == value; }));
}

TEST(fixed_vector, assign_1) {
  std::vector<int> vec{1, 2, 3, 4};
  FixedVec a;
  a.assign(vec.begin(), vec.end());
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, assign_2) {
  std::vector<int> vec{1, 2, 3, 4};
  FixedVec a;
  a.assign({1, 2, 3, 4});
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, const_at) {
  int value = 33;
  const FixedVec a{value};
  ASSERT_EQ(a.at(0), value);
}

TEST(fixed_vector, at) {
  int value = 33;
  FixedVec a{0};
  a.at(0) = value;
  ASSERT_EQ(a.at(0), value);
}

TEST(fixed_vector, const_front) {
  int value = 33;
  const FixedVec a{value};
  ASSERT_EQ(a.front(), value);
}

TEST(fixed_vector, front) {
  int value = 33;
  FixedVec a{0};
  a.front() = value;
  ASSERT_EQ(a.front(), value);
}

TEST(fixed_vector, const_back) {
  int value = 33;
  const FixedVec a{1, value};
  ASSERT_EQ(a.back(), value);
}

TEST(fixed_vector, back) {
  int value = 33;
  FixedVec a{1, 0};
  a.back() = value;
  ASSERT_EQ(a.back(), value);
}

TEST(fixed_vector, const_data) {
  int value = 33;
  const FixedVec a{value};
  ASSERT_EQ(*a.data(), value);
}

TEST(fixed_vector, data) {
  int value = 33;
  FixedVec a{0};
  *a.data() = value;
  ASSERT_EQ(*a.data(), value);
}

TEST(fixed_vector, const_begin) {
  int value = 33;
  const FixedVec a{value};
  ASSERT_EQ(*a.begin(), value);
}

TEST(fixed_vector, begin) {
  int value = 33;
  FixedVec a{0};
  *a.begin() = value;
  ASSERT_EQ(*a.begin(), value);
}

TEST(fixed_vector, cbegin) {
  int value = 33;
  FixedVec a{value};
  ASSERT_EQ(*a.cbegin(), value);
}

TEST(fixed_vector, const_end) {
  const FixedVec a{0, 1, 2};
  ASSERT_EQ(a.begin() + a.size(), a.end());
}

TEST(fixed_vector, end) {
  FixedVec a{0, 1, 2};
  ASSERT_EQ(a.begin() + a.size(), a.end());
}

TEST(fixed_vector, cend) {
  FixedVec a{0, 1, 2};
  ASSERT_EQ(a.cbegin() + a.size(), a.cend());
}

TEST(fixed_vector, const_rbegin) {
  int value = 33;
  const FixedVec a{0, value};
  ASSERT_EQ(*a.rbegin(), value);
}

TEST(fixed_vector, rbegin) {
  int value = 33;
  FixedVec a{0, 0};
  *a.rbegin() = value;
  ASSERT_EQ(*a.rbegin(), value);
}

TEST(fixed_vector, crbegin) {
  int value = 33;
  FixedVec a{0, value};
  ASSERT_EQ(*a.crbegin(), value);
}

TEST(fixed_vector, const_rend) {
  const FixedVec a{0, 1, 2};
  ASSERT_EQ(a.rbegin() + a.size(), a.rend());
}

TEST(fixed_vector, rend) {
  FixedVec a{0, 1, 2};
  ASSERT_EQ(a.rbegin() + a.size(), a.rend());
}

TEST(fixed_vector, crend) {
  FixedVec a{0, 1, 2};
  ASSERT_EQ(a.crbegin() + a.size(), a.crend());
}

TEST(fixed_vector, insert_0) {
  std::vector<int> vec{0, 1, 2, 3};
  FixedVec a{1, 2};
  a.insert(a.begin(), 0);
  a.insert(a.end(), 3);
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, insert_1) {
  std::vector<int> vec{0, 1, 2, 3};
  FixedVec a{1, 2};
  int zero = 0;
  int three = 3;
  a.insert(a.begin(), std::move(zero));
  a.insert(a.end(), std::move(three));
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, insert_2) {
  std::vector<int> vec{0, 0, 1, 2, 3, 3};
  FixedVec a{1, 2};
  a.insert(a.begin(), 2, 0);
  a.insert(a.end(), 2, 3);
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, insert_3) {
  std::vector<int> vec{0, 0, 1, 2, 3, 3};
  FixedVec a{1, 2};
  int zero = 0;
  int three = 3;
  a.insert(a.begin(), 2, std::move(zero));
  a.insert(a.end(), 2, std::move(three));
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, insert_4) {
  std::vector<int> vec{0, 0, 1, 2, 3, 3};
  FixedVec a{1, 2};
  std::vector<int> zeros{0, 0};
  std::vector<int> threes{3, 3};
  a.insert(a.begin(), zeros.begin(), zeros.end());
  a.insert(a.end(), threes.begin(), threes.end());
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, insert_5) {
  std::vector<int> vec{0, 0, 1, 2, 3, 3};
  FixedVec a{1, 2};
  a.insert(a.begin(), {0, 0});
  a.insert(a.end(), {3, 3});
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, emplace) {
  std::vector<int> vec{0, 1, 2, 3};
  FixedVec a{1, 2};
  a.emplace(a.begin(), 0);
  a.emplace(a.end(), 3);
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, erase_0) {
  std::vector<int> vec{1, 2};
  FixedVec a{0, 1, 2, 3};
  a.erase(a.begin());
  a.erase(a.end() - 1);
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, erase_1) {
  std::vector<int> vec{1, 2};
  FixedVec a{0, 0, 1, 2, 3, 3};
  a.erase(a.begin(), a.begin() + 2);
  a.erase(a.end() - 2, a.end());
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, push_back_0) {
  std::vector<int> vec{0, 1, 2, 3};
  FixedVec a{0, 1, 2};
  a.push_back(3);
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, push_back_1) {
  std::vector<int> vec{0, 1, 2, 3};
  FixedVec a{0, 1, 2};
  int three = 3;
  a.push_back(std::move(three));
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, emplace_back) {
  std::vector<int> vec{0, 1, 2, 3};
  FixedVec a{0, 1, 2};
  a.emplace_back(3);
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, pop_back) {
  std::vector<int> vec{0, 1, 2};
  FixedVec a{0, 1, 2, 3};
  a.pop_back();
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, resize_0) {
  std::vector<int> vec{0, 1, 2};
  FixedVec a{0, 1, 2};
  a.resize(3);
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, resize_1) {
  std::vector<int> vec{0, 1, 2};
  FixedVec a{0, 1, 2};
  a.resize(3, 9527);
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, resize_2) {
  std::vector<int> vec{0};
  FixedVec a{0, 1, 2};
  a.resize(1);
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, resize_3) {
  std::vector<int> vec{0};
  FixedVec a{0, 1, 2};
  a.resize(1, 9527);
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, resize_4) {
  std::vector<int> vec{0, 1, 2, 0, 0};
  FixedVec a{0, 1, 2};
  a.resize(5);
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, resize_5) {
  std::vector<int> vec{0, 1, 2, 3, 3};
  FixedVec a{0, 1, 2};
  a.resize(5, 3);
  ASSERT_TRUE(std::equal(a.begin(), a.end(), vec.begin()));
}

TEST(fixed_vector, swap) {
  std::vector<int> vec_0{0, 1, 2, 0, 0};
  std::vector<int> vec_1{0, 1, 2, 3, 3};
  FixedVec a_0(vec_1.begin(), vec_1.end());
  FixedVec a_1(vec_0.begin(), vec_0.end());
  a_0.swap(a_1);
  ASSERT_TRUE(std::equal(a_0.begin(), a_0.end(), vec_0.begin()));
  ASSERT_TRUE(std::equal(a_1.begin(), a_1.end(), vec_1.begin()));
}

void WithTwoVector(std::function<void(const std::vector<int>&, const std::vector<int>&)> Handler) {
  std::vector<int> a{0, 1, 2, 3, 4};
  std::vector<int> b{0, 1, 2, 3};
  std::vector<int> c{4, 3, 2};
  Handler(a, a);
  Handler(a, b);
  Handler(a, c);
  Handler(b, a);
  Handler(b, b);
  Handler(b, c);
  Handler(c, a);
  Handler(c, b);
  Handler(c, c);
}

#define TEST_LOGICAL_OPERATOR(test_name, logical_op)                                             \
  TEST(fixed_vector, test_name) {                                                                \
    WithTwoVector([](const std::vector<int>& lhs, const std::vector<int>& rhs) {                 \
      ASSERT_EQ((lhs logical_op rhs),                                                            \
                (FixedVec(lhs.begin(), lhs.end()) logical_op FixedVec(rhs.begin(), rhs.end()))); \
    });                                                                                          \
  }

TEST_LOGICAL_OPERATOR(eq, ==);
TEST_LOGICAL_OPERATOR(ne, !=);
TEST_LOGICAL_OPERATOR(gt, >);
TEST_LOGICAL_OPERATOR(ge, >=);
TEST_LOGICAL_OPERATOR(lt, <);
TEST_LOGICAL_OPERATOR(le, <=);

}  // namespace test

}  // namespace oneflow
