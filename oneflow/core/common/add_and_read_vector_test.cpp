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
#include "gtest/gtest.h"
#include "oneflow/core/common/add_and_read_vector.h"

namespace oneflow {
namespace test {

TEST(AddAndReadVector, simple0) {
  AddAndReadVector<int> vec;
  ASSERT_EQ(vec.size(), 0);
}

TEST(AddAndReadVector, simple1) {
  AddAndReadVector<int> vec;
  vec.push_back(0);
  ASSERT_EQ(vec.at(0), 0);
  ASSERT_EQ(vec.size(), 1);
}

TEST(AddAndReadVector, simple2) {
  AddAndReadVector<int> vec;
  vec.push_back(0);
  ASSERT_EQ(vec.at(0), 0);
  ASSERT_EQ(vec.size(), 1);
  vec.push_back(1);
  ASSERT_EQ(vec.at(1), 1);
  ASSERT_EQ(vec.size(), 2);
}

TEST(AddAndReadVector, simple3) {
  AddAndReadVector<int> vec;
  vec.push_back(0);
  ASSERT_EQ(vec.at(0), 0);
  ASSERT_EQ(vec.size(), 1);
  vec.push_back(1);
  ASSERT_EQ(vec.at(1), 1);
  ASSERT_EQ(vec.size(), 2);
  vec.push_back(2);
  ASSERT_EQ(vec.at(2), 2);
  ASSERT_EQ(vec.size(), 3);
}

TEST(AddAndReadVector, simple4) {
  AddAndReadVector<int> vec;
  vec.push_back(0);
  ASSERT_EQ(vec.at(0), 0);
  ASSERT_EQ(vec.size(), 1);
  vec.push_back(1);
  ASSERT_EQ(vec.at(1), 1);
  ASSERT_EQ(vec.size(), 2);
  vec.push_back(2);
  ASSERT_EQ(vec.at(2), 2);
  ASSERT_EQ(vec.size(), 3);
  vec.push_back(3);
  ASSERT_EQ(vec.at(3), 3);
  ASSERT_EQ(vec.size(), 4);
}

TEST(AddAndReadVector, simple5) {
  AddAndReadVector<int> vec;
  vec.push_back(0);
  ASSERT_EQ(vec.at(0), 0);
  ASSERT_EQ(vec.size(), 1);
  vec.push_back(1);
  ASSERT_EQ(vec.at(1), 1);
  ASSERT_EQ(vec.size(), 2);
  vec.push_back(2);
  ASSERT_EQ(vec.at(2), 2);
  ASSERT_EQ(vec.size(), 3);
  vec.push_back(3);
  ASSERT_EQ(vec.at(3), 3);
  ASSERT_EQ(vec.size(), 4);
  vec.push_back(4);
  ASSERT_EQ(vec.at(4), 4);
  ASSERT_EQ(vec.size(), 5);
}

TEST(AddAndReadVector, simple6) {
  AddAndReadVector<int> vec;
  vec.push_back(0);
  ASSERT_EQ(vec.at(0), 0);
  ASSERT_EQ(vec.size(), 1);
  vec.push_back(1);
  ASSERT_EQ(vec.at(1), 1);
  ASSERT_EQ(vec.size(), 2);
  vec.push_back(2);
  ASSERT_EQ(vec.at(2), 2);
  ASSERT_EQ(vec.size(), 3);
  vec.push_back(3);
  ASSERT_EQ(vec.at(3), 3);
  ASSERT_EQ(vec.size(), 4);
  vec.push_back(4);
  ASSERT_EQ(vec.at(4), 4);
  ASSERT_EQ(vec.size(), 5);
  vec.push_back(5);
  ASSERT_EQ(vec.at(5), 5);
  ASSERT_EQ(vec.size(), 6);
}

TEST(AddAndReadVector, simple7) {
  AddAndReadVector<int> vec;
  vec.push_back(0);
  ASSERT_EQ(vec.at(0), 0);
  ASSERT_EQ(vec.size(), 1);
  vec.push_back(1);
  ASSERT_EQ(vec.at(1), 1);
  ASSERT_EQ(vec.size(), 2);
  vec.push_back(2);
  ASSERT_EQ(vec.at(2), 2);
  ASSERT_EQ(vec.size(), 3);
  vec.push_back(3);
  ASSERT_EQ(vec.at(3), 3);
  ASSERT_EQ(vec.size(), 4);
  vec.push_back(4);
  ASSERT_EQ(vec.at(4), 4);
  ASSERT_EQ(vec.size(), 5);
  vec.push_back(5);
  ASSERT_EQ(vec.at(5), 5);
  ASSERT_EQ(vec.size(), 6);
  vec.push_back(6);
  ASSERT_EQ(vec.at(6), 6);
  ASSERT_EQ(vec.size(), 7);
}

TEST(AddAndReadVector, simple8) {
  AddAndReadVector<int> vec;
  vec.push_back(0);
  ASSERT_EQ(vec.at(0), 0);
  ASSERT_EQ(vec.size(), 1);
  vec.push_back(1);
  ASSERT_EQ(vec.at(1), 1);
  ASSERT_EQ(vec.size(), 2);
  vec.push_back(2);
  ASSERT_EQ(vec.at(2), 2);
  ASSERT_EQ(vec.size(), 3);
  vec.push_back(3);
  ASSERT_EQ(vec.at(3), 3);
  ASSERT_EQ(vec.size(), 4);
  vec.push_back(4);
  ASSERT_EQ(vec.at(4), 4);
  ASSERT_EQ(vec.size(), 5);
  vec.push_back(5);
  ASSERT_EQ(vec.at(5), 5);
  ASSERT_EQ(vec.size(), 6);
  vec.push_back(6);
  ASSERT_EQ(vec.at(6), 6);
  ASSERT_EQ(vec.size(), 7);
  vec.push_back(7);
  ASSERT_EQ(vec.at(7), 7);
  ASSERT_EQ(vec.size(), 8);
}

TEST(AddAndReadVector, simple9) {
  AddAndReadVector<int> vec;
  vec.push_back(0);
  ASSERT_EQ(vec.at(0), 0);
  ASSERT_EQ(vec.size(), 1);
  vec.push_back(1);
  ASSERT_EQ(vec.at(1), 1);
  ASSERT_EQ(vec.size(), 2);
  vec.push_back(2);
  ASSERT_EQ(vec.at(2), 2);
  ASSERT_EQ(vec.size(), 3);
  vec.push_back(3);
  ASSERT_EQ(vec.at(3), 3);
  ASSERT_EQ(vec.size(), 4);
  vec.push_back(4);
  ASSERT_EQ(vec.at(4), 4);
  ASSERT_EQ(vec.size(), 5);
  vec.push_back(5);
  ASSERT_EQ(vec.at(5), 5);
  ASSERT_EQ(vec.size(), 6);
  vec.push_back(6);
  ASSERT_EQ(vec.at(6), 6);
  ASSERT_EQ(vec.size(), 7);
  vec.push_back(7);
  ASSERT_EQ(vec.at(7), 7);
  ASSERT_EQ(vec.size(), 8);
  vec.push_back(8);
  ASSERT_EQ(vec.at(8), 8);
  ASSERT_EQ(vec.size(), 9);
}

}  // namespace test
}  // namespace oneflow
