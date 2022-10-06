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
#include "oneflow/core/intrusive/static_counter.h"
#include "oneflow/core/intrusive/intrusive.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

namespace {

DEFINE_STATIC_COUNTER(static_counter);

static_assert(STATIC_COUNTER(static_counter) == 0, "");

TEST(StaticCounter, eq0) { static_assert(STATIC_COUNTER(static_counter) == 0, ""); }

INCREASE_STATIC_COUNTER(static_counter);

static_assert(STATIC_COUNTER(static_counter) == 1, "");

TEST(StaticCounter, eq1) { static_assert(STATIC_COUNTER(static_counter) == 1, ""); }

static_assert(STATIC_COUNTER(static_counter) == 1, "");

TEST(StaticCounter, eq1_again) { static_assert(STATIC_COUNTER(static_counter) == 1, ""); }

INCREASE_STATIC_COUNTER(static_counter);

static_assert(STATIC_COUNTER(static_counter) == 2, "");

TEST(StaticCounter, eq2) { static_assert(STATIC_COUNTER(static_counter) == 2, ""); }

// clang-format off
REFLECTIVE_CLASS_BEGIN(FooBar);
  FooBar() = default;
  static_assert(STATIC_COUNTER(field_counter) == 0, "");
REFLECTIVE_CLASS_END(FooBar);
// clang-format on

}  // namespace

}  // namespace test

}  // namespace oneflow
