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
#ifndef ONEFLOW_CORE_COMMON_EXPECT_H_
#define ONEFLOW_CORE_COMMON_EXPECT_H_

#include "oneflow/core/common/throw.h"

#define EXPECT(expr) \
  if (!(expr)) THROW(CheckFailedError)

#define EXPECT_EQ(lhs, rhs) EXPECT((lhs) == (rhs))
#define EXPECT_NE(lhs, rhs) EXPECT((lhs) != (rhs))
#define EXPECT_LT(lhs, rhs) EXPECT((lhs) < (rhs))
#define EXPECT_LE(lhs, rhs) EXPECT((lhs) <= (rhs))
#define EXPECT_GT(lhs, rhs) EXPECT((lhs) > (rhs))
#define EXPECT_GE(lhs, rhs) EXPECT((lhs) >= (rhs))
#define EXPECT_ISNULL(expr) EXPECT((expr) == nullptr)
#define EXPECT_NOTNULL(expr) EXPECT((expr) != nullptr)
#define EXPECT_STREQ(lhs, rhs) EXPECT(std::string(lhs) == std::string(rhs))
#define EXPECT_STRNE(lhs, rhs) EXPECT(std::string(lhs) != std::string(rhs))

#endif  // ONEFLOW_CORE_COMMON_EXPECT_H_
