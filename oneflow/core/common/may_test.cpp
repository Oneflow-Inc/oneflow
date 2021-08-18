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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/job/sbp_parallel.h"

namespace oneflow {
namespace test {
namespace {

Maybe<void> test_check_eq_or_return() {
  Symbol<DType> dtype = JUST(DType::Get(DataType::kFloat));
  DataType data_type = DataType::kFloat;
  Symbol<cfg::SbpParallel> sbp = JUST(MakeBroadcastSbpParallel());
  if (data_type == dtype) {
    // do nothing
  }
  if (data_type == sbp) {
    // do nothing
  }
  CHECK_EQ_OR_RETURN(dtype, data_type);
  return Maybe<void>::Ok();
}

}  // namespace

TEST(Maybe, CHECK_EQ_OR_RETURN) {
  CHECK_JUST(test_check_eq_or_return());
  // auto may_ret = test_check_eq_or_return();

  // ASSERT_EQ(may_ret.IsOk(), true);
}

}  // namespace test
}  // namespace oneflow
