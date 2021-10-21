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
#ifndef ONEFLOW_CORE_COMMON_BALANCED_SPLITTER_H_
#define ONEFLOW_CORE_COMMON_BALANCED_SPLITTER_H_

#include <stdint.h>
#include "oneflow/core/common/range.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

// For example
// BalancedSplitter splitter(20, 6)
// the result of splitter.At
//     0    [0, 4)
//     1    [4, 8)
//     2    [8, 11)
//     3    [11, 14)
//     4    [14, 17)
//     5    [17, 20)
class BalancedSplitter final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(BalancedSplitter);
  BalancedSplitter() = delete;
  ~BalancedSplitter() = default;

  BalancedSplitter(int64_t total_num, int64_t split_num);

  Range At(int64_t idx) const;
  Range At(int64_t first_idx, int64_t last_idx) const;

 private:
  int64_t base_part_size_;
  int64_t base_begin_idx_;
  int64_t split_num_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_BALANCED_SPLITTER_H_
