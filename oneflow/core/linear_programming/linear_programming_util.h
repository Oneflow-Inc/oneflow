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

#ifndef ONEFLOW_CORE_LINEAR_PROGRAMMING_LINEAR_PROGRAMMING_UTIL_H_
#define ONEFLOW_CORE_LINEAR_PROGRAMMING_LINEAR_PROGRAMMING_UTIL_H_

#include <cstdlib>
#include <vector>
#include "oneflow/core/common/hash_container.h"

namespace oneflow {
namespace linear_programming {
// A sparse matrix whose rows and columns might be eliminated.
class SparsePrimalMatrix {
 public:
  std::vector<HashMap<int32_t, double>> rows_;
  std::vector<HashMap<int32_t, double>> columns_;
  std::vector<int32_t> rows_all2compact_;
  std::vector<int32_t> columns_all2compact_;
  std::vector<int32_t> rows_compact2all_;
  std::vector<int32_t> columns_compact2all_;

  int32_t original_column_size_ = -1;
  int32_t primal_row_size_ = -1;
  int32_t primal_column_size_ = -1;

  SparsePrimalMatrix() = default;
  ~SparsePrimalMatrix() = default;

  void Insert(int32_t i, int32_t j, double val);

  void ExpandArtificialVariables();
  void EliminateArtificialVariables();

  void HideRow(int32_t i);
  void HideColumn(int32_t j);
  void InitPrimalMatrix();
};
}  // namespace linear_programming
}  // namespace oneflow

#endif  // ONEFLOW_CORE_LINEAR_PROGRAMMING_LINEAR_PROGRAMMING_UTIL_H_