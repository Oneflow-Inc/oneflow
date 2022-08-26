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

#include "oneflow/core/linear_programming/linear_programming_util.h"
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {
namespace linear_programming {
void SparsePrimalMatrix::Insert(int32_t i, int32_t j, double val) {
  CHECK_LT(original_column_size_, 0) << "The matrix is fixed, not more insertion is permitted.";
  rows_[i][j] = val;
  columns_[j][i] = val;
}

void SparsePrimalMatrix::ExpandArtificialVariables() {
  if (original_column_size_ < 0) {
    double original_column = columns_.size();
    for (int32_t i = 0; i < rows_.size(); i++) { Insert(i, i + original_column_size_, 1.0); }
    original_column_size_ = original_column;
    columns_all2compact_.resize(columns_.size());
  }
  for (int32_t j = original_column_size_; j < columns_all2compact_.size(); j++) {
    columns_all2compact_[j] = 1;
  }
}

void SparsePrimalMatrix::EliminateArtificialVariables() {
  for (int32_t j = original_column_size_; j < columns_all2compact_.size(); j++) { HideColumn(j); }
}

void SparsePrimalMatrix::HideRow(int32_t i) { rows_all2compact_[i] = -1; }

void SparsePrimalMatrix::HideColumn(int32_t j) { columns_all2compact_[j] = -1; }

void SparsePrimalMatrix::InitPrimalMatrix() {
  auto InitPrimalDimension = [](int32_t* primal_size, std::vector<int32_t>& all2compact,
                                std::vector<int32_t>& compact2all) {
    *primal_size = 0;
    for (int32_t i = 0; i < all2compact.size(); i++) {
      if (all2compact[i] >= 0) {
        all2compact[i] = *primal_size;
        (*primal_size)++;
        compact2all[*primal_size] = i;
      }
    }
    compact2all.resize(*primal_size);
  };
  InitPrimalDimension(&primal_row_size_, rows_all2compact_, rows_compact2all_);
  InitPrimalDimension(&primal_column_size_, columns_all2compact_, columns_compact2all_);
}

}  // namespace linear_programming
}  // namespace oneflow