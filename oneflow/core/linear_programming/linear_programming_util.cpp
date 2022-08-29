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
#include <cmath>
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {
namespace linear_programming {

// a{basis}' * b
double SparseInnerProduct(const std::vector<double>& a, const std::vector<int32_t>& basis,
                          const HashMap<int32_t, double>& b) {
  double product = 0.0;
  for (const auto& index2val : b) { product += a[basis[index2val.first]] * index2val.second; }
  return product;
}

// a' * b{all2compact}
double SparseInnerProduct(const std::vector<double>& a, const HashMap<int32_t, double>& b,
                          const std::vector<int32_t>& all2compact) {
  double product = 0.0;
  for (const auto& index2val : b) {
    int32_t k = all2compact[index2val.first];
    if (k >= 0) { product += a[k] * index2val.second; }
  }
  return product;
}

SparseMatrix::SparseMatrix(int32_t row_size, int32_t column_size) {
  rows_.reserve(row_size);
  columns_.reserve(column_size);
}

void SparseMatrix::Insert(int32_t i, int32_t j, double val) {
  if (i >= rows_.size()) { rows_.resize(i + 1); }
  rows_[i][j] = val;
  if (j >= columns_.size()) { columns_.resize(j + 1); }
  columns_[j][i] = val;
}

void SparseMatrix::Eye(int32_t n) {
  rows_.clear();
  columns_.clear();
  for (int32_t i = n - 1; i >= 0; i--) { Insert(i, i, 1.0); }
}

// p = c{basis} * this_sparse_matrix
// Specifically, basis would be basic_column2compact_primal_column_.
void SparseMatrix::VectorMatrixMultiplication(const std::vector<double>& c,
                                              const std::vector<int32_t>& basis,
                                              std::vector<double>& p) {
  p.resize(columns_.size());
  for (int32_t j = 0; j < p.size(); j++) { p[j] = SparseInnerProduct(c, basis, columns_[j]); }
}

void SparsePrimalMatrix::ExpandArtificialVariables() {
  if (original_column_size_ < 0) {
    original_column_size_ = original_matrix_.columns_.size();
    for (int32_t i = 0; i < original_matrix_.rows_.size(); i++) {
      original_matrix_.Insert(i, i + original_column_size_, 1.0);
    }
    columns_all2compact_.resize(original_matrix_.columns_.size());
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

// The revised simplex method
void LinearProgrammingSolver::RevisedSimplexMethod() {
  while (1) {
    inverse_base_matrix_.VectorMatrixMultiplication(c_, basic_column2compact_primal_column_, p_);
    for (int32_t compact_primal_column = 0;
         compact_primal_column < compact_primal_column2basic_column_.size();
         compact_primal_column++) {
      // Not a basic column
      if (compact_primal_column2basic_column_[compact_primal_column] < 0) {
        int32_t all_primal_column = primal_matrix_.columns_compact2all_[compact_primal_column];
        // c_bar_j = c_j - p' * A_j
        double cost_difference =
            c_[compact_primal_column]
            - SparseInnerProduct(p_, primal_matrix_.original_matrix_.columns_[all_primal_column],
                                 primal_matrix_.columns_all2compact_);
        if (NumericalLT0(cost_difference)) {
          // Compute u = B * A_j
        }
      }
    }
  }
}

// Phase 1, solve for a initial feasible solution and corresponding basis.
void LinearProgrammingSolver::Solve4InitFeasibleSolution() {
  // Introduce artificial variables
  primal_matrix_.ExpandArtificialVariables();
  primal_matrix_.InitPrimalMatrix();
  // Set up the auxiliary problem with cost sum(y_i)
  const auto& rows_compact2all = primal_matrix_.rows_compact2all_;
  int32_t compact_column_size = primal_matrix_.columns_compact2all_.size();
  int32_t compact_row_size = rows_compact2all.size();
  int32_t compact_primal_column_size = compact_column_size - compact_row_size;
  c_.resize(compact_column_size);
  for (int32_t j = 0; j < compact_primal_column_size; j++) { c_[j] = 0; }
  for (int32_t j = compact_primal_column_size; j < compact_column_size; j++) { c_[j] = 1; }
  // Set up the right hand size of the constrains
  x_.resize(compact_row_size);
  for (int32_t i = 0; i < compact_row_size; i++) {
    x_[i] = primal_constrain_b_[rows_compact2all[i]];
  }
  // Initialize the inverse base matrix B = I
  inverse_base_matrix_.Eye(compact_row_size);
  // Initialize the mapping between the compact primal column and the basic column
  basic_column2compact_primal_column_.resize(compact_row_size);
  compact_primal_column2basic_column_.resize(compact_column_size);
  for (int32_t j = 0; j < compact_row_size; j++) {
    int32_t artificial_column = j + compact_primal_column_size;
    basic_column2compact_primal_column_[j] = artificial_column;
    compact_primal_column2basic_column_[artificial_column] = j;
  }
  for (int32_t j = 0; j < compact_primal_column_size; j++) {
    compact_primal_column2basic_column_[j] = -1;
  }
  // Deal with potential floating point error
  ComputeAbsoluteError0();
  // Apply the revised simplex method to the auxiliary problem
}

// Compute absolute error for 0
void LinearProgrammingSolver::ComputeAbsoluteError0() {
  double max_abs_val = 0.0;
  for (int32_t r : primal_matrix_.rows_compact2all_) {
    for (const auto& column2val : primal_matrix_.original_matrix_.rows_[r]) {
      max_abs_val = std::max(max_abs_val, abs(column2val.second));
    }
  }
  zero_plus_ = max_abs_val * floating_point_error;
  zero_minus_ = -zero_plus_;
}

// Numerically less than zero, x < 0
bool LinearProgrammingSolver::NumericalLT0(double x) { return x < zero_minus_; }

}  // namespace linear_programming
}  // namespace oneflow