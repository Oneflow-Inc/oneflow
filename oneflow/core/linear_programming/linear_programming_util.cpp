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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {
namespace linear_programming {

namespace {
// a{basis}' * b
double InnerProduct(const std::vector<double>& a, const std::vector<int32_t>& basis,
                    const std::vector<double>& b) {
  double product = 0.0;
  for (int32_t k = 0; k < basis.size(); k++) { product += a[basis[k]] * b[k]; }
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

// a' * b{all2compact}
double SparseInnerProduct(const HashMap<int32_t, double>& a, const HashMap<int32_t, double>& b,
                          const std::vector<int32_t>& all2compact) {
  double product = 0.0;
  for (const auto& index2val : b) {
    int32_t k = all2compact[index2val.first];
    if (k >= 0) {
      const auto& it = a.find(k);
      if (it != a.end()) { product += it->second * index2val.second; }
    }
  }
  return product;
}

// Elementary row operation: division
// a = a/u
void ElementaryRowOperationDivision(HashMap<int32_t, double>& a, double u) {
  for (auto& index2val : a) { index2val.second /= u; }
}

// Elementary row operation: subtraction
// b = b - a*u
void ElementaryRowOperationSbutraction(HashMap<int32_t, double>& b,
                                       const HashMap<int32_t, double>& a, double u,
                                       double safeguard) {
  if (abs(u) < safeguard) { return; }
  for (const auto& a_index2val : a) {
    auto it = b.find(a_index2val.first);
    if (it != b.end()) {
      it->second -= a_index2val.second * u;
      if (abs(it->second) < safeguard) { b.erase(it); }
    } else {
      b[a_index2val.first] = -a_index2val.second * u;
    }
  }
}

}  // namespace

SparseMatrix::SparseMatrix(int32_t row_size, int32_t column_size) { rows_.reserve(row_size); }

void SparsePrimalMatrix::Insert(int32_t i, int32_t j, double val) {
  if (i >= rows_.size()) { rows_.resize(i + 1); }
  rows_[i][j] = val;
  if (j >= columns_.size()) { columns_.resize(j + 1); }
  columns_[j][i] = val;
}

void SparseMatrix::SetValue(int32_t i, int32_t j, double val) {
  if (i >= rows_.size()) { rows_.resize(i + 1); }
  if (j >= column_size_) { column_size_ = j + 1; }
  rows_[i][j] = val;
}

void SparseMatrix::Eye(int32_t n) {
  rows_.clear();
  column_size_ = n;
  for (int32_t i = n - 1; i >= 0; i--) { SetValue(i, i, 1.0); }
}

// p = c{basis} * this_sparse_matrix
// Specifically, basis would be basic_column2compact_primal_column_.
void SparseMatrix::VectorMatrixMultiplication(const std::vector<double>& c,
                                              const std::vector<int32_t>& basis,
                                              std::vector<double>& p) const {
  p.clear();
  p.resize(column_size_, 0);
  for (int32_t i = 0; i < rows_.size(); i++) {
    for (const auto& column2val : rows_[i]) {
      p[column2val.first] += c[basis[i]] * column2val.second;
    }
  }
}

// u = this_sparse_matrix * a{all2compact}
void SparseMatrix::MatrixVectorMultiplication(const HashMap<int32_t, double>& a,
                                              const std::vector<int32_t>& all2compact,
                                              std::vector<double>& u) const {
  u.resize(rows_.size());
  for (int32_t i = 0; i < u.size(); i++) { u[i] = SparseInnerProduct(rows_[i], a, all2compact); }
}

void SparsePrimalMatrix::ExpandArtificialVariables() {
  if (original_column_size_ < 0) {
    original_column_size_ = columns_.size();
    for (int32_t i = 0; i < rows_.size(); i++) { Insert(i, i + original_column_size_, 1.0); }
    columns_all2compact_.resize(columns_.size());
  }
  for (int32_t j = original_column_size_; j < columns_all2compact_.size(); j++) {
    columns_all2compact_[j] = 1;
  }
}

void SparsePrimalMatrix::EliminateArtificialVariables() {
  for (int32_t j = original_column_size_; j < columns_all2compact_.size(); j++) { HideColumn(j); }
}

void SparsePrimalMatrix::HideRow(int32_t i) {
  rows_all2compact_[i] = -1;
  for (const auto& column2val : rows_[i]) {
    if (std::all_of(columns_[column2val.first].begin(), columns_[column2val.first].end(),
                    [&](const std::pair<const int32_t, double>& row2val) {
                      return rows_all2compact_[row2val.first] < 0;
                    })) {
      // Not using HideColumn to avoid recursive Hiding
      columns_all2compact_[column2val.first] = -1;
    }
  }
}

void SparsePrimalMatrix::HideColumn(int32_t j) { columns_all2compact_[j] = -1; }

void SparsePrimalMatrix::InitPrimalMatrix() {
  auto InitPrimalDimension = [](int32_t* primal_size, std::vector<int32_t>& all2compact,
                                std::vector<int32_t>& compact2all) {
    *primal_size = 0;
    compact2all.resize(all2compact.size());
    for (int32_t i = 0; i < all2compact.size(); i++) {
      if (all2compact[i] >= 0) {
        all2compact[i] = *primal_size;
        compact2all[*primal_size] = i;
        (*primal_size)++;
      }
    }
    compact2all.resize(*primal_size);
  };
  InitPrimalDimension(&primal_row_size_, rows_all2compact_, rows_compact2all_);
  InitPrimalDimension(&primal_column_size_, columns_all2compact_, columns_compact2all_);
}

void SparsePrimalMatrix::ActivateAllRowColumns() {
  auto ActivateDimension = [](int32_t size, std::vector<int32_t>& all2compact,
                              std::vector<int32_t>& compact2all) {
    all2compact.resize(size);
    compact2all.resize(size);
    for (int32_t i = 0; i < size; i++) {
      all2compact[i] = i;
      compact2all[i] = i;
    }
  };
  ActivateDimension(rows_.size(), rows_all2compact_, rows_compact2all_);
  ActivateDimension(columns_.size(), columns_all2compact_, columns_compact2all_);
}

// The revised simplex method
void LinearProgrammingSolver::RevisedSimplexMethod() {
  while (1) {
    // Whether we found the negative c_bar_j
    bool not_found_negative_c_bar = true;
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
            - SparseInnerProduct(p_, primal_matrix_.columns_[all_primal_column],
                                 primal_matrix_.rows_all2compact_);
        if (NumericalLT0(cost_difference)) {
          not_found_negative_c_bar = false;
          // Compute u = B * A_j
          ComputeU4ColumnJ(all_primal_column);

          // Record the existing variable x_basis{i}, and corresponding theta = x_basis{i}/u_i
          double min_theta = -1.0;
          int32_t min_i = -1;
          // NOTE: Brand's Rule, the order of index really matter.
          for (int32_t i : compact_primal_column2basic_column_) {
            if (i >= 0) {
              if (NumericalGT0(u_[i])) {
                double theta = x_[i] / u_[i];
                if (min_i == -1 || min_theta > theta + zero_plus_) {
                  min_theta = theta;
                  min_i = i;
                }
              }
            }
          }
          // Or this one?
          // for (int32_t i = 0; i < u_.size(); i++) {
          //   if (NumericalGT0(u_[i])) {
          //     double theta = x_[i] / u_[i];
          //     if (min_i == -1 || min_theta > theta + zero_plus_) {
          //       min_theta = theta;
          //       min_i = i;
          //     }
          //   }
          // }
          // If all the u_i >= 0, the algorithm terminates with the optimal cost -Inf.
          if (min_i == -1) {
            std::cout << "All the u_i <= 0, the algorithm terminates with the optimal cost -Inf."
                      << std::endl;
            is_solved = SolveLpTag::kInfCost;
            return;
          }
          // Replace the min_i-th basis variables with compact_primal_column
          ReplaceBasisVariable(min_i, compact_primal_column);

          // Move to the next step
          break;
        }
      }
    }
    // If all the c_bar_j is positive or zero, then we found a feasible solution x_
    if (not_found_negative_c_bar) { return; }
  }
}

// Compute u = B * A_j
void LinearProgrammingSolver::ComputeU4ColumnJ(int32_t j) {
  inverse_base_matrix_.MatrixVectorMultiplication(primal_matrix_.columns_[j],
                                                  primal_matrix_.rows_all2compact_, u_);
}

// Replace the l-th basis variable with x_j.
// Drive basis{l} out and get j in.
// u_ = B * A_j should be precomputed.
void LinearProgrammingSolver::ReplaceBasisVariable(int32_t l, int32_t j) {
  // Rounding u
  for (int32_t i = 0; i < u_.size(); i++) {
    if (abs(u_[i]) < zero_plus_) { u_[i] = 0.0; }
  }
  // Update x_i
  double theta = x_[l] / u_[l];
  for (int32_t i = 0; i < u_.size(); i++) {
    if (i != l) {
      x_[i] -= theta * u_[i];
    } else {
      x_[i] = theta;
    }
  }
  // Update inverse_base_matrix_
  // B[l, :] /= u_[l], u_[l] -> 1
  ElementaryRowOperationDivision(inverse_base_matrix_.rows_[l], u_[l]);
  // B[i, :] -= B[l, :] * u_[i]/u_[l], u_[i] -> 0, for all i != l
  // The order of ElementaryRowOperation() can not be reverse
  for (int32_t i = 0; i < u_.size(); i++) {
    if (i != l) {
      ElementaryRowOperationSbutraction(inverse_base_matrix_.rows_[i],
                                        inverse_base_matrix_.rows_[l], u_[i], zero_plus_);
    }
  }
  // Replace basis[l] with compact_primal_column
  compact_primal_column2basic_column_[basic_column2compact_primal_column_[l]] = -1;
  compact_primal_column2basic_column_[j] = l;
  basic_column2compact_primal_column_[l] = j;
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
  for (int32_t j = 0; j < compact_row_size; j++) {
    int32_t artificial_column = j + compact_primal_column_size;
    basic_column2compact_primal_column_[j] = artificial_column;
  }
  InitCompact2Basis(compact_column_size);

  // Deal with potential floating point error
  ComputeAbsoluteError0();
  // Apply the revised simplex method to the auxiliary problem
  RevisedSimplexMethod();
  // Drive artificial variables out of the basis
  int32_t first_remove_row = -1;
  for (int32_t l = 0; l < compact_row_size; l++) {
    // Find those artificial variables
    if (basic_column2compact_primal_column_[l] >= compact_primal_column_size) {
      bool not_found_linear_independent_column = true;
      for (int32_t j = 0; j < compact_primal_column_size; j++) {
        // Find those primal variables which is not in the basis
        if (compact_primal_column2basic_column_[j] < 0) {
          // if inverse_base_matrix[l, :] * A_j != 0
          if (NumericalGT0(
                  abs(SparseInnerProduct(inverse_base_matrix_.rows_[l], primal_matrix_.columns_[j],
                                         primal_matrix_.rows_all2compact_)))) {
            not_found_linear_independent_column = false;
            // Compute u = B * A_j
            ComputeU4ColumnJ(j);
            // Replace the l-th basis variables with x_j
            ReplaceBasisVariable(l, j);
            // Move on to the next step
            break;
          }
        }
      }
      if (not_found_linear_independent_column) {
        // Remove this row and corresponding columns
        primal_matrix_.HideRow(rows_compact2all[l]);
        // Mark the l-th row which would be removed later
        if (first_remove_row < 0) { first_remove_row = l; }
      }
    }
  }

  // Remove the marked rows and columns of the inverse_base_matrix_
  if (first_remove_row >= 0) {
    int32_t compact_row_id = first_remove_row;
    std::vector<int32_t> removed_columns;
    // Remove marked rows and mark columns
    for (int32_t l = first_remove_row; l < compact_row_size; l++) {
      if (primal_matrix_.rows_all2compact_[primal_matrix_.rows_compact2all_[l]] < 0) {
        // Mark the l-th column which would be removed later
        removed_columns.push_back(l);
      } else {
        // Move the other rows forward
        inverse_base_matrix_.rows_[compact_row_id] = inverse_base_matrix_.rows_[l];
        x_[compact_row_id] = x_[l];
        basic_column2compact_primal_column_[compact_row_id] =
            basic_column2compact_primal_column_[l];
        compact_row_id++;
      }
    }
    inverse_base_matrix_.rows_.resize(compact_row_id);
    x_.resize(compact_row_id);
    basic_column2compact_primal_column_.resize(compact_row_id);
    // Note: The artificial columns of the inverse base matrix are already removed due to the
    // sparsity. Remove marked columns

    // Adjust the position of the primal cost
    int32_t compact_column_id = 0;
    for (int32_t j = 0; j < primal_cost_.size(); j++) {
      if (primal_matrix_.columns_all2compact_[primal_matrix_.columns_compact2all_[j]] >= 0) {
        // Move the column forward
        if (compact_column_id < j) { primal_cost_[compact_column_id] = primal_cost_[j]; }
        compact_column_id++;
      }
    }
    primal_cost_.resize(compact_column_id);
    InitCompact2Basis(compact_column_id);
    // Adjust the primal matrix
    primal_matrix_.InitPrimalMatrix();
  }
}

// Compute absolute error for 0
void LinearProgrammingSolver::ComputeAbsoluteError0() {
  double max_abs_val = 0.0;
  for (int32_t r : primal_matrix_.rows_compact2all_) {
    for (const auto& column2val : primal_matrix_.rows_[r]) {
      max_abs_val = std::max(max_abs_val, abs(column2val.second));
    }
  }
  zero_plus_ = max_abs_val * floating_point_error;
  zero_minus_ = -zero_plus_;
}

// Numerically less than zero, x < 0
bool LinearProgrammingSolver::NumericalLT0(double x) { return x < zero_minus_; }
// Numerically greater than zero, x > 0
bool LinearProgrammingSolver::NumericalGT0(double x) { return x > zero_plus_; }

// the optimal cost of the primal linear programming problem
double LinearProgrammingSolver::OptimalCost() {
  if (is_solved == SolveLpTag::kFiniteCost) {
    return InnerProduct(c_, basic_column2compact_primal_column_, x_);
  }
  return -GetMaxVal<float>();
}

// Init the map from compact variables to the basic variables
void LinearProgrammingSolver::InitCompact2Basis(int32_t column_size) {
  compact_primal_column2basic_column_.clear();
  compact_primal_column2basic_column_.resize(column_size, -1);
  for (int32_t j = 0; j < basic_column2compact_primal_column_.size(); j++) {
    compact_primal_column2basic_column_[basic_column2compact_primal_column_[j]] = j;
  }
}

}  // namespace linear_programming
}  // namespace oneflow