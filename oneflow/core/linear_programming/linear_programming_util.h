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

// How the problem is solved.
enum SolveLpTag : int {
  kInit = 1,        // 1: Have not solve the problem yet
  kFiniteCost = 2,  // 2: find a finite optimal cost
  kInfCost = 3,     // 3: find the optimal cost to be -Inf.
  kNoSolution = 4   // 4: No feasible basic solution
};

static const double floating_point_error = 1e-14;

// A sparse matrix with row major
class SparseMatrix {
 public:
  std::vector<HashMap<int32_t, double>> rows_;
  int32_t column_size_ = 0;

  SparseMatrix() = default;
  SparseMatrix(int32_t row_size, int32_t column_size);
  ~SparseMatrix() = default;

  void SetValue(int32_t i, int32_t j, double val);

  void Eye(int32_t n);

  // p = c{basis} * this_sparse_matrix
  // Specifically, basis would be basic_column2compact_primal_column_.
  void VectorMatrixMultiplication(const std::vector<double>& c, const std::vector<int32_t>& basis,
                                  std::vector<double>& p) const;
  // u = this_sparse_matrix * a{all2compact}
  void MatrixVectorMultiplication(const HashMap<int32_t, double>& a,
                                  const std::vector<int32_t>& all2compact,
                                  std::vector<double>& u) const;
};

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

// Use the two-phase revised simplex method to solve linear programming problem
// Consider the standard form problem:
//      minimize cx
//      subject to Ax = b
//                  x >= 0.
class LinearProgrammingSolver {
 public:
  // primal matrix A
  SparsePrimalMatrix primal_matrix_;
  // inverse base matrix B
  SparseMatrix inverse_base_matrix_;
  // row vector c
  std::vector<double> primal_cost_;
  std::vector<double> c_;
  // column vector b, and variables x with all x_i >= 0
  std::vector<double> primal_constrain_b_;
  std::vector<double> x_;
  // row vector, p_ = basis_cost * inverse_base_matrix_
  std::vector<double> p_;
  // column vector u, u = inverse_base_matrix_ * A_j
  std::vector<double> u_;
  // A map from the basic columns to the compact primal column
  std::vector<int32_t> basic_column2compact_primal_column_;
  // A map from the compact primal column to the basic columns
  std::vector<int32_t> compact_primal_column2basic_column_;
  // The floating point error cause by addition and subtraction
  double zero_minus_ = 0.0;
  double zero_plus_ = 0.0;
  // How the linear programming problem is solved
  SolveLpTag is_solved = SolveLpTag::kInit;

  // Phase 1, solve for a initial feasible solution and corresponding basis.
  void Solve4InitFeasibleSolution();

  // the optimal cost of the primal linear programming problem
  double OptimalCost();

 private:
  // The revised simplex method
  void RevisedSimplexMethod();

  // Compute absolute error for 0
  void ComputeAbsoluteError0();

  // Numerically less than zero, x < 0
  bool NumericalLT0(double x);
  // Numerically greater than zero, x > 0
  bool NumericalGT0(double x);
};

}  // namespace linear_programming
}  // namespace oneflow

#endif  // ONEFLOW_CORE_LINEAR_PROGRAMMING_LINEAR_PROGRAMMING_UTIL_H_