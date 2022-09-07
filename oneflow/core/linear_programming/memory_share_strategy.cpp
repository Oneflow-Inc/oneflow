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

#include "oneflow/core/linear_programming/memory_share_strategy.h"
#include <glog/logging.h>
#include <algorithm>
#include "oneflow/core/register/runtime_register_desc.h"

namespace oneflow {
namespace linear_programming {

void MemoryShareStrategy::ConstructMip4MemoryOnly(
    const HashMap<RegstDescProto*, std::vector<RegstDescProto*>>& regst2mutual_exclusion_regsts) {
  // Initialization
  InitRegister(regst2mutual_exclusion_regsts);
  InitRegisterInformation();

  int32_t total_register_num = regst2mutual_exclusion_regsts.size();
  int32_t row = 0;
  auto& primal_matrix = mips_.lps_.primal_matrix_;
  auto& primal_b = mips_.lps_.primal_constrain_b_;
  // Compute total number of rows
  // Which requires that regst2mutual_exclusion_regsts is symmetric.
  // If register_i exclude register_j, then register_j exclude register_i.
  int32_t total_row = 0;
  for (const auto& register_i2exclusive_registers : regst2mutual_exclusion_regsts) {
    total_row += register_i2exclusive_registers.second.size();
  }
  total_row = total_row / 2 * 3 + total_register_num;
  primal_b.resize(total_row);

  // Assemble x_i + l_i <= z
  AssembleZ(&row);

  // Assemble x_i + l_i <= x_j + M * t1_ij
  //          x_j + l_j <= x_i + M * t2_ij
  //          t1_ij + t2_ij = 1
  //          t1_ij, t2_ij belongs to {0, 1}
  // Transfer to
  //          -x_i + x_j + M * t1_ij - s1_ij = l_i
  //          x_i - x_j + M * t2_ij - s2_ij = l_j
  // Suppose each pair (i, j) s.t. i<j, is corresponding to an index
  // t1_ij: 2*total_register_num + 4*index + 1
  // t2_ij: 2*total_register_num + 4*index + 2
  // s1_ij: 2*total_register_num + 4*index + 3
  // s2_ij: 2*total_register_num + 4*index + 4
  start_row_exclusion = row;
  start_column_exclusion = 2 * total_register_num + 1;
  int32_t column = start_column_exclusion;
  for (const auto& register_i2exclusive_registers : regst2mutual_exclusion_regsts) {
    int32_t i = register2index_[register_i2exclusive_registers.first];
    for (const auto& register_j : register_i2exclusive_registers.second) {
      int32_t j = register2index_[register_j];
      if (i < j) {
        // -x_i + x_j + M * t1_ij - s1_ij = l_i
        primal_matrix.Insert(row, i + 1, -1.0);       // -x_i
        primal_matrix.Insert(row, j + 1, 1.0);        // x_j
        primal_matrix.Insert(row, column, large_m_);  // M * t1_ij
        primal_matrix.Insert(row, column + 2, -1.0);  // -s1_ij
        primal_b[row] = register_size_[i];            // l_i
        row++;

        // x_i - x_j + M * t2_ij - s2_ij = l_j
        primal_matrix.Insert(row, i + 1, 1.0);            // x_i
        primal_matrix.Insert(row, j + 1, -1.0);           // -x_j
        primal_matrix.Insert(row, column + 1, large_m_);  // M * t2_ij
        primal_matrix.Insert(row, column + 3, -1.0);      // -s2_ij
        primal_b[row] = register_size_[j];                // l_j
        row++;

        // t1_ij + t2_ij = 1
        primal_matrix.Insert(row, column, 1.0);      // t1_ij
        primal_matrix.Insert(row, column + 1, 1.0);  // t2_ij
        primal_b[row] = 1.0;                         // 1
        row++;

        column += 4;
      }
    }
  }
  end_row_exclusion = row;
  end_column_exclusion = column;

  CHECK_EQ(row, primal_b.size())
      << "Inconsistent rows, bug occurs while assembling mix-integer programming";

  // Assemble cost for minimizing z
  MinimizeZ();
}

void MemoryShareStrategy::ExportMemoryOffsets(
    size_t* mem_block_size, HashMap<RegstDescProto*, int64_t>* regst_desc2offset) {
  *mem_block_size = mem_block_size_;
  for (const auto& pair : register2index_) {
    (*regst_desc2offset)[pair.first] = register_size_[pair.second];
  }
}

void MemoryShareStrategy::StealPosition(
    const HashMap<RegstDescProto*, int64_t>& regst_desc2offset) {
  auto& primal_matrix = mips_.lps_.primal_matrix_;
  primal_matrix.ActivateAllRowColumns();
  for (int32_t start_row = start_row_exclusion; start_row < end_row_exclusion;
       start_row += num_row_group) {
    int32_t i = -1;
    int32_t j = -1;
    // -x_i + x_j + M * t1_ij - s1_ij = l_i
    for (const auto& pair : primal_matrix.rows_[start_row]) {
      if (pair.first < start_column_exclusion) {
        if (pair.second < 0.0) {
          i = pair.first - 1;  // -x_i, i+1 -> -1.0
        } else {
          j = pair.first - 1;  // x_j, j+1 -> 1.0
        }
      }
    }
    if (regst_desc2offset.at(index2register_[i]) < regst_desc2offset.at(index2register_[j])) {
      // x_i < x_j --> x_i + l_i <= x_j
      // Eliminate the other row
      primal_matrix.HideRow(start_row + 1);
    } else {
      // x_j <= x_i --> x_j + l_j <= x_i
      // Eliminate the other row
      primal_matrix.HideRow(start_row);
    }
    // Eliminate the row: t1_ij + t2_ij = 1
    primal_matrix.HideRow(start_row + 2);
    // Eliminate the columns: t1_ij, t2_ij and the corresponding artificial variable
    for (const auto& pair : primal_matrix.rows_[start_row + num_row_group - 1]) {
      primal_matrix.HideColumn(pair.first);
    }
  }
}

// Initialization
void MemoryShareStrategy::InitRegister(
    const HashMap<RegstDescProto*, std::vector<RegstDescProto*>>& regst2mutual_exclusion_regsts) {
  int32_t total_register_num = regst2mutual_exclusion_regsts.size();
  index2register_.resize(total_register_num);
  int32_t register_id = 0;
  for (const auto& pair : regst2mutual_exclusion_regsts) {
    index2register_[register_id] = pair.first;
    register_id++;
  }
}

void MemoryShareStrategy::InitRegisterInformation() {
  int32_t total_register_num = index2register_.size();
  register_size_.resize(total_register_num);
  int64_t large_m = 1;
  for (int32_t register_id = 0; register_id < total_register_num; register_id++) {
    const auto& register_ = index2register_[register_id];
    int64_t register_size = RtRegstDesc(*register_).TotalMainByteSize4AllRegst();
    register_size_[register_id] = register_size;
    register2index_[register_] = register_id;
    large_m += register_size;
  }
  large_m_ = double(large_m);
}

void MemoryShareStrategy::StealCompactPosition(
    const HashMap<RegstDescProto*, int64_t>& regst_desc2offset,
    const HashMap<RegstDescProto*, std::vector<RegstDescProto*>>& regst2mutual_exclusion_regsts) {
  // Initialization
  InitRegister(regst2mutual_exclusion_regsts);
  int32_t total_register_num = regst2mutual_exclusion_regsts.size();

  // Sort index2register_
  std::sort(index2register_.begin(), index2register_.end(),
            [&](RegstDescProto* i, RegstDescProto* j) {
              return regst_desc2offset.at(i) < regst_desc2offset.at(j);
            });
  // Update other information
  InitRegisterInformation();

  left_registers.clear();
  left_registers.resize(total_register_num);
  // should_visit[i] indicates whether we should visit register[order[i]].
  std::vector<bool> should_visit(total_register_num, false);
  // Find all the k < i, eliminates k < j,
  // since k < i and i < j have already implied that.
  std::function<void(int32_t)> EliminateRedundantRelationship = [&](int32_t i) {
    // If i is already eliminate, skip it.
    if (should_visit[i]) {
      // Eliminate i
      should_visit[i] = false;
      for (int32_t k : left_registers[i]) {
        // Eliminate all the k < i
        EliminateRedundantRelationship(k);
      }
    }
  };

  // Generate a compact relationship of position
  // For example we have 3 relationship: x1 < x2, x2 < x3, x1 < x3
  // We would delete the redundant relationship (x1 < x3)
  for (int32_t j = 0; j < total_register_num; j++) {
    // Init should visit with all orders of the excluded register
    for (const auto& register_i : regst2mutual_exclusion_regsts.at(index2register_[j])) {
      int32_t i = register2index_[register_i];
      // Mark all the excluded registers on the left
      if (i < j) { should_visit[i] = true; }
    }

    for (int32_t i = j - 1; i >= 0; i--) {
      if (should_visit[i]) {
        // i < j
        left_registers[j].push_back(i);
        // Find all the k < i, eliminates k < j,
        // since k < i and i < j have already implied that.
        // Also reset should_visit[i] to false,
        // since we have already visited i.
        EliminateRedundantRelationship(i);
      }
    }
  }

  // Compute total number of rows
  int32_t total_row = total_register_num;
  for (const auto& left_register : left_registers) { total_row += left_register.size(); }

  // Reserve the space
  auto& primal_matrix = mips_.lps_.primal_matrix_;
  auto& primal_b = mips_.lps_.primal_constrain_b_;
  primal_b.resize(total_row);

  int32_t row = 0;
  // Assemble x_i + l_i <= z
  AssembleZ(&row);

  // Assemble x_i + l_i <= x_j
  // Transfer to
  //          -x_i + x_j - s_ij = l_i
  // Suppose each pair (i, j) s.t. i<j, is corresponding to an index
  // s1_ij: 2*total_register_num + index (column)
  start_row_exclusion = row;
  start_column_exclusion = 2 * total_register_num + 1;
  int32_t column = start_column_exclusion;
  for (int32_t j = 0; j < total_register_num; j++) {
    for (int32_t i : left_registers[j]) {
      // -x_i + x_j - s_ij = l_i
      primal_matrix.Insert(row, i + 1, -1.0);   // -x_i
      primal_matrix.Insert(row, j + 1, 1.0);    // x_j
      primal_matrix.Insert(row, column, -1.0);  // -s_ij
      primal_b[row] = register_size_[i];        // l_i
      row++;
      column++;
    }
  }
  end_row_exclusion = row;
  end_column_exclusion = column;

  CHECK_EQ(row, primal_b.size())
      << "Inconsistent rows, bug occurs while assembling mix-integer programming";

  // Assemble cost for minimizing z
  MinimizeZ();

  // Activate all the rows and columns since it is compact already
  primal_matrix.ActivateAllRowColumns();
}

void MemoryShareStrategy::AssembleZ(int32_t* row) {
  int32_t total_register_num = register_size_.size();
  auto& primal_matrix = mips_.lps_.primal_matrix_;
  auto& primal_b = mips_.lps_.primal_constrain_b_;
  // Assemble x_i + l_i <= z,
  // where z is the total size of the memory block.
  // We need to make sure that the right hand size >= 0. We transfer the formula to
  // z - x_i - s_i = l_i
  // z: 0
  // x_i: [1, total_register_num], i+1
  // s_i: [total_register_num+1, 2*total_register_num], i+total_register_num+1
  for (int32_t i = 0; i < total_register_num; i++) {
    primal_matrix.Insert(*row, 0, 1.0);                            // z
    primal_matrix.Insert(*row, i + 1, -1.0);                       // -x_i
    primal_matrix.Insert(*row, i + total_register_num + 1, -1.0);  // -s_i
    primal_b[*row] = register_size_[i];                            // l_i
    (*row)++;
  }
}

// Assemble cost for minimizing z
void MemoryShareStrategy::MinimizeZ() {
  // Minimize z
  // c = [1, 0, 0, 0, ..., 0]
  mips_.lps_.primal_cost_.resize(end_column_exclusion);
  mips_.lps_.primal_cost_[0] = 1.0;
}

// Compute optimal cost with compact relationship
size_t MemoryShareStrategy::ComputeOptimalCost4CompactRelationship() {
  int32_t total_register_num = index2register_.size();
  register_offset_.clear();
  register_offset_.resize(total_register_num, -1);
  int64_t mem_block_size = 0;
  for (int32_t i = 0; i < total_register_num; i++) {
    mem_block_size =
        std::max(mem_block_size, ComputeOffset4CompactRelationship(i) + register_size_[i]);
  }
  mem_block_size_ = size_t(mem_block_size);
  return mem_block_size_;
}

// Compute offset with compact relationship
int64_t MemoryShareStrategy::ComputeOffset4CompactRelationship(int32_t i) {
  if (register_offset_[i] < 0) {
    if (left_registers[i].size() == 0) {
      register_offset_[i] = 0;
    } else {
      for (int32_t j : left_registers[i]) {
        register_offset_[i] =
            std::max(register_offset_[i], ComputeOffset4CompactRelationship(j) + register_size_[j]);
      }
    }
  }
  return register_offset_[i];
}

}  // namespace linear_programming
}  // namespace oneflow