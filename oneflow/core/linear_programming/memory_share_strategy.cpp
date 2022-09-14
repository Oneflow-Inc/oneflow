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
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"
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
  total_register_num_ = regst2mutual_exclusion_regsts.size();
  index2register_.resize(total_register_num_);
  int32_t register_id = 0;
  for (const auto& pair : regst2mutual_exclusion_regsts) {
    index2register_[register_id] = pair.first;
    register_id++;
  }
}

void MemoryShareStrategy::InitRegisterInformation() {
  total_register_num_ = index2register_.size();
  register_size_.resize(total_register_num_);
  int64_t large_m = 1;
  for (int32_t register_id = 0; register_id < total_register_num_; register_id++) {
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
  excluded_registers.clear();
  excluded_registers.resize(total_register_num);
  // should_visit[i] indicates whether we should visit register[order[i]].
  // should_visit[i] = 0: should not visit i, or have already visited i..
  // should_visit[i] = 1: should visit i, i is excluded with j
  // should_visit[i] = 2: should visit i, i is not excluded with j
  should_visit.clear();
  should_visit.resize(total_register_num, 0);
  // Find all the k < i, eliminates k < j,
  // since k < i and i < j have already implied that.
  std::function<void(int32_t)> EliminateRedundantRelationship = [&](int32_t i) {
    // If i is already eliminate, skip it.
    if (should_visit[i]) {
      // Eliminate i
      should_visit[i] = 0;
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
    auto& excluded_register_j = excluded_registers[j];
    // Init should visit with all orders of the excluded register
    for (const auto& register_i : regst2mutual_exclusion_regsts.at(index2register_[j])) {
      // Copy the data to excluded registers
      excluded_register_j.insert(register2index_[register_i]);
    }

    // Mark all the registers on the left
    for (int32_t i = 0; i < j; i++) {
      if (Exclude(i, j)) {
        should_visit[i] = 1;
      } else {
        should_visit[i] = 2;
      }
    }

    for (int32_t i = j - 1; i >= 0; i--) {
      if (should_visit[i] == 1) {
        // i < j
        left_registers[j].insert(i);
        // Find all the k < i, eliminates k < j,
        // since k < i and i < j have already implied that.
        // Also reset should_visit[i] to false,
        // since we have already visited i.
        EliminateRedundantRelationship(i);
      }
      // for should_visit[i] == 2, just clear it
      should_visit[i] = 0;
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

void MemoryShareStrategy::GenerateCompactPosition(
    const HashMap<RegstDescProto*, std::vector<RegstDescProto*>>& regst2mutual_exclusion_regsts) {
  HashMap<RegstDescProto*, int64_t> regst_desc2offset;
  int64_t offset = 0;
  for (const auto& pair : regst2mutual_exclusion_regsts) {
    regst_desc2offset[pair.first] = offset;
    offset++;
  }
  StealCompactPosition(regst_desc2offset, regst2mutual_exclusion_regsts);
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
    // An initial value x would be store as -x - 1.
    register_offset_[i] = -register_offset_[i] - 1;
    for (int32_t j : left_registers[i]) {
      register_offset_[i] =
          std::max(register_offset_[i], ComputeOffset4CompactRelationship(j) + register_size_[j]);
    }
  }
  return register_offset_[i];
}

size_t MemoryShareStrategy::ComputeOptimalAdjustedCost() {
  // Initial optimal cost
  size_t optimal_cost = ComputeOptimalCostFrom0();
  // All the registers excluded with register i are sorted from left to right
  // std::vector<int32_t> order;
  // auto CompareRegisterPosition = [&](int32_t i, int32_t j) {
  //   return register_offset_[i] < register_offset_[j];
  // };
  backup_registers.clear();
  backup_registers.resize(index2register_.size());
  // The number of steps that the optimal cost does not decrease
  int32_t step_no_decrease = 0;
  for (int32_t m = 0; m < total_register_num_; m++) {
    for (int32_t i = 0; i < index2register_.size(); i++) {
      std::cout << "i: " << i << ", step no decrease: " << step_no_decrease << std::endl;
      EliminateRegister(i);
      size_t cost_without_i = ComputeOptimalCostFrom0();
      std::cout << "Get rid of " << i << ", size: " << register_size_[i]
                << ", Guess cost: " << cost_without_i << std::endl;
      // Find the offset of i which has the minimum cost
      int64_t min_x_i = -1;
      if (cost_without_i < optimal_cost) {
        // Find the minimum cost
        int64_t min_cost = optimal_cost;
        // const auto& excluded_register = excluded_registers[i];
        // // Sort all the excluded registers.
        // order.resize(excluded_register.size());
        // int32_t index = 0;
        // for (int32_t k : excluded_register) {
        //   order[index] = k;
        //   index++;
        // }
        // std::sort(order.begin(), order.end(), CompareRegisterPosition);
        // Back up the current register offset with elimination of i
        auto register_offset_backup = register_offset_;
        // Try to insert the register i into the sorted excluded registers
        HashSet<int64_t> all_x_i;
        for (int32_t j : excluded_registers[i]) {
          // Insert i before j
          all_x_i.insert(register_offset_backup[j]);
          // Insert i after j
          all_x_i.insert(register_offset_backup[j] + register_size_[j]);
        }

        for (int64_t x_i : all_x_i) {
          int64_t cost_insert_i = ComputeOptimalCostWithOccupation(i, x_i, register_offset_backup);
          std::cout << "Insert i at " << x_i << ", cost: " << cost_insert_i << ", Less? "
                    << (cost_insert_i < optimal_cost) << std::endl;
          // Check if we found a smaller cost
          if (cost_insert_i < min_cost) {
            min_cost = cost_insert_i;
            min_x_i = x_i;
            if (min_cost <= cost_without_i) { break; }
          }
        }
        // Found a smaller cost
        if (min_x_i >= 0) {
          InsertRegister(i, min_x_i, register_offset_backup);
          optimal_cost = ComputeOptimalCostFrom0();
          std::cout << "Insert cost: " << min_cost << ", optimal cost: " << optimal_cost
                    << std::endl;
        }
      }
      // Found a smaller cost
      if (min_x_i >= 0) {
        // Move to a new status with smaller cost, dump the backup of the offset.
        ClearBackup();
        step_no_decrease = 0;
      } else {
        // Recover to the original status
        RecoverFromBackup(i);
        // Terminate it if no cost reduce for any of the adjustment.
        step_no_decrease++;
        if (step_no_decrease >= total_register_num_) { break; }
      }
      int64_t recovery_cost = ComputeOptimalCostFrom0();
      std::cout << "After recovery: " << recovery_cost << " Less? "
                << (recovery_cost < optimal_cost) << std::endl;
      CHECK_JUST(CheckConflict());
    }
    if (step_no_decrease >= total_register_num_) { break; }
  }
  return 0;
}

// Let x_i occupy some space [x_i, x_i + l_i), then we recompute the optimal cost
size_t MemoryShareStrategy::ComputeOptimalCostWithOccupation(
    int32_t i, int64_t x_i, const std::vector<int64_t>& register_offset_backup) {
  // The end of register i.
  int64_t e_i = x_i + register_size_[i];
  register_offset_.clear();
  register_offset_.resize(total_register_num_, -1);
  for (int32_t k : excluded_registers[i]) {
    // x_k + l_k > x_i
    // k is behind i
    if (register_offset_backup[k] + register_size_[k] > x_i) {
      register_offset_[k] = -e_i - 1;
    } else {
      register_offset_[k] = register_offset_backup[k];
    }
  }
  return std::max(size_t(e_i), ComputeOptimalCost4CompactRelationship());
}

// Eliminate one register
void MemoryShareStrategy::EliminateRegister(int32_t i) {
  int32_t total_register_num = index2register_.size();
  // TODO: Init back up registers
  backup_registers[i] = left_registers[i];
  for (auto j : excluded_registers[i]) {
    if (register_offset_[i] < register_offset_[j]) {
      should_visit.clear();
      should_visit.resize(total_register_num, 0);
      // should_visit[i] = 0: should not visit i, or have already visited i..
      // should_visit[i] = 1: should visit i, i is excluded with j
      // should_visit[i] = 2: should visit i, i is not excluded with j
      // should_visit[i] = -1: i is visited, i is excluded with j
      // should_visit[i] = -2: i is visited, i is not excluded with j
      for (int32_t k = 0; k < total_register_num; k++) {
        if (register_offset_[k] < register_offset_[j]) {
          if (Exclude(k, j)) {
            should_visit[k] = 1;
          } else {
            should_visit[k] = 2;
          }
        }
      }
      // Eliminate all the grandsons of the excluded registers
      for (int32_t k : excluded_registers[j]) {
        if (should_visit[k] == 1) { EliminateRedundantRelationshipIgnore(i, k); }
      }
      for (int32_t k : excluded_registers[j]) {
        if (should_visit[k] == -1) {
          if (left_registers[j].insert(k).second) { backup_registers[j].insert(k); }
        }
      }
      if (left_registers[j].erase(i)) { backup_register_behind_i.insert(j); }
    }
  }
  left_registers[i].clear();
}

// Find all the left registers of i and link those compact excluded ones for j.
// Not including i itself.
void MemoryShareStrategy::LookForwardLink(int32_t i, int32_t j) {
  for (int32_t k : left_registers[i]) {
    if (Exclude(k, j)) {
      left_registers[j].insert(k);
      backup_registers[j].insert(k);
    } else if (should_visit[k]) {
      LookForwardLink(k, j);
      should_visit[k] = 0;
    }
  }
}

// Whether x_i < x_j
bool MemoryShareStrategy::CompactLessThan(int32_t i, int32_t j) {
  return left_registers[j].find(i) != left_registers[j].end();
}

// Whether i and j occurs simultaneously
bool MemoryShareStrategy::Exclude(int32_t i, int32_t j) {
  return excluded_registers[i].find(j) != excluded_registers[i].end();
}

// If the previous strategy has fewer cost, recover to the previous one from the backup.
void MemoryShareStrategy::RecoverFromBackup(int32_t i) {
  for (int32_t j = 0; j < total_register_num_; j++) {
    if (i == j) {
      left_registers[i] = backup_registers[i];
    } else {
      for (int32_t k : backup_registers[j]) { left_registers[j].erase(k); }
    }
  }
  for (int32_t j : backup_register_behind_i) { left_registers[j].insert(i); }
  ClearBackup();
}

// Clear backup
void MemoryShareStrategy::ClearBackup() {
  for (auto& backup_register : backup_registers) { backup_register.clear(); }
  backup_register_behind_i.clear();
}

size_t MemoryShareStrategy::ComputeOptimalCostFrom0() {
  register_offset_.clear();
  register_offset_.resize(index2register_.size(), -1);
  return ComputeOptimalCost4CompactRelationship();
}

// Insert register i at position [x_i, x_i + l_i)
void MemoryShareStrategy::InsertRegister(int32_t i, int64_t x_i,
                                         const std::vector<int64_t>& original_register_offset) {
  // should_visit[j] = -1, j is visited and x_i < x_j is implied by left_registers
  // should_visit[j] = -2, j is visited and x_j < x_i is implied by left_registers
  // should_visit[j] = -3, j is not excluded with i, irrelevant to i.
  // should_visit[j] = 0, j is not excluded with i, but we have not visit j yet.
  // should_visit[j] = 1, j is excluded with i, x_i < x_j, but we have not visit j yet.
  // should_visit[j] = 2, j is excluded with i, x_j < x_i, but we have not visit j yet.
  should_visit.clear();
  should_visit.resize(total_register_num_, 0);
  for (int32_t j : excluded_registers[i]) {
    if (original_register_offset[j] + register_size_[j] > x_i) {
      // e_j = x_j + l_j > x_i, move j behind i
      should_visit[j] = 1;
    } else {
      // e_j < x_i, j is in front of i, we do not need to move j
      should_visit[j] = 2;
    }
  }
  // Kill all the grandsons
  for (int32_t j : excluded_registers[i]) {
    if (should_visit[j] == 2) { VisitForwardEliminateCompactRelationship(i, j); }
  }
  // All the grandsons are visited. Those left with should_visit[j] == 2 are children of i.
  for (int32_t j : excluded_registers[i]) {
    if (should_visit[j] == 2) {
      left_registers[i].insert(j);
      should_visit[j] = -2;
    }
  }
  // Visit the fathers and grandpas of i
  for (int32_t j : excluded_registers[i]) { VisitBackwardInsertCompactRelationship(i, j); }
}

// Visit j and add the compact relationship while inserting i
// Find all the k s.t. x_k < x_j < x_i, j is excluded with i.
// Thus the first input must satisfy should_visit[j] = 2.
// After that, we may have should_visit[k] = 0.
void MemoryShareStrategy::VisitForwardEliminateCompactRelationship(int32_t i, int32_t j) {
  // If we have not visited j.
  if (should_visit[j] >= 0) {
    // Visit all the children of j, they will not have compact relationship with i
    for (int32_t k : left_registers[j]) {
      VisitForwardEliminateCompactRelationship(i, k);
      should_visit[k] = -2;
    }
  }
}

// All the -2 must be found before this function.
// Look through all the undetermined j, define the status of them.
void MemoryShareStrategy::VisitBackwardInsertCompactRelationship(int32_t i, int32_t j) {
  // If we have not visited j.
  // NOTE: should_visit[j] != 2 at the current moment.
  if (should_visit[j] >= 0) {
    // Visit all the children of j
    for (int32_t k : left_registers[j]) {
      // Visit k if k is not visited and not excluded.
      if (should_visit[k] == 0) { VisitBackwardInsertCompactRelationship(i, k); }
      // x_i < x_k, then x_i < x_k < x_j
      if (should_visit[k] == 1 || should_visit[k] == -1) {
        should_visit[j] = -1;
        return;
      }
    }
    // Look through all the children, can not found i in the grandsons.
    if (should_visit[j] == 0) {
      should_visit[j] = -3;
    } else {
      // should_visit[j] == 1
      left_registers[j].insert(i);
      should_visit[j] = -1;
    }
  }
}

// Eliminate children of j but ignore i.
void MemoryShareStrategy::EliminateRedundantRelationshipIgnore(int32_t i, int32_t j) {
  // Ignore i
  if (i == j) { return; }
  if (should_visit[j] > 0) {
    // Do not look into it again
    should_visit[j] = -should_visit[j];
    for (int32_t k : left_registers[j]) {
      EliminateRedundantRelationshipIgnore(i, k);
      should_visit[k] = 0;
    }
  }
}

// Check whether the current offset does not introduce any conflict
Maybe<void> MemoryShareStrategy::CheckConflict() {
  CHECK_EQ_OR_RETURN(index2register_.size(), register_offset_.size())
      << "Not equal size, we might be calling CheckConflict() at a wrong time.";
  for (int32_t i = 0; i < index2register_.size(); i++) {
    CHECK_GE_OR_RETURN(register_offset_[i], 0) << "Register offset is not computed.";
    for (int32_t j : excluded_registers[i]) {
      CHECK_OR_RETURN(register_offset_[i] + register_size_[i] <= register_offset_[j]
                      || register_offset_[j] + register_size_[j] <= register_offset_[i])
          << "Two registers overlap";
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace linear_programming
}  // namespace oneflow