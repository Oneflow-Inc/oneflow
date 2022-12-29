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

#include "oneflow/core/job/memory_share_strategy.h"
#include <glog/logging.h>
#include <algorithm>
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/register/runtime_register_desc.h"

namespace oneflow {

namespace {
constexpr int32_t kMaxIterStep = 100;
}  // anonymous namespace

bool IsLifetimeExcluded(const std::pair<int32_t, int32_t>& a,
                        const std::pair<int32_t, int32_t>& b) {
  return a.first < b.second && b.first < a.second;
}

// Initialization
void MemoryShareStrategy::InitRegister(
    const HashMap<RegstDescProto*, std::pair<int32_t, int32_t>>& register2lifetime) {
  total_register_num_ = register2lifetime.size();
  index2register_.resize(total_register_num_);
  int32_t register_id = 0;
  for (const auto& pair : register2lifetime) {
    index2register_[register_id] = pair.first;
    register_id++;
  }
}

void MemoryShareStrategy::InitRegisterInformation(
    const HashMap<RegstDescProto*, size_t>& mem_reused_regst2size) {
  total_register_num_ = index2register_.size();
  register_size_.resize(total_register_num_);
  for (int32_t register_id = 0; register_id < total_register_num_; register_id++) {
    const auto& register_ = index2register_[register_id];
    int64_t register_size = mem_reused_regst2size.at(register_);
    register_size_[register_id] = register_size;
    register2index_[register_] = register_id;
  }
  order_.resize(total_register_num_);
  for (int32_t i = 0; i < total_register_num_; i++) { order_[i] = i; }
}

// Steal a compact position as the initial strategy
void MemoryShareStrategy::StealCompactPosition(
    const HashMap<RegstDescProto*, int64_t>& regst_desc2offset,
    const HashMap<RegstDescProto*, size_t>& mem_reused_regst2size,
    const HashMap<RegstDescProto*, std::pair<int32_t, int32_t>>& register2lifetime) {
  // Initialization
  InitRegister(register2lifetime);

  // Sort index2register_
  std::sort(index2register_.begin(), index2register_.end(),
            [&](RegstDescProto* i, RegstDescProto* j) {
              return regst_desc2offset.at(i) < regst_desc2offset.at(j);
            });
  // Update other information
  InitRegisterInformation(mem_reused_regst2size);

  left_registers_.clear();
  left_registers_.resize(total_register_num_);
  excluded_registers_.clear();
  excluded_registers_.resize(total_register_num_);
  // should_visit_[i] indicates whether we should visit register[i].
  // should_visit_[i] = 0: should not visit i, or have already visited i..
  // should_visit_[i] = 1: should visit i, i is excluded with j
  // should_visit_[i] = 2: should visit i, i is not excluded with j
  should_visit_.clear();
  should_visit_.resize(total_register_num_, 0);
  register_offset_.resize(total_register_num_);
  // Generate a compact relationship of position
  // For example we have 3 relationship: x1 < x2, x2 < x3, x1 < x3
  // We would delete the redundant relationship (x1 < x3)
  for (int32_t j = 0; j < total_register_num_; j++) {
    const auto& register_j = index2register_[j];
    register_offset_[j] = regst_desc2offset.at(register_j);
    auto& excluded_register_j = excluded_registers_[j];
    const auto& lifetime_j = register2lifetime.at(register_j);
    // Init should visit with all orders of the excluded register
    for (int32_t i = j + 1; i < total_register_num_; i++) {
      if (IsLifetimeExcluded(lifetime_j, register2lifetime.at(index2register_[i]))) {
        // Copy the data to excluded registers
        excluded_register_j.insert(i);
        excluded_registers_[i].insert(j);
      }
    }
  }

  for (int32_t j = 0; j < total_register_num_; j++) { ResetCompactPosition(j); }
}

// Generate a compact position with the order of occurrence
// Not recommended
void MemoryShareStrategy::GenerateCompactPosition(
    const HashMap<RegstDescProto*, size_t>& mem_reused_regst2size,
    const HashMap<RegstDescProto*, std::pair<int32_t, int32_t>>& register2lifetime) {
  HashMap<RegstDescProto*, int64_t> regst_desc2offset;
  int64_t offset = 0;
  for (const auto& pair : register2lifetime) {
    regst_desc2offset[pair.first] = offset;
    offset++;
  }
  StealCompactPosition(regst_desc2offset, mem_reused_regst2size, register2lifetime);
}

// Compute optimal cost with compact relationship
size_t MemoryShareStrategy::ComputeOptimalCost4CompactRelationship() {
  int64_t mem_block_size = 0;
  for (int32_t i = 0; i < total_register_num_; i++) {
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
    for (int32_t j : left_registers_[i]) {
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
  // std::vector<int32_t> order_;
  // auto CompareRegisterPosition = [&](int32_t i, int32_t j) {
  //   return register_offset_[i] < register_offset_[j];
  // };
  backup_registers_.clear();
  backup_registers_.resize(total_register_num_);
  // The number of steps that the optimal cost does not decrease
  int32_t step_no_decrease = 0;
  for (int32_t m = 0; m < max_iteration_step_; m++) {
    for (int32_t i = 0; i < total_register_num_; i++) {
      EliminateRegister(i);
      size_t cost_without_i = ComputeOptimalCostFrom0();
      // Find the offset of i which has the minimum cost
      int64_t min_x_i = -1;
      if (cost_without_i < optimal_cost) {
        // Find the minimum cost
        int64_t min_cost = optimal_cost;
        // Back up the current register offset with elimination of i
        auto register_offset_backup = register_offset_;
        // Try to insert the register i into the sorted excluded registers
        HashSet<int64_t> all_x_i;
        for (int32_t j : excluded_registers_[i]) {
          // Insert i before j
          all_x_i.insert(register_offset_backup[j]);
          // Insert i after j
          all_x_i.insert(register_offset_backup[j] + register_size_[j]);
        }

        for (int64_t x_i : all_x_i) {
          int64_t cost_insert_i = ComputeOptimalCostWithOccupation(i, x_i, register_offset_backup);
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
        // Adjust the offset after recovery
        ComputeOptimalCostFrom0();
        // Terminate it if no cost reduce for any of the adjustment.
        step_no_decrease++;
        if (step_no_decrease >= total_register_num_) { break; }
      }
    }
    if (step_no_decrease >= total_register_num_) { break; }
  }
  CHECK_JUST(CheckConflict());
  return optimal_cost;
}

// Let x_i occupy some space [x_i, x_i + l_i), then we recompute the optimal cost
size_t MemoryShareStrategy::ComputeOptimalCostWithOccupation(
    int32_t i, int64_t x_i, const std::vector<int64_t>& register_offset_backup) {
  // The end of register i.
  int64_t e_i = x_i + register_size_[i];
  register_offset_.clear();
  register_offset_.resize(total_register_num_, -1);
  for (int32_t k : excluded_registers_[i]) {
    // x_k + l_k > x_i
    // k is behind i
    if (register_offset_backup[k] + register_size_[k] > x_i) {
      register_offset_[k] = -e_i - 1;
    } else {
      register_offset_[k] = register_offset_backup[k];
    }
  }
  register_offset_[i] = x_i;
  return ComputeOptimalCost4CompactRelationship();
}

// Eliminate one register
void MemoryShareStrategy::EliminateRegister(int32_t i) {
  // Init back up registers
  backup_registers_[i] = left_registers_[i];
  for (auto j : excluded_registers_[i]) {
    if (register_offset_[i] < register_offset_[j]) {
      should_visit_.clear();
      should_visit_.resize(total_register_num_, 0);
      // should_visit_[i] = 0: should not visit i, or have already visited i..
      // should_visit_[i] = 1: should visit i, i is excluded with j
      // should_visit_[i] = 2: should visit i, i is not excluded with j
      // should_visit_[i] = -1: i is visited, i is excluded with j
      // should_visit_[i] = -2: i is visited, i is not excluded with j
      for (int32_t k = 0; k < total_register_num_; k++) {
        if (register_offset_[k] < register_offset_[j]) {
          if (Exclude(k, j)) {
            should_visit_[k] = 1;
          } else {
            should_visit_[k] = 2;
          }
        }
      }
      // Eliminate all the grandsons of the excluded registers
      for (int32_t k : excluded_registers_[j]) {
        if (should_visit_[k] == 1) { EliminateRedundantRelationshipIgnore(i, k); }
      }
      for (int32_t k : excluded_registers_[j]) {
        if (should_visit_[k] == -1) {
          if (left_registers_[j].insert(k).second) { backup_registers_[j].insert(k); }
        }
      }
      if (left_registers_[j].erase(i)) { backup_register_behind_i_.insert(j); }
    }
  }
  left_registers_[i].clear();
}

// Whether i and j occurs simultaneously
bool MemoryShareStrategy::Exclude(int32_t i, int32_t j) {
  return excluded_registers_[i].find(j) != excluded_registers_[i].end();
}

// If the previous strategy has fewer cost, recover to the previous one from the backup.
void MemoryShareStrategy::RecoverFromBackup(int32_t i) {
  for (int32_t j = 0; j < total_register_num_; j++) {
    if (i == j) {
      left_registers_[i] = backup_registers_[i];
    } else {
      for (int32_t k : backup_registers_[j]) { left_registers_[j].erase(k); }
    }
  }
  for (int32_t j : backup_register_behind_i_) { left_registers_[j].insert(i); }
  ClearBackup();
}

// Clear backup
void MemoryShareStrategy::ClearBackup() {
  for (auto& backup_register : backup_registers_) { backup_register.clear(); }
  backup_register_behind_i_.clear();
}

size_t MemoryShareStrategy::ComputeOptimalCostFrom0() {
  register_offset_.clear();
  register_offset_.resize(total_register_num_, -1);
  return ComputeOptimalCost4CompactRelationship();
}

// Insert register i at position [x_i, x_i + l_i)
void MemoryShareStrategy::InsertRegister(int32_t i, int64_t x_i,
                                         const std::vector<int64_t>& original_register_offset) {
  ComputeOptimalCostWithOccupation(i, x_i, original_register_offset);
  std::sort(order_.begin(), order_.end(),
            [&](int32_t k, int32_t j) { return register_offset_[k] < register_offset_[j]; });
  for (int32_t j : order_) {
    if (register_offset_[i] <= register_offset_[j]) { ResetCompactPosition(j); }
  }
}

// Eliminate children of j but ignore i.
void MemoryShareStrategy::EliminateRedundantRelationshipIgnore(int32_t i, int32_t j) {
  // Ignore i
  if (i == j) { return; }
  if (should_visit_[j] > 0) {
    // Do not look into it again
    should_visit_[j] = -should_visit_[j];
    for (int32_t k : left_registers_[j]) {
      EliminateRedundantRelationshipIgnore(i, k);
      should_visit_[k] = 0;
    }
  }
}

// Check whether the current offset does not introduce any conflict
Maybe<void> MemoryShareStrategy::CheckConflict() {
  CHECK_EQ_OR_RETURN(index2register_.size(), register_offset_.size())
      << "Not equal size, we might be calling CheckConflict() at a wrong time.";
  for (int32_t i = 0; i < total_register_num_; i++) {
    CHECK_GE_OR_RETURN(register_offset_[i], 0) << "Register offset is not computed.";
    for (int32_t j : excluded_registers_[i]) {
      CHECK_OR_RETURN(register_offset_[i] + register_size_[i] <= register_offset_[j]
                      || register_offset_[j] + register_size_[j] <= register_offset_[i])
          << "Two registers overlap";
    }
  }
  return Maybe<void>::Ok();
}

// Update the offset with the adjusted strategy
void MemoryShareStrategy::UpdateOffset(size_t* mem_block_size,
                                       HashMap<RegstDescProto*, int64_t>* regst_desc2offset) {
  size_t optimal_cost = ComputeOptimalAdjustedCost();
  if (optimal_cost < *mem_block_size) {
    VLOG(3) << "Original cost: " << *mem_block_size << ", updated cost: " << optimal_cost;
    *mem_block_size = optimal_cost;
    for (auto& pair : *regst_desc2offset) {
      pair.second = register_offset_[register2index_[pair.first]];
    }
  }
}

// Find all the k < i, eliminates k < j,
// since k < i and i < j have already implied that.
void MemoryShareStrategy::EliminateRedundantRelationship(int32_t i) {
  // If i is already eliminate, skip it.
  if (should_visit_[i]) {
    for (int32_t k : left_registers_[i]) {
      // Eliminate all the k < i
      EliminateRedundantRelationship(k);
      // Eliminate left[i]
      should_visit_[k] = 0;
    }
  }
}

// Reset the compact position for the registers
void MemoryShareStrategy::ResetCompactPosition(int32_t j) {
  left_registers_[j].clear();
  // Mark all the registers on the left
  for (int32_t i = 0; i < total_register_num_; i++) {
    if (register_offset_[i] < register_offset_[j]) {
      if (Exclude(i, j)) {
        should_visit_[i] = 1;
      } else {
        should_visit_[i] = 2;
      }
    } else {
      // Might be unnecessary since we clear up should_visit_ before.
      should_visit_[i] = 0;
    }
  }

  for (int32_t i = 0; i < total_register_num_; i++) {
    if (should_visit_[i] == 1) {
      // Find all the k < i, eliminates k < j,
      // since k < i and i < j have already implied that.
      // Also reset should_visit_[i] to false,
      // since we have already visited i.
      EliminateRedundantRelationship(i);
    }
  }

  for (int32_t i = 0; i < total_register_num_; i++) {
    if (should_visit_[i] == 1) {
      // i < j
      left_registers_[j].insert(i);
    }
    // Might be unnecessary since we clear up should_visit_ before.
    should_visit_[i] = 0;
  }
}

// Update the maximum iteration step with the current size and lower bound
void MemoryShareStrategy::UpdateMaxIteration(size_t mem_block_size, size_t lower_bound) {
  if (lower_bound > 0) {
    max_iteration_step_ = ((mem_block_size - lower_bound) * 100) / lower_bound;
  } else {
    // A graph only containing several 0 size tensors might have lower bound = 0.
    // Check test_div.py::TestDiv::test_0_size_div for example.
    max_iteration_step_ = 0;
  }
  // if mem_block_size is closed to the maximum number of type size_t, then we might have a negative
  // value for (mem_block_size - lower_bound) * 100
  // In this case, we just set a large max_iteration_step_
  if (max_iteration_step_ < 0) { max_iteration_step_ = kMaxIterStep; }
}

// Adaptively update the offset of registers to minimize the total memory
void MemoryShareStrategy::AdaptivelyUpdateOffset(
    const HashMap<RegstDescProto*, size_t>& mem_reused_regst2size,
    const HashMap<RegstDescProto*, std::pair<int32_t, int32_t>>& register2lifetime,
    size_t lower_bound, size_t* mem_block_size,
    HashMap<RegstDescProto*, int64_t>* regst_desc2offset) {
  VLOG(3) << "Current memory size: " << *mem_block_size << ", lower bound : " << lower_bound;
  if (*mem_block_size > lower_bound) {
    UpdateMaxIteration(*mem_block_size, lower_bound);
    VLOG(3) << "max iteration step: " << max_iteration_step_;
    if (max_iteration_step_ > 0) {
      StealCompactPosition(*regst_desc2offset, mem_reused_regst2size, register2lifetime);
      UpdateOffset(mem_block_size, regst_desc2offset);
    }
    VLOG(3) << "After compression, memory size: " << *mem_block_size;
  }
}

// Set the offset of registers to minimize the total memory
// Iterating from a random order might take a lot of steps to reach the optimal cost.
// Therefore, this function is not recommended with an initial offset provided.
void MemoryShareStrategy::GenerateOffset(
    const HashMap<RegstDescProto*, size_t>& mem_reused_regst2size,
    const HashMap<RegstDescProto*, std::pair<int32_t, int32_t>>& register2lifetime,
    size_t* mem_block_size, HashMap<RegstDescProto*, int64_t>* regst_desc2offset) {
  max_iteration_step_ = kMaxIterStep;
  VLOG(3) << "max iteration step: " << max_iteration_step_;
  GenerateCompactPosition(mem_reused_regst2size, register2lifetime);
  UpdateOffset(mem_block_size, regst_desc2offset);
}

}  // namespace oneflow
