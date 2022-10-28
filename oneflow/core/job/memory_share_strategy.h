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

#ifndef ONEFLOW_CORE_JOB_MEMORY_SHARE_STRATEGY_H_
#define ONEFLOW_CORE_JOB_MEMORY_SHARE_STRATEGY_H_

#include <vector>
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/register/register_desc.pb.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

// NOTE: Another trick to save times.
// Comparing two numbers is faster than asking the existence in a HashSet.
bool IsLifetimeExcluded(const std::pair<int32_t, int32_t>& a, const std::pair<int32_t, int32_t>& b);

class MemoryShareStrategy {
 public:
  // Adaptively update the offset of registers to minimize the total memory
  void AdaptivelyUpdateOffset(
      const HashMap<RegstDescProto*, size_t>& mem_reused_regst2size,
      const HashMap<RegstDescProto*, std::pair<int32_t, int32_t>>& register2lifetime,
      size_t lower_bound, size_t* mem_block_size,
      HashMap<RegstDescProto*, int64_t>* regst_desc2offset);

  // Set the offset of registers to minimize the total memory
  // Iterating from a random order might take a lot of steps to reach the optimal cost.
  // Therefore, this function is not recommended with an initial offset provided.
  void GenerateOffset(
      const HashMap<RegstDescProto*, size_t>& mem_reused_regst2size,
      const HashMap<RegstDescProto*, std::pair<int32_t, int32_t>>& register2lifetime,
      size_t* mem_block_size, HashMap<RegstDescProto*, int64_t>* regst_desc2offset);

 private:
  size_t mem_block_size_;
  int32_t max_iteration_step_;
  std::vector<int64_t> register_offset_;
  std::vector<int64_t> register_size_;
  HashMap<RegstDescProto*, int32_t> register2index_;
  std::vector<RegstDescProto*> index2register_;
  // left registers store the first registers on the left, which have smaller offsets.
  // For example, 1 < 2 < 3 < 5
  //                  2 < 4 < 5
  // Then
  //      left_registers_[1] = {}
  //      left_registers_[2] = {1}
  //      left_registers_[3] = {2}
  //      left_registers_[4] = {2}
  //      left_registers_[5] = {3, 4}
  //  We know that 1 < 3, but 1 is not in left_registers_[3],
  //  since we only store the first registers.
  std::vector<HashSet<int32_t>> left_registers_;
  // Store all the registers which exist simultaneously.
  std::vector<HashSet<int32_t>> excluded_registers_;
  // Back up the changes
  std::vector<HashSet<int32_t>> backup_registers_;
  HashSet<int32_t> backup_register_behind_i_;
  // A buffer which implies whether we should visit a register
  std::vector<int32_t> should_visit_;
  int32_t total_register_num_;
  std::vector<int32_t> order_;

  // Mid-level interfaces
  // Steal a compact position as the initial strategy
  void StealCompactPosition(
      const HashMap<RegstDescProto*, int64_t>& regst_desc2offset,
      const HashMap<RegstDescProto*, size_t>& mem_reused_regst2size,
      const HashMap<RegstDescProto*, std::pair<int32_t, int32_t>>& register2lifetime);
  // Generate a compact position with the order of occurrence
  void GenerateCompactPosition(
      const HashMap<RegstDescProto*, size_t>& mem_reused_regst2size,
      const HashMap<RegstDescProto*, std::pair<int32_t, int32_t>>& register2lifetime);
  // Update the offset with the adjusted strategy
  void UpdateOffset(size_t* mem_block_size, HashMap<RegstDescProto*, int64_t>* regst_desc2offset);
  // Update the maximum iteration step with the current size and lower bound
  void UpdateMaxIteration(size_t mem_block_size, size_t lower_bound);

  // Initialization
  void InitRegister(const HashMap<RegstDescProto*, std::pair<int32_t, int32_t>>& register2lifetime);
  void InitRegisterInformation(const HashMap<RegstDescProto*, size_t>& mem_reused_regst2size);
  // Adjust the original strategy, return the updated optimal cost
  size_t ComputeOptimalAdjustedCost();
  // Eliminate one register
  void EliminateRegister(int32_t i);
  // Eliminate children of j but ignore i.
  void EliminateRedundantRelationshipIgnore(int32_t i, int32_t j);
  // Whether i and j occurs simultaneously
  bool Exclude(int32_t i, int32_t j);
  // If the previous strategy without the elimination of i has fewer cost, recover to the previous
  // one from the backup.
  void RecoverFromBackup(int32_t i);
  // Clear backup
  void ClearBackup();
  // Let x_i occupy some space [x_i, x_i + l_i), then we recompute the optimal cost
  size_t ComputeOptimalCostWithOccupation(int32_t i, int64_t x_i,
                                          const std::vector<int64_t>& register_offset_backup);
  // Insert register i at position [x_i, x_i + l_i)
  void InsertRegister(int32_t i, int64_t x_i, const std::vector<int64_t>& original_register_offset);

  // Compute optimal cost with compact relationship
  size_t ComputeOptimalCost4CompactRelationship();
  size_t ComputeOptimalCostFrom0();
  // Compute offset with compact relationship
  int64_t ComputeOffset4CompactRelationship(int32_t i);
  // Check whether the current offset does not introduce any conflict
  Maybe<void> CheckConflict();
  // Reset the compact position for the registers with should_visit_ = 0
  void ResetCompactPosition(int32_t j);
  // Find all the k < i, eliminates k < j,
  // since k < i and i < j have already implied that.
  void EliminateRedundantRelationship(int32_t i);
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_MEMORY_SHARE_STRATEGY_H_
