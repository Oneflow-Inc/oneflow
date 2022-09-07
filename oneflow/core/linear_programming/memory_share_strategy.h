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

#ifndef ONEFLOW_CORE_LINEAR_PROGRAMMING_MEMORY_SHARE_STRATEGY_H_
#define ONEFLOW_CORE_LINEAR_PROGRAMMING_MEMORY_SHARE_STRATEGY_H_

#include <vector>
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/linear_programming/mix_integer_programming_util.h"
#include "oneflow/core/register/register_desc.pb.h"
namespace oneflow {
namespace linear_programming {
class MemoryShareStrategy {
 public:
  MixIntegerProgrammingSolver mips_;
  size_t mem_block_size_;
  std::vector<int64_t> register_offset_;
  std::vector<int64_t> register_size_;
  HashMap<RegstDescProto*, int32_t> register2index_;
  std::vector<RegstDescProto*> index2register_;
  // A large enough number M
  double large_m_;
  int32_t start_row_exclusion, end_row_exclusion;
  int32_t start_column_exclusion, end_column_exclusion;
  const int32_t num_row_group = 3;

  void ConstructMip4MemoryOnly(
      const HashMap<RegstDescProto*, std::vector<RegstDescProto*>>& regst2mutual_exclusion_regsts);

  void ExportMemoryOffsets(size_t* mem_block_size,
                           HashMap<RegstDescProto*, int64_t>* regst_desc2offset);

  void StealPosition(const HashMap<RegstDescProto*, int64_t>& regst_desc2offset);

  void StealCompactPosition(
      const HashMap<RegstDescProto*, int64_t>& regst_desc2offset,
      const HashMap<RegstDescProto*, std::vector<RegstDescProto*>>& regst2mutual_exclusion_regsts);

 private:
  // left registers store the first registers on the left, which have smaller offsets.
  // For example, 1 < 2 < 3 < 5
  //                  2 < 4 < 5
  // Then
  //      left_registers[1] = {}
  //      left_registers[2] = {1}
  //      left_registers[3] = {2}
  //      left_registers[4] = {2}
  //      left_registers[5] = {3, 4}
  //  We know that 1 < 3, but 1 is not in left_registers[3],
  //  since we only store the first registers.
  std::vector<std::vector<int32_t>> left_registers;
  // Initialization
  void InitRegister(
      const HashMap<RegstDescProto*, std::vector<RegstDescProto*>>& regst2mutual_exclusion_regsts);
  void InitRegisterInformation();

  // Assemble x_i + l_i <= z
  // z = max(x_i) for all the i
  void AssembleZ(int32_t* row);
  // Assemble cost for minimizing z
  void MinimizeZ();
  // Compute optimal cost with compact relationship
  size_t ComputeOptimalCost4CompactRelationship();
  // Compute offset with compact relationship
  int64_t ComputeOffset4CompactRelationship(int32_t i);
};
}  // namespace linear_programming
}  // namespace oneflow

#endif  // ONEFLOW_CORE_LINEAR_PROGRAMMING_MEMORY_SHARE_STRATEGY_H_