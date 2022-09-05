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
#include "oneflow/core/register/runtime_register_desc.h"

namespace oneflow {
namespace linear_programming {
void MemoryShareStrategy::ConstructMip4MemoryOnly(
    const HashMap<RegstDescProto*, std::vector<RegstDescProto*> >& regst2mutual_exclusion_regsts) {
  // Initialization
  int32_t total_register_num = regst2mutual_exclusion_regsts.size();
  register_offset_.resize(total_register_num);
  register_size_.resize(total_register_num);
  int32_t register_id = 0;
  int64_t large_m = 1;
  for (const auto& pair : regst2mutual_exclusion_regsts) {
    register_id++;
    int64_t register_size = RtRegstDesc(*pair.first).TotalMainByteSize4AllRegst();
    register_size_[register_id] = register_size;
    register2index[pair.first] = register_id;
    large_m += register_size;
  }
  large_m_ = double(large_m);

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

  // Assemble x_i + l_i <= z,
  // where z is the total size of the memory block.
  // We need to make sure that the right hand size >= 0. We transfer the formula to
  // z - x_i - s_i = l_i
  // z: 0
  // x_i: [1, total_register_num], i+1
  // s_i: [total_register_num+1, 2*total_register_num], i+total_register_num+1
  for (int32_t i = 0; i < total_register_num; i++) {
    primal_matrix.Insert(row, 0, 1.0);                            // z
    primal_matrix.Insert(row, i + 1, -1.0);                       // -x_i
    primal_matrix.Insert(row, i + total_register_num + 1, -1.0);  // -s_i
    primal_b[row] = register_size_[i];                            // l_i
    row++;
  }

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
  int32_t index = 0;
  for (const auto& register_i2exclusive_registers : regst2mutual_exclusion_regsts) {
    int32_t i = register2index[register_i2exclusive_registers.first];
    for (const auto& register_j : register_i2exclusive_registers.second) {
      int32_t j = register2index[register_j];
      if (i < j) {
        // -x_i + x_j + M * t1_ij - s1_ij = l_i
        primal_matrix.Insert(row, i + 1, -1.0);                                       // -x_i
        primal_matrix.Insert(row, j + 1, 1.0);                                        // x_j
        primal_matrix.Insert(row, 2 * total_register_num + 4 * index + 1, large_m_);  // M * t1_ij
        primal_matrix.Insert(row, 2 * total_register_num + 4 * index + 3, -1.0);      // -s1_ij
        primal_b[row] = register_size_[i];                                            // l_i
        row++;

        // x_i - x_j + M * t2_ij - s2_ij = l_j
        primal_matrix.Insert(row, i + 1, 1.0);                                        // x_i
        primal_matrix.Insert(row, j + 1, -1.0);                                       // -x_j
        primal_matrix.Insert(row, 2 * total_register_num + 4 * index + 2, large_m_);  // M * t2_ij
        primal_matrix.Insert(row, 2 * total_register_num + 4 * index + 4, -1.0);      // -s2_ij
        primal_b[row] = register_size_[j];                                            // l_j
        row++;

        // t1_ij + t2_ij = 1
        primal_matrix.Insert(row, 2 * total_register_num + 4 * index + 1, 1.0);  // t1_ij
        primal_matrix.Insert(row, 2 * total_register_num + 4 * index + 2, 1.0);  // t2_ij
        primal_b[row] = 1.0;                                                     // 1
        row++;
      }
    }
  }

  CHECK_EQ(row, primal_b.size())
      << "Inconsistent rows, bug occurs while assembling mix-integer programming";

  // Minimize z
  // c = [1, 0, 0, 0, ..., 0]
  mips_.lps_.primal_cost_.resize(2 * total_register_num + 4 * index + 1);
  mips_.lps_.primal_cost_[0] = 1.0;
}

void MemoryShareStrategy::ExportMemoryOffsets(
    size_t* mem_block_size, HashMap<RegstDescProto*, int64_t>* regst_desc2offset) {
  *mem_block_size = mem_block_size_;
  for (const auto& pair : register2index) {
    (*regst_desc2offset)[pair.first] = register_size_[pair.second];
  }
}
}  // namespace linear_programming
}  // namespace oneflow