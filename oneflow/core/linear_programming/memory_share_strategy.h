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
  HashMap<RegstDescProto*, int32_t> register2index;
  // A large enough number M
  double large_m_;

  void ConstructMip4MemoryOnly(
      const HashMap<RegstDescProto*, int64_t>& regst_desc2size,
      const HashMap<RegstDescProto*, std::vector<RegstDescProto*>>& regst2mutual_exclusion_regsts);

  void ExportMemoryOffsets(size_t* mem_block_size,
                           HashMap<RegstDescProto*, int64_t>* regst_desc2offset);
};
}  // namespace linear_programming
}  // namespace oneflow

#endif  // ONEFLOW_CORE_LINEAR_PROGRAMMING_MEMORY_SHARE_STRATEGY_H_