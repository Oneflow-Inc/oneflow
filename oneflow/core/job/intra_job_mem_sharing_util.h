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
#ifndef ONEFLOW_CORE_JOB_IN_JOB_MEM_SHARING_UTIL_H_
#define ONEFLOW_CORE_JOB_IN_JOB_MEM_SHARING_UTIL_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/job/memory_share_strategy.h"
#include "oneflow/core/job/plan.pb.h"
#include <functional>
#include <string>

namespace oneflow {

struct IntraJobMemSharingUtil {
  static void InferMemBlockId4MemReusedRegst(
      Plan* plan, const std::function<bool(const std::string&, const std::string&)>&
                      IsOpNameDataOrCtrlReachable);
};

template<class T>
struct MemBlockResultInfo {
  size_t mem_block_size;
  HashMap<T, int64_t> regst_desc2offset;
};

// Judge whether a is suitable than b for a gap
inline bool SuitableThan(int64_t a, int64_t b) {
  // The number have orders
  // A non-negative number is always more suitable than a negative number
  // If a number is non-negative, then the smaller the better
  // If a number is negative, then the larger the better
  // 0 > 1 > 2 > ... > 999999999 > -1 > -2 > ... > -99999999
  // Now we flip the positive part to make it "the larger the better".
  if (a >= 0) { a = GetMaxVal<int64_t>() - a; }
  if (b >= 0) { b = GetMaxVal<int64_t>() - b; }
  return a > b;
}

template<class T>
void MemReusedAlgorithmAllocateByOrder(
    const bool compact_insert, const std::vector<T>& order,
    const HashMap<T, size_t>& regst_desc2size,
    const HashMap<T, std::pair<int32_t, int32_t>>& regst2lifetime, MemBlockResultInfo<T>* result) {
  HashMap<T, int64_t>* regst_desc2offset = &(result->regst_desc2offset);
  // NOTE: It is important to make the variables local.
  // It took me several days to find out that using passed-in vector for size, order, and lifetime
  // would double the running time. Switch HashMap to vector
  int32_t total_register_num = order.size();
  std::vector<int64_t> order2size(total_register_num);
  std::vector<std::pair<int32_t, int32_t>> order2lifetime(total_register_num);
  std::vector<int64_t> order2offset(total_register_num);
  for (int32_t i = 0; i < total_register_num; i++) {
    order2size[i] = regst_desc2size.at(order[i]);
    order2lifetime[i] = regst2lifetime.at(order[i]);
  }
  size_t buffer_size = 1;
  // Sort by offset
  auto comp = [&order2offset](const auto& a, const auto& b) {
    if (order2offset[a] != order2offset[b]) { return order2offset[a] < order2offset[b]; }
    // Make sure we have a stable order even if we have the same offset for different registers
    return a < b;
  };
  std::set<int32_t, decltype(comp)> sorted_registers(comp);
  // Decide offset following the given order
  for (int32_t inserting_id = 0; inserting_id < total_register_num; inserting_id++) {
    const auto& inserting_lifetime = order2lifetime[inserting_id];
    // At the beginning, try to insert the offset in the front of the whole memory pool.
    int64_t inserting_offset = 0;
    int64_t inserting_end = inserting_offset + order2size[inserting_id];
    if (compact_insert) {
      // Find the most suitable gap for the register
      int64_t gap_head = 0;
      int64_t inserting_size = order2size[inserting_id];
      // difference = length of gap - length of the inserting register
      int64_t diff_gap = 0, suitable_diff_gap = -1 - inserting_size;
      for (const auto& curr_register : sorted_registers) {
        // Ignore those non-excluded registers
        if (IsLifetimeExcluded(inserting_lifetime, order2lifetime[curr_register])) {
          if (gap_head < order2offset[curr_register]) {
            // Find one gap
            diff_gap = (order2offset[curr_register] - gap_head) - inserting_size;
            // Compared with the previous suitable gap
            if (SuitableThan(diff_gap, suitable_diff_gap)) {
              suitable_diff_gap = diff_gap;
              // We may insert the register into the gap
              inserting_offset = gap_head;
            }
            // Update gap head
            gap_head = order2offset[curr_register] + order2size[curr_register];
          } else {
            // No gap, update gap head
            gap_head = std::max(gap_head, order2offset[curr_register] + order2size[curr_register]);
          }
        }
      }
      // Deal with the buffer_size, which may be the final gap
      diff_gap = (buffer_size - gap_head) - inserting_size;
      // Compared with the previous suitable gap
      if (SuitableThan(diff_gap, suitable_diff_gap)) {
        suitable_diff_gap = diff_gap;
        // We may insert the register into the gap
        inserting_offset = gap_head;
      }
      // If no gap large enough to contain the current register
      if (suitable_diff_gap < 0) {
        // Prolong the maximum memory pool size by (-suitable_diff_gap)
        buffer_size -= suitable_diff_gap;
        int64_t gap_end = suitable_diff_gap + inserting_size + inserting_offset;
        for (auto reverse_it = sorted_registers.rbegin(); reverse_it != sorted_registers.rend();
             reverse_it++) {
          // All the registers with offset < gap_end maintain their position
          if (order2offset[*reverse_it] < gap_end) { break; }
          // All the registers with offset >= gap_end move backward
          order2offset[*reverse_it] -= suitable_diff_gap;
        }
      }

    } else {
      for (const auto& curr_register : sorted_registers) {
        // i: inserting register, j: current register
        // x: register offset, l: register size
        // If x_i + l_i <= x_j, then the inserting register would be placed at x_i
        if (order2offset[curr_register] >= inserting_end) { break; }
        // If i and j are excluded, and x_i + l_i > x_j,
        // then we try to place i at x_j + l_j and check the following registers
        if (IsLifetimeExcluded(inserting_lifetime, order2lifetime[curr_register])) {
          int64_t curr_end = order2offset[curr_register] + order2size[curr_register];
          // Can not set inserting offset = current end directly.
          // We might have two excluded registers like this:
          // register a: [100, 10000]
          // register b: [500, 600]
          if (inserting_offset < curr_end) {
            inserting_offset = curr_end;
            inserting_end = inserting_offset + order2size[inserting_id];
          }
        }
      }
      // Update total size
      if (inserting_end > buffer_size) { buffer_size = inserting_end; }
    }
    // Either we break the loop or the loop terminated naturally, we can place i at inserting_offset
    order2offset[inserting_id] = inserting_offset;
    sorted_registers.insert(inserting_id);
  }

  result->mem_block_size = buffer_size;
  // Switch vector to HashMap
  for (int32_t i = 0; i < total_register_num; i++) {
    (*regst_desc2offset)[order[i]] = order2offset[i];
  }
}

template<class T>
void MemReusedMemSizeFirstAlgo(const bool compact_insert,
                               const HashMap<T, std::pair<int32_t, int32_t>>& regst2lifetime,
                               const HashMap<T, size_t>& mem_reused_regst2size,
                               MemBlockResultInfo<T>* result) {
  std::vector<T> order;
  order.reserve(regst2lifetime.size());
  for (const auto& pair : regst2lifetime) { order.emplace_back(pair.first); }
  std::sort(order.begin(), order.end(), [&](T lhs, T rhs) {
    size_t l_value = mem_reused_regst2size.at(lhs);
    size_t r_value = mem_reused_regst2size.at(rhs);
    if (l_value == r_value) { return regst2lifetime.at(lhs).first < regst2lifetime.at(rhs).first; }
    return l_value > r_value;
  });
  MemReusedAlgorithmAllocateByOrder(compact_insert, order, mem_reused_regst2size, regst2lifetime,
                                    result);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_IN_JOB_MEM_SHARING_UTIL_H_
