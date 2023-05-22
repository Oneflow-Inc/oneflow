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
#include "oneflow/user/kernels/unique_kernel_util.h"

namespace oneflow {

template<typename KEY, typename IDX>
struct UniqueKernelUtil<DeviceType::kCPU, KEY, IDX> {
  static void Unique(ep::Stream* stream, int64_t n, const KEY* in, IDX* num_unique, KEY* unique_out,
                     IDX* idx_out, void* workspace, int64_t workspace_size_in_bytes, bool sorted) {
    UniqueKernelUtil<DeviceType::kCPU, KEY, IDX>::UniqueWithCounts(
        stream, n, in, num_unique, unique_out, idx_out, nullptr, workspace, workspace_size_in_bytes,
        sorted);
  }
  static void UniqueWithCounts(ep::Stream* stream, int64_t n, const KEY* in, IDX* num_unique,
                               KEY* unique_out, IDX* idx_out, IDX* count, void* workspace,
                               int64_t workspace_size_in_bytes, bool sorted) {
    std::vector<int64_t> sorted_idx(n);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    if (sorted) {
      std::sort(sorted_idx.begin(), sorted_idx.end(),
                [&in](size_t a, size_t b) { return in[a] < in[b]; });
    }

    HashMap<KEY, IDX> map;
    for (int64_t i : sorted_idx) {
      KEY in_i = in[i];
      auto it = map.find(in_i);
      if (it == map.end()) {
        IDX idx = map.size();
        if (count != nullptr) { count[idx] = 1; }
        idx_out[i] = idx;
        unique_out[idx] = in_i;
        map[in_i] = idx;
      } else {
        IDX idx = it->second;
        if (count != nullptr) { count[idx] += 1; }
        idx_out[i] = idx;
      }
    }
    *num_unique = map.size();
  }

  static void GetUniqueWorkspaceSizeInBytes(ep::Stream* stream, int64_t n,
                                            int64_t* workspace_size_in_bytes) {
    *workspace_size_in_bytes = 1;
  }
  static void GetUniqueWithCountsWorkspaceSizeInBytes(ep::Stream* stream, int64_t n,
                                                      int64_t* workspace_size_in_bytes) {
    *workspace_size_in_bytes = 1;
  }
};

#define INSTANTIATE_UNIQUE_KERNEL_UTIL_CPU(key_type_pair, idx_type_pair)              \
  template struct UniqueKernelUtil<DeviceType::kCPU, OF_PP_PAIR_FIRST(key_type_pair), \
                                   OF_PP_PAIR_FIRST(idx_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_UNIQUE_KERNEL_UTIL_CPU, ARITHMETIC_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ);
#undef INSTANTIATE_UNIQUE_KERNEL_UTIL_CPU

}  // namespace oneflow
