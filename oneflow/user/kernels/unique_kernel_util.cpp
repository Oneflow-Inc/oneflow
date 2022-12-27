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
                     IDX* idx_out, void* workspace, int64_t workspace_size_in_bytes,
                     const bool sorted) {
    UniqueKernelUtil<DeviceType::kCPU, KEY, IDX>::UniqueWithCounts(
        stream, n, in, num_unique, unique_out, idx_out, nullptr, workspace, workspace_size_in_bytes,
        sorted);
  }
  static void UniqueWithCounts(ep::Stream* stream, int64_t n, const KEY* in, IDX* num_unique,
                               KEY* unique_out, IDX* idx_out, IDX* count, void* workspace,
                               int64_t workspace_size_in_bytes, const bool sorted) {
    HashMap<KEY, IDX> map;
    FOR_RANGE(int64_t, i, 0, n) {
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

    if (sorted) {
      /*HashMap cannot be sorted, here the key is sorted using the auxiliary Vector.
      After that the index correction of the output is performed.*/
      IDX index_now = 0;
      std::vector<KEY> map_keys(map.size());
      for (auto it = map.begin(); it != map.end(); ++it) {
        map_keys[index_now] = it->first;
        if (count != nullptr) { count[index_now] = 0; }
        index_now++;
      }
      std::sort(map_keys.begin(), map_keys.end());

      for (int i = 0; i < map.size(); i++) { map[map_keys[i]] = i; }

      FOR_RANGE(int64_t, i, 0, n) {
        KEY in_i = in[i];
        auto it = map.find(in_i);
        IDX idx = it->second;
        idx_out[i] = idx;
        unique_out[idx] = in_i;
        if (count != nullptr) { count[idx] += 1; }
      }
    }
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
