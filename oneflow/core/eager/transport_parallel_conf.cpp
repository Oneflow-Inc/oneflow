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
#include "oneflow/core/common/util.h"
#include "oneflow/core/eager/transport_parallel_conf.h"

namespace oneflow {
namespace eager {

Maybe<ParallelConfGroupList> MakeTransportInstructionParallelConfs(
    const ParallelDesc& src_parallel_desc, const ParallelDesc& dst_parallel_desc) {
  CHECK_OR_RETURN(src_parallel_desc != dst_parallel_desc) << "no need to transport data";
  CHECK_EQ_OR_RETURN(src_parallel_desc.parallel_num(), dst_parallel_desc.parallel_num())
      << "only one-to-one supported";
  CHECK_EQ_OR_RETURN(src_parallel_desc.device_type(), kCPU) << "only cpu device supported";
  CHECK_EQ_OR_RETURN(dst_parallel_desc.device_type(), kCPU) << "only cpu device supported";
  auto parallel_confs = std::make_shared<ParallelConfGroupList>();

  // Initialize remainder_machine_id2parallel_ids with all parallel_ids
  std::map<int64_t, std::set<int64_t>> remainder_machine_id2parallel_ids;
  FOR_RANGE(int64_t, parallel_id, 0, src_parallel_desc.parallel_num()) {
    int64_t machine_id = JUST(src_parallel_desc.MachineId4ParallelId(parallel_id));
    remainder_machine_id2parallel_ids[machine_id].insert(parallel_id);
  }
  std::set<int64_t> last_src_machine_ids;
  std::set<int64_t> last_dst_machine_ids;
  while (!remainder_machine_id2parallel_ids.empty()) {
    std::set<int64_t> cur_parallel_ids;
    std::set<int64_t> cur_src_machine_ids;
    std::set<int64_t> cur_dst_machine_ids;
    for (const auto& pair : remainder_machine_id2parallel_ids) {
      CHECK(!pair.second.empty());
      int64_t parallel_id = *pair.second.begin();
      int64_t src_machine_id = JUST(src_parallel_desc.MachineId4ParallelId(parallel_id));
      int64_t dst_machine_id = JUST(dst_parallel_desc.MachineId4ParallelId(parallel_id));
      if (cur_src_machine_ids.count(src_machine_id) > 0) { continue; }
      if (cur_dst_machine_ids.count(dst_machine_id) > 0) { continue; }
      cur_parallel_ids.insert(parallel_id);
      cur_src_machine_ids.insert(src_machine_id);
      cur_dst_machine_ids.insert(dst_machine_id);
    }
    ParallelConfGroup* parallel_conf_group = nullptr;
    if (last_src_machine_ids == cur_src_machine_ids
        && last_dst_machine_ids == cur_dst_machine_ids) {
      int64_t cur_size = parallel_confs->parallel_conf_group_size();
      CHECK_GT_OR_RETURN(cur_size, 0);
      parallel_conf_group = parallel_confs->mutable_parallel_conf_group(cur_size - 1);
    } else {
      parallel_conf_group = parallel_confs->mutable_parallel_conf_group()->Add();
    }
    auto* parallel_conf = parallel_conf_group->mutable_parallel_conf()->Add();
    parallel_conf->set_device_tag("cpu");
    for (int64_t parallel_id : cur_parallel_ids) {
      int64_t machine_id = JUST(src_parallel_desc.MachineId4ParallelId(parallel_id));
      int64_t device_id = JUST(src_parallel_desc.DeviceId4ParallelId(parallel_id));
      parallel_conf->add_device_name(std::to_string(machine_id) + ":" + std::to_string(device_id));
      auto* remainder_parallel_ids = &remainder_machine_id2parallel_ids[machine_id];
      remainder_parallel_ids->erase(parallel_id);
      if (remainder_parallel_ids->empty()) { remainder_machine_id2parallel_ids.erase(machine_id); }
    }
    last_src_machine_ids = cur_src_machine_ids;
    last_dst_machine_ids = cur_dst_machine_ids;
  }
  return parallel_confs;
}

}  // namespace eager
}  // namespace oneflow
