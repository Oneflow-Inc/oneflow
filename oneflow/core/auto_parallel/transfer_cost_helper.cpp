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

#include "oneflow/core/auto_parallel/transfer_cost_helper.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

namespace {

int32_t GetTransferCostConfigIdx4NodeNum(int32_t node_num) {
  if (node_num < 4) {
    return 0;
  } else if (node_num < 8) {
    return 1;
  } else {
    return 2;
  }
}

}  // namespace

Maybe<double> TransferCostHelper::AskSymmetricTransferCost(size_t data_size, Type type) const {
  const TransferCostConfig& transfer_config =
      Global<ResourceDesc, ForSession>::Get()->GetTransferCostConfig();
  int32_t node_num = Global<ResourceDesc, ForSession>::Get()->GpuDeviceNum();
  int32_t idx = GetTransferCostConfigIdx4NodeNum(node_num);

  const TransferCostConfig::Value* value = nullptr;
  switch (type) {
    case kNcclAll2All: value = &transfer_config.nccl_all2all().at(idx); break;
    case kNcclAllReduce: value = &transfer_config.nccl_all_reduce().at(idx); break;
    case kNcclAllGather: value = &transfer_config.nccl_all_gather().at(idx); break;
    case kNcclReduceScatter: value = &transfer_config.nccl_reduce_scatter().at(idx); break;
    default: return Error::TypeError() << "Unknown transfer type";
  }
  double k = data_size < value->x0() ? value->k1() : value->k2();
  return k * (data_size - value->x0()) + value->b();
}

const TransferCostHelper& GetTransferCostHelper() {
  static TransferCostHelper helper;
  return helper;
}

}  // namespace oneflow
