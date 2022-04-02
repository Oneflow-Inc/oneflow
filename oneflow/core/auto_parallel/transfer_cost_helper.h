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

#ifndef ONEFLOW_CORE_AUTO_PARALLEL_TRANSFER_COST_HELPER_H_
#define ONEFLOW_CORE_AUTO_PARALLEL_TRANSFER_COST_HELPER_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class TransferCostHelper final {
 public:
  enum Type : int {
    kNcclAll2All = 0,
    kNcclAllReduce,
    kNcclAllGather,
    kNcclReduceScatter,
  };

  Maybe<double> AskSymmetricTransferCost(size_t data_size, Type type) const;
};

const TransferCostHelper& GetTransferCostHelper();

}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PARALLEL_TRANSFER_COST_HELPER_H_
