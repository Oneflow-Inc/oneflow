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
#ifndef ONEFLOW_CORE_FRAMEWORK_NN_GRAPH_UTILS_H_
#define ONEFLOW_CORE_FRAMEWORK_NN_GRAPH_UTILS_H_

#include <set>
#include <string>
#include "oneflow/core/common/env_var/env_var.h"
#include "oneflow/core/common/env_var/lazy.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/framework/nn_graph.h"
#include "oneflow/core/rpc/include/base.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

// A templated function that broadcasts data from the master process to worker processes in a
// multi-threaded manner. Return push/pull keys only in master process.
namespace oneflow {
template<typename X, typename Y>
std::set<std::string> MultiThreadBroadcastFromMasterToWorkers(size_t world_size,
                                                              const std::string& prefix,
                                                              const X& master_data,
                                                              Y* worker_data) {
  std::set<std::string> keys;
  const std::string& broadcast_strategy = ThreadLocalEnvString<ONEFLOW_BROADCAST_STRATEGY>();
  const size_t thread_num = ThreadLocalEnvInteger<ONEFLOW_LAZY_COMPILE_RPC_THREAD_NUM>();

  if (broadcast_strategy == "LOCAL_RANK_PROXY") {
    // The network communication inside the machine will be many times faster than the network
    // communication across machines. Therefore, all the GPUs inside the machine use intra-machine
    // transmission. The master is responsible for sending the data to the rank whose local_rank is
    // 0 on the machine and other rank whose local_rank is not 0 will pull kv from local_rank_0. let
    // n be the number of machines and k be the number of GPUs on each machine.

    // Through this strategy, only one cross-machine transmission
    // plus one intra-machine transmission is required,
    // while the original solution requires two cross-machine transmissions

    // Only when n <= k (make sure n <= 16 or 32), the strategy will have a good effect.
    // Other situations may not work well because the load on the master is too high.
    if (GlobalProcessCtx::IsThisProcessMaster()) {
      std::mutex mtx4keys;
      std::string data;
      master_data.SerializeToString(&data);
      const size_t node_size = Singleton<GlobalProcessCtx>::Get()->NodeSize();
      MultiThreadLoop(
          node_size,
          [&](int i) {
            const size_t single_node_process_num =
                Singleton<GlobalProcessCtx>::Get()->NumOfProcessPerNode();
            const size_t target_rank = single_node_process_num * i;
            std::string key = prefix + std::to_string(i);
            Singleton<CtrlClient>::Get()->PushRankKV(target_rank, key, data);
            std::lock_guard<std::mutex> lock(mtx4keys);
            CHECK(keys.insert(key).second);
          },
          thread_num);
    } else {
      const size_t rank = Singleton<GlobalProcessCtx>::Get()->Rank();
      const size_t local_rank = Singleton<GlobalProcessCtx>::Get()->LocalRank();
      const size_t node_id = Singleton<GlobalProcessCtx>::Get()->NodeId(rank);
      std::string key = prefix + std::to_string(node_id);
      const size_t target_rank = rank - local_rank;
      Singleton<CtrlClient>::Get()->PullRankKV(target_rank, key, worker_data);
    }

  } else {
    // other broadcast case
    const size_t split_num = std::sqrt(world_size);
    BalancedSplitter bs(world_size, split_num);
    if (GlobalProcessCtx::IsThisProcessMaster()) {
      std::mutex mtx4keys;
      std::string data;
      master_data.SerializeToString(&data);
      MultiThreadLoop(
          split_num,
          [&](int i) {
            std::string key = prefix + std::to_string(i);
            Singleton<CtrlClient>::Get()->PushKV(key, data);
            std::lock_guard<std::mutex> lock(mtx4keys);
            CHECK(keys.insert(key).second);
          },
          thread_num);
    } else {
      const int64_t bs_index = bs.GetRangeIndexForVal(GlobalProcessCtx::Rank());
      std::string key = prefix + std::to_string(bs_index);
      Singleton<CtrlClient>::Get()->PullKV(key, worker_data);
    }
  }

  return keys;
}
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_NN_GRAPH_UTILS_H_
