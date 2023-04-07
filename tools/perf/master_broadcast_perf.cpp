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
#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <unistd.h>
#include <arpa/inet.h>
#include <string>
#include <vector>
#include <chrono>
#include "oneflow/core/common/env_var/env_var.h"
#include "oneflow/core/common/env_var/lazy.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/framework/nn_graph.h"
#include "oneflow/core/framework/shut_down_util.h"
#include "oneflow/core/rpc/include/base.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/api/cpp/api.h"

using namespace oneflow;

#define WORLD_SIZE 3

bool HasEnvVar(const std::string& key) {
  const char* value = getenv(key.c_str());
  return value != nullptr;
}

std::string GetEnvVar(const std::string& key, const std::string& default_value) {
  const char* value = getenv(key.c_str());
  if (value == nullptr) { return default_value; }
  return std::string(value);
}

int64_t GetEnvVar(const std::string& key, int64_t default_value) {
  const char* value = getenv(key.c_str());
  if (value == nullptr) { return default_value; }
  return std::atoll(value);
}

class DistributeOneFlowEnv {
 public:
  explicit DistributeOneFlowEnv(size_t world_size, size_t rank, const std::string& master_host,
                                size_t master_port, const std::string& ctrl_host, size_t ctrl_port,
                                size_t node_size) {
    EnvProto env_proto;
    CompleteEnvProto(env_proto, rank, world_size, master_host, master_port, ctrl_host, ctrl_port,
                     node_size);
    env_ctx_ = std::make_shared<EnvGlobalObjectsScope>(env_proto);
  }
  ~DistributeOneFlowEnv() { env_ctx_.reset(); }

  void CompleteEnvProto(EnvProto& env_proto, size_t rank, size_t world_size,
                        const std::string& master_host, size_t master_port,
                        const std::string& ctrl_host, size_t ctrl_port, size_t node_size) {
    auto bootstrap_conf = env_proto.mutable_ctrl_bootstrap_conf();
    auto master_addr = bootstrap_conf->mutable_master_addr();
    // TODO: addr和port作为参数传入
    // const std::string addr = "127.0.0.1";
    // size_t master_port = 49155;
    // Adding 1 here is to ensure that master_port cannot overlap with ctrl_port on the master.
    // size_t port = master_port + rank + 1;

    master_addr->set_host(master_host);
    master_addr->set_port(master_port);

    bootstrap_conf->set_world_size(world_size);
    bootstrap_conf->set_rank(rank);
    bootstrap_conf->set_host(ctrl_host);
    bootstrap_conf->set_ctrl_port(ctrl_port);
    bootstrap_conf->set_node_size(node_size);

    auto cpp_logging_conf = env_proto.mutable_cpp_logging_conf();
    if (HasEnvVar("GLOG_log_dir")) {
      cpp_logging_conf->set_log_dir(GetEnvVar("GLOG_log_dir", ""));
      LOG(INFO) << "LOG DIR: " << cpp_logging_conf->log_dir();
    }
    if (HasEnvVar("GLOG_logtostderr")) {
      cpp_logging_conf->set_logtostderr(GetEnvVar("GLOG_logtostderr", -1));
    }
    if (HasEnvVar("GLOG_logbuflevel")) {
      cpp_logging_conf->set_logbuflevel(GetEnvVar("GLOG_logbuflevel", -1));
    }
    if (HasEnvVar("GLOG_minloglevel")) {
      cpp_logging_conf->set_minloglevel(GetEnvVar("GLOG_minloglevel", -1));
    }
  }

 private:
  std::shared_ptr<EnvGlobalObjectsScope> env_ctx_;
};

class TestEnvScope {
 public:
  explicit TestEnvScope(size_t world_size, size_t rank, const std::string& master_host,
                        size_t master_port, const std::string& ctrl_host, size_t ctrl_port,
                        size_t node_size) {
    if (Singleton<DistributeOneFlowEnv>::Get() == nullptr) {
      Singleton<DistributeOneFlowEnv>::New(world_size, rank, master_host, master_port, ctrl_host,
                                           ctrl_port, node_size);
    }
  }

  ~TestEnvScope() {
    if (Singleton<DistributeOneFlowEnv>::Get() != nullptr) {
      Singleton<DistributeOneFlowEnv>::Delete();
    }
  }
};

template<typename X, typename Y>
std::set<std::string> MultiThreadBroadcastFromMasterToWorkers(size_t world_size,
                                                              const std::string& prefix,
                                                              const X& master_data,
                                                              Y* worker_data) {
  std::set<std::string> keys;
  char* broadcast_strategy = std::getenv("BROADCAST_STRATEGY");
  const size_t thread_num = ThreadLocalEnvInteger<ONEFLOW_LAZY_COMPILE_RPC_THREAD_NUM>();
  // optimize n <= k case
  if (broadcast_strategy && std::string(broadcast_strategy) == "LOCAL_RANK_PROXY") {
    if (GlobalProcessCtx::IsThisProcessMaster()) {
      std::mutex mtx4keys;
      // Unlike the implementation within the framework, here we directly transmit std::string.
      const std::string& data = master_data;
      size_t node_size = Singleton<GlobalProcessCtx>::Get()->NodeSize();
      MultiThreadLoop(
          node_size,
          [&](int i) {
            const size_t single_node_process_num =
                Singleton<GlobalProcessCtx>::Get()->NumOfProcessPerNode();
            size_t target_rank = single_node_process_num * i;
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
      size_t target_rank = rank - local_rank;
      Singleton<CtrlClient>::Get()->PullRankKV(target_rank, key, worker_data);
    }

    // other broadcast case
  } else {
    const size_t split_num = std::sqrt(world_size);
    BalancedSplitter bs(world_size, split_num);
    if (GlobalProcessCtx::IsThisProcessMaster()) {
      std::mutex mtx4keys;
      // Unlike the implementation within the framework, here we directly transmit std::string.
      const std::string& data = master_data;
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
      const int64_t bs_index = bs.GetRangIndex(GlobalProcessCtx::Rank());
      std::string key = prefix + std::to_string(bs_index);
      Singleton<CtrlClient>::Get()->PullKV(key, worker_data);
    }
  }

  return keys;
}

int main(int argc, char* argv[]) {
  if (argc == 1) {
    LOG(FATAL) << "Error: must set world_size, rank, master_host, master_port, ctrl_host, "
                  "ctrl_port, iteration_num, data_size and node_size";
  }
  size_t world_size = std::stoi(argv[1]);
  size_t rank = std::stoi(argv[2]);
  std::string master_host = argv[3];
  size_t master_port = std::stoi(argv[4]);
  std::string ctrl_host = argv[5];
  size_t ctrl_port = std::stoi(argv[6]);
  size_t iteration_num = std::stoi(argv[7]);
  size_t data_size = std::stoi(argv[8]);
  size_t node_size = std::stoi(argv[9]);

  std::string master_data, worker_data;
  if (rank == 0) { master_data.resize(data_size * 1024); }

  TestEnvScope scope(world_size, rank, master_host, master_port, ctrl_host, ctrl_port, node_size);

  auto rank_duration = std::chrono::milliseconds::zero();
  auto total_duration = std::chrono::milliseconds::zero();
  for (size_t i = 0; i < iteration_num; ++i) {
    std::string prefix("test" + std::to_string(i));
    auto start_time = std::chrono::high_resolution_clock::now();

    std::set<std::string> keys =
        MultiThreadBroadcastFromMasterToWorkers(world_size, prefix, master_data, &worker_data);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    rank_duration += duration;

    // sync all process
    Singleton<CtrlClient>::Get()->Barrier("iteration_" + std::to_string(i));
    if (rank == 0) {
      end_time = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
      total_duration += rank_duration;
    }
  }

  auto average_rank_duration = rank_duration / iteration_num;
  // push perf result to master
  if (rank != 0) {
    nlohmann::json result_json;
    result_json["rank"] = rank;
    result_json["cost_time"] = std::to_string(average_rank_duration.count()) + " ms";
    std::string key = "result_of_rank_" + std::to_string(rank);
    // Singleton<CtrlClient>::Get()->PushMasterKV(key, result_json.dump());
    // PushMasterKV不支持std::string
    Singleton<CtrlClient>::Get()->PushKV(key, result_json.dump());
  }

  std::vector<nlohmann::json> results;
  if (rank == 0) {
    nlohmann::json result_json;
    // add master result
    result_json["rank_0"] = std::to_string(average_rank_duration.count()) + " ms";

    // add worker result
    for (size_t i = 1; i < world_size; ++i) {
      std::string key = "result_of_rank_" + std::to_string(i);
      std::string result_data;
      Singleton<CtrlClient>::Get()->PullKV(key, &result_data);
      nlohmann::json result = nlohmann::json::parse(result_data);
      size_t rank = result["rank"];
      std::string cost_time = result["cost_time"];
      result_json["rank_" + std::to_string(rank)] = cost_time;
    }

    auto average_total_duration = total_duration / iteration_num;
    result_json["total_cost"] = std::to_string(average_total_duration.count()) + " ms";
    // dump json
    std::string filename = "result.json";
    std::ofstream output_file(filename);
    output_file << result_json.dump();
    output_file.close();
  }
  LOG(INFO) << "Finish perf";

  return 0;
}