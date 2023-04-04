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
#include <gtest/gtest.h>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <unistd.h>
#include <arpa/inet.h>
#include <vector>
#include <chrono>
#include "oneflow/core/common/env_var/env_var.h"
#include "oneflow/core/common/env_var/lazy.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/framework/nn_graph.h"
#include "oneflow/core/framework/shut_down_util.h"
#include "oneflow/core/job/env.pb.h"
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
  explicit DistributeOneFlowEnv(size_t rank, size_t world_size) {
    EnvProto env_proto;
    CompleteEnvProto(env_proto, rank, world_size);
    env_ctx_ = std::make_shared<EnvGlobalObjectsScope>(env_proto);
  }
  ~DistributeOneFlowEnv() { env_ctx_.reset(); }

  void CompleteEnvProto(EnvProto& env_proto, size_t rank, size_t world_size) {
    auto bootstrap_conf = env_proto.mutable_ctrl_bootstrap_conf();
    auto master_addr = bootstrap_conf->mutable_master_addr();
    const std::string addr = "127.0.0.1";
    // 大概率不会被占用的连续1000个端口范围是从49152到50151
    size_t master_port = 49152;
    size_t port = master_port + rank;

    master_addr->set_host(addr);
    master_addr->set_port(master_port);

    bootstrap_conf->set_world_size(world_size);
    bootstrap_conf->set_rank(rank);
    bootstrap_conf->set_ctrl_port(port);
    LOG(INFO) << "rank " << rank << " binds on " << port << " port";
    bootstrap_conf->set_host("127.0.0.1");

    auto cpp_logging_conf = env_proto.mutable_cpp_logging_conf();
    if (HasEnvVar("GLOG_log_dir")) { cpp_logging_conf->set_log_dir(GetEnvVar("GLOG_log_dir", "")); }
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

  // int32_t SetPort(size_t port) {
  //   int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  //   CHECK_GE(sock, 0) << "fail to find a free port.";
  //   int optval = 1;
  //   setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

  //   LOG(INFO) << "try to bind " << port;
  //   struct sockaddr_in sockaddr {};
  //   memset(&sockaddr, 0, sizeof(sockaddr));
  //   sockaddr.sin_family = AF_INET;
  //   sockaddr.sin_port = htons(port);
  //   // 单机多进程
  //   sockaddr.sin_addr.s_addr = inet_addr("127.0.0.1");
  //   int error = bind(sock, (struct sockaddr*)&sockaddr, sizeof(sockaddr));
  //   if (error == 0) { return port; }
  //   return -1;
  // }

 private:
  std::shared_ptr<EnvGlobalObjectsScope> env_ctx_;
};

class TestEnvScope {
 public:
  explicit TestEnvScope(size_t rank, size_t world_size) {
    if (Singleton<DistributeOneFlowEnv>::Get() == nullptr) {
      Singleton<DistributeOneFlowEnv>::New(rank, world_size);
    }
    SetShuttingDown(false);
  }
  ~TestEnvScope() {
    if (Singleton<DistributeOneFlowEnv>::Get() != nullptr) {
      Singleton<DistributeOneFlowEnv>::Delete();
    }
    SetShuttingDown();
  }
};

template<typename X, typename Y>
std::set<std::string> MultiThreadBroadcastFromMasterToWorkers(size_t world_size,
                                                              const std::string& prefix,
                                                              const X& master_data,
                                                              Y* worker_data) {
  const size_t thread_num = ThreadLocalEnvInteger<ONEFLOW_LAZY_COMPILE_RPC_THREAD_NUM>();
  const size_t split_num = std::sqrt(world_size);
  BalancedSplitter bs(world_size, split_num);
  std::set<std::string> keys;
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
    const int64_t bs_index = bs.GetRangIndex(GlobalProcessCtx::Rank());
    std::string key = prefix + std::to_string(bs_index);
    Singleton<CtrlClient>::Get()->PullKV(key, worker_data);
  }
  return keys;
}

template<typename X>
std::set<std::string> DivideConquerBroadcastMasterSide(
    const std::vector<size_t>& target_worker_ranks, const std::string& prefix,
    const X& master_data) {
  const size_t thread_num = ThreadLocalEnvInteger<ONEFLOW_LAZY_COMPILE_RPC_THREAD_NUM>();
  std::set<std::string> keys;
  std::string data;
  std::mutex mtx4keys;
  master_data.SerializeToString(&data);

  MultiThreadLoop(
      target_worker_ranks.size(),
      [&](int i) {
        std::string key = prefix + std::to_string(target_worker_ranks[i]);
        Singleton<CtrlClient>::Get()->PushKV(key, data);
        std::lock_guard<std::mutex> lock(mtx4keys);
        CHECK(keys.insert(key).second);
      },
      thread_num);
  return keys;
}

template<typename Y>
std::set<std::string> DivideConquerBroadcastWorkerSide(
    const std::vector<size_t>& target_worker_ranks, const std::string& prefix, Y* worker_data) {
  if (target_worker_ranks.empty()) { return {}; }
  const size_t thread_num = ThreadLocalEnvInteger<ONEFLOW_LAZY_COMPILE_RPC_THREAD_NUM>();
  std::set<std::string> keys;
  std::string data;
  std::string key = prefix + std::to_string(GlobalProcessCtx::Rank());
  std::mutex mtx4keys;
  Singleton<CtrlClient>::Get()->PullKV(key, worker_data);

  MultiThreadLoop(
      target_worker_ranks.size(),
      [&](int i) {
        std::string key = prefix + std::to_string(target_worker_ranks[i]);
        Singleton<CtrlClient>::Get()->PushKV(key, data);
        std::lock_guard<std::mutex> lock(mtx4keys);
        CHECK(keys.insert(key).second);
      },
      thread_num);

  return keys;
}

void GetResponsibleWorkers(std::vector<std::vector<size_t>>* target_workers, size_t start,
                           size_t end) {
  if (end > start) { return; }
  size_t mid = start + (end - start) / 2;
  (*target_workers)[start].emplace_back(mid);
  GetResponsibleWorkers(target_workers, start, mid);
  GetResponsibleWorkers(target_workers, mid + 1, end);
}

template<typename X, typename Y>
std::set<std::string> MultiThreadDivideConquerBroadcastFromMasterToWokers(
    size_t world_size, const std::string& prefix,
    const std::vector<std::vector<size_t>>& target_workers, const X& master_data, Y* worker_data) {
  std::set<std::string> keys;
  if (GlobalProcessCtx::IsThisProcessMaster()) {
    keys = DivideConquerBroadcastMasterSide(target_workers[0], prefix, master_data);
  } else {
    size_t rank = Singleton<GlobalProcessCtx>::Get()->Rank();
    keys = DivideConquerBroadcastWorkerSide(target_workers[rank], prefix, worker_data);
  }
  return keys;
}

class MultiThreadBroadCastTest : public ::testing::Test {
 protected:
  void SetUp() override {
    world_size_ = WORLD_SIZE;
    prefix_ = "test";
    std::string data;  // 10MB
    data.resize(10 * 1024 * 1024);
    master_data_.ParseFromString(data);
  }

  size_t world_size_;
  std::string prefix_;
  Job master_data_;
  Job worker_data_;
};

// /**
TEST_F(MultiThreadBroadCastTest, ProxyPerformanceTest) {
  // 创建1000个进程
  size_t rank = 0;
  int status = 0;

  for (size_t i = 1; i < world_size_; ++i) {
    pid_t pid = fork();
    if (pid == 0) {
      rank = i;
      break;
    } else if (pid < 0) {
      exit(EXIT_FAILURE);
    }
  }

  if (rank != 0) {
    // workers waiting for master
    sleep(1);
  }
  TestEnvScope scope(rank, world_size_);

  LOG(INFO) << "world size: " << Singleton<GlobalProcessCtx>::Get()->WorldSize();
  auto start_time = std::chrono::high_resolution_clock::now();
  std::set<std::string> keys =
      MultiThreadBroadcastFromMasterToWorkers(world_size_, prefix_, master_data_, &worker_data_);
  // LOG(INFO) << "key's size: " << keys.size();
  // synchronize all process

  auto end_tiem = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_tiem - start_time);

  LOG(INFO) << "Rank " << rank << " spend time: " << duration.count() << " ms";

  for (size_t i = 1; i < world_size_; ++i) {
    pid_t pid = wait(&status);
    if (pid == -1) {
      LOG(ERROR) << "wait error";
      exit(EXIT_FAILURE);
    }
  }
}

/**
TEST_F(MultiThreadBroadCastTest, DivideConquerPerformanceTest) {
  // 下面这两步提到进程初始化中
  std::vector<std::vector<size_t>> target_workers(world_size_);
  GetResponsibleWorkers(&target_workers, 0, world_size_ - 1);

  size_t rank = 0;
  int status = 0;
  for (size_t i = 1; i < world_size_; ++i) {
    pid_t pid = fork();
    if (pid == 0) {
      rank = i;
      break;
    } else if (pid < 0) {
      exit(EXIT_FAILURE);
    }
  }

  TestEnvScope scope(rank, world_size_);
  auto start_time = std::chrono::high_resolution_clock::now();
  std::set<std::string> keys = MultiThreadDivideConquerBroadcastFromMasterToWokers(
      world_size_, prefix_, target_workers, master_data_, &worker_data_);
  // synchronize all process

  auto end_tiem = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_tiem - start_time);

  LOG(INFO) << "Rank " << rank << " spend time: " << duration.count() << " ms";

  for (size_t i = 1; i < world_size_; ++i) {
    pid_t pid = wait(&status);
    if (pid == -1) {
      LOG(ERROR) << "wait error";
      exit(EXIT_FAILURE);
    }
  }
}
*/