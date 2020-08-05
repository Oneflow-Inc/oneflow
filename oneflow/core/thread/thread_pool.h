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
#ifndef ONEFLOW_CORE_THREAD_THREAD_POOL_H_
#define ONEFLOW_CORE_THREAD_THREAD_POOL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/channel.h"

namespace oneflow {

class ThreadPool final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ThreadPool);
  ThreadPool() = delete;
  ThreadPool(int32_t thread_num);
  ~ThreadPool();

  int32_t thread_num() const { return threads_.size(); }
  void AddWork(const std::function<void()>& work);

 private:
  std::vector<Channel<std::function<void()>>> work_chans_;
  std::vector<std::thread> threads_;

  std::atomic<size_t> work_cnt_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_THREAD_POOL_H_
