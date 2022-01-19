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
#include "oneflow/core/thread/thread_pool.h"

namespace oneflow {

ThreadPool::ThreadPool(int32_t thread_num)
    : work_chans_(thread_num), threads_(thread_num), work_cnt_(0) {
  FOR_RANGE(int32_t, i, 0, thread_num) {
    Channel<std::function<void()>>* chan = &(work_chans_.at(i));
    threads_[i] = std::thread([chan]() {
      std::function<void()> work;
      while (chan->Receive(&work) == kChannelStatusSuccess) { work(); }
    });
  }
}

ThreadPool::~ThreadPool() {
  FOR_RANGE(int32_t, i, 0, work_chans_.size()) {
    work_chans_.at(i).Close();
    threads_.at(i).join();
  }
}

void ThreadPool::AddWork(const std::function<void()>& work) {
  const size_t cur_chan_idx =
      work_cnt_.fetch_add(1, std::memory_order_relaxed) % work_chans_.size();
  work_chans_.at(cur_chan_idx).Send(work);
}

}  // namespace oneflow
