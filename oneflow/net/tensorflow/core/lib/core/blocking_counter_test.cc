/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/platform/test.h"

#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace {

TEST(BlockingCounterTest, TestZero) {
  BlockingCounter bc(0);
  bc.Wait();
}

TEST(BlockingCounterTest, TestSingleThread) {
  BlockingCounter bc(2);
  bc.DecrementCount();
  bc.DecrementCount();
  bc.Wait();
}

TEST(BlockingCounterTest, TestMultipleThread) {
  int N = 3;
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test", N);

  BlockingCounter bc(N);
  for (int i = 0; i < N; ++i) {
    thread_pool->Schedule([&bc] { bc.DecrementCount(); });
  }

  bc.Wait();
  delete thread_pool;
}

}  // namespace
}  // namespace tensorflow
