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

#include "oneflow/core/vm/lazy_job_stream_policy.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/framework/nn_graph_if.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

void LazyJobStreamPolicy::WaitUntilQueueEmptyIfFrontNNGraphNotEquals(
    const std::shared_ptr<NNGraphIf>& nn_graph) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (queue_.empty()) { return; }
  const auto& last_nn_graph = queue_.front().lock();
  if (!last_nn_graph) { return; }
  if (last_nn_graph == nn_graph) { return; }
  cond_.wait(lock, [this]() { return queue_.empty(); });
}

void LazyJobStreamPolicy::EnqueueNNGraph(const std::shared_ptr<NNGraphIf>& nn_graph) {
  std::unique_lock<std::mutex> lock(mutex_);
  queue_.emplace(nn_graph);
}

void LazyJobStreamPolicy::DequeueNNGraph() {
  std::unique_lock<std::mutex> lock(mutex_);
  queue_.pop();
  cond_.notify_all();
}

void LazyJobStreamPolicy::InitInstructionStatus(const Stream& stream,
                                                InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(NaiveInstrStatusQuerier) < kInstructionStatusBufferBytes, "");
  NaiveInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer());
}

void LazyJobStreamPolicy::DeleteInstructionStatus(const Stream& stream,
                                                  InstructionStatusBuffer* status_buffer) const {
  auto* ptr = NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer());
  ptr->~NaiveInstrStatusQuerier();
}

bool LazyJobStreamPolicy::QueryInstructionStatusLaunched(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return NaiveInstrStatusQuerier::Cast(status_buffer.buffer())->launched();
}

bool LazyJobStreamPolicy::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return NaiveInstrStatusQuerier::Cast(status_buffer.buffer())->done();
}

void LazyJobStreamPolicy::Run(Instruction* instruction) const { instruction->Compute(); }

}  // namespace vm
}  // namespace oneflow
