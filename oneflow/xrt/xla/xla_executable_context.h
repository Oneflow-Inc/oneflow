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
#ifndef ONEFLOW_XRT_XLA_XLA_EXECUTABLE_CONTEXT_H_
#define ONEFLOW_XRT_XLA_XLA_EXECUTABLE_CONTEXT_H_

#include "oneflow/xrt/executable.h"
#include "oneflow/xrt/parameter.h"
#include "oneflow/xrt/types.h"
#include "oneflow/xrt/xla/xla_allocator.h"

#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/stream_executor/stream.h"

namespace oneflow {
namespace xrt {
namespace mola {

class XlaExecutableRunContext {
 public:
  XlaExecutableRunContext(const ExecutableRunOptions& run_options, const XrtDevice& device);

  virtual ~XlaExecutableRunContext() = default;

  const std::vector<xla::ShapedBuffer*>& PopulateInputs(
      const std::vector<Parameter>& inputs, const std::vector<xla::Shape>& input_shapes);

  // Populate output params to reuse the buffers in allocator. This helps
  // to reduce memory occupancy and avoid extra copy between temporary
  // buffers and output buffers.
  void PopulateResultBuffers(const std::vector<Parameter>& outputs,
                             xla::LocalExecutable* executable);

  // Returns run options.
  const ExecutableRunOptions& run_options() const { return run_options_; }

  // Returns device type.
  const XrtDevice& device() const { return device_; }
  // Returns device ordinal.
  int device_ordinal() const { return device_ordinal_; }
  // Returns xla local client.
  xla::LocalClient* client() const { return client_; }
  // Returns xla executing stream
  se::Stream* stream() const { return stream_.get(); }
  // Returns buffer allocator.
  XlaAllocator* allocator() const { return allocator_.get(); }

  Eigen::ThreadPoolDevice* host_device() const { return host_device_; }

  int64_t rng_seed() const {
    return (run_options_.random_seed) == -1 ? tensorflow::GetXLARandomSeed()
                                            : run_options_.random_seed;
  }

  // Reserve allocator memory size. Do nothing if the allocator's buffer
  // capacity >= size. Otherwise it should wait for all launched kernels
  // to finish, then resize the memory buffer thread-safety.
  void ReserveWorkspace(size_t size) { allocator_->ReserveWorkspace(size); }
  // Increase kernel launched count on the allocator.
  void LockWorkspace() { allocator_->LockWorkspace(); }
  // Decrease kernel launched count on the allocator.
  void UnlockWorkspace() { allocator_->UnlockWorkspace(); }

 private:
  ExecutableRunOptions run_options_;

  XrtDevice device_;
  int device_ordinal_ = -1;

  xla::LocalClient* client_;

  std::shared_ptr<se::Stream> stream_;
  std::shared_ptr<XlaAllocator> allocator_;

  Eigen::ThreadPoolDevice* host_device_;

  std::vector<std::shared_ptr<xla::ShapedBuffer>> shaped_buffers_;
  std::vector<xla::ShapedBuffer*> input_buffers_;
};

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_XLA_EXECUTABLE_CONTEXT_H_
