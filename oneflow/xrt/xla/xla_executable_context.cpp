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
#include "oneflow/xrt/xla/xla_executable_context.h"
#include "oneflow/xrt/xla/xla_macro.h"
#include "oneflow/xrt/xla/xla_resource_manager.h"

#include "tensorflow/compiler/jit/xla_lib/xla_runtime_util.h"

namespace oneflow {
namespace xrt {
namespace mola {

XlaExecutableRunContext::XlaExecutableRunContext(const ExecutableRunOptions &run_options,
                                                 const XrtDevice &device)
    : run_options_(run_options), device_(device) {
  client_ = resource_mgr::GetOrCreateLocalClient(device);
  device_ordinal_ = run_options_.device_ordinal;
  if (device_ordinal_ < 0) { device_ordinal_ = client_->default_device_ordinal(); }
  MOLA_CHECK_AND_ASSIGN(stream_, client_->mutable_backend()->BorrowStream(device_ordinal_));
  host_device_ = resource_mgr::GetOrCreateEigenHostDevice();

  DeviceBufferAllocator *buffer_allocator = resource_mgr::GetOrCreateBufferAllocator(
      device, run_options.stream, stream_.get(), device_ordinal_);
  allocator_.reset(new XlaAllocator(client_->platform(), buffer_allocator));
}

const std::vector<xla::ShapedBuffer *> &XlaExecutableRunContext::PopulateInputs(
    const std::vector<Parameter> &inputs, const std::vector<xla::Shape> &input_shapes) {
  const auto &return_params = run_options_.return_params;
  CHECK_GT(return_params.size(), 0) << "Need one output at least.";

  namespace se = tensorflow::se;
  // Translate input blobs to xla ShapedBuffer suitable running the executable
  shaped_buffers_.resize(inputs.size());
  input_buffers_.resize(inputs.size());
  for (int i = 0; i < input_shapes.size(); ++i) {
    const xla::Shape &shape = input_shapes[i];
    const xla::Shape on_device_shape =
        client_->backend().transfer_manager()->HostShapeToDeviceShape(shape);
    CHECK(!on_device_shape.IsTuple()) << "Tuple shape is not allowed for xla input buffers";
    int64_t data_size = inputs[i].byte_size();
    const char *data_ptr = inputs[i].data<char>();

    // Buffer is nullptr if the blob is body disabled. It should be assigned
    // by a real pointer to prevent check failure while runing the XLA
    // executable, so here we assign the first input or output buffer to it
    // since it's sure that this entry should never be modified at any time.
    if (data_size > 0 && !data_ptr) { data_ptr = return_params[0].data<char>(); }
    se::DeviceMemoryBase memory_base =
        se::DeviceMemoryBase(const_cast<char *>(data_ptr), data_size);
    shaped_buffers_[i] = std::make_shared<xla::ShapedBuffer>(
        /*on_host_shape=*/shape, /*on_device_shape=*/shape, client_->platform(), device_ordinal_);
    shaped_buffers_[i]->set_buffer(memory_base, /*index=*/{});
    input_buffers_[i] = shaped_buffers_[i].get();
  }
  return input_buffers_;
}

void XlaExecutableRunContext::PopulateResultBuffers(const std::vector<Parameter> &outputs,
                                                    xla::LocalExecutable *executable) {
  std::vector<int64_t> allocation_indices;
  xla::ResultAllocationIndices(executable, &allocation_indices);
  CHECK_EQ(outputs.size(), allocation_indices.size());

  std::vector<se::DeviceMemoryBase> device_buffers;
  device_buffers.reserve(outputs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    char *data = outputs[i].data<char>();
    device_buffers.emplace_back(data, outputs[i].byte_size());
  }
  allocator_->PopulateDeviceMemory(device_buffers, allocation_indices);
}

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
