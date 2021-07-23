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
#include "oneflow/xrt/tvm/executable.h"
#include <dlpack/dlpack.h>
#include <cstdint>

namespace oneflow {
namespace xrt {
namespace of_tvm {

namespace {

bool IsAligned(void* data_ptr, std::uintptr_t alignment) {
  auto mask = alignment - 1;
  CHECK((alignment & mask) == 0) << "Wrong alignment: " << alignment;
  return (reinterpret_cast<std::uintptr_t>(data_ptr) & mask) == 0;
}

DLDeviceType XrtDev2DLDev(XrtDevice dev) {
  if (dev == XrtDevice::CPU_X86) {
    return DLDeviceType::kDLCPU;
  } else if (dev == XrtDevice::GPU_CUDA) {
    return DLDeviceType::kDLGPU;
  } else {
    LOG(FATAL) << "Unsupported DLDeviceType for XrtDevice: " << dev;
  }
}

DLDataType XrtDType2DLDtype(DataType data_type) {
  DLDataType dl_type;
  dl_type.lanes = 1;
  switch (data_type) {
    case DataType::kChar:
      dl_type.code = DLDataTypeCode::kDLInt;
      dl_type.bits = 8;
      break;
    case DataType::kFloat:
      dl_type.code = DLDataTypeCode::kDLFloat;
      dl_type.bits = 32;
      break;
    case DataType::kInt8:
      dl_type.code = DLDataTypeCode::kDLInt;
      dl_type.bits = 8;
      break;
    case DataType::kInt32:
      dl_type.code = DLDataTypeCode::kDLInt;
      dl_type.bits = 32;
      break;
    case DataType::kUInt8:
      dl_type.code = DLDataTypeCode::kDLUInt;
      dl_type.bits = 8;
      break;
    case DataType::kFloat16:
      dl_type.code = DLDataTypeCode::kDLFloat;
      dl_type.bits = 16;
      break;
    default: LOG(FATAL) << "Unsupport data type: " << data_type << " for Xrt with TVM";
  }
  return dl_type;
}

DLManagedTensor XrtParameter2DLManagedTensor(const Parameter& para, TVMContext ctx) {
  DLManagedTensor ret;
  ret.manager_ctx = nullptr;
  ret.deleter = nullptr;
  auto& tensor = ret.dl_tensor;
  // CHECK(IsAligned(para.data(), tvm::runtime::kAllocAlignment));
  tensor.data = para.data();
  tensor.ctx = ctx;
  tensor.ndim = para.shape().NumAxes();
  tensor.shape = const_cast<int64_t*>(para.shape().dim_vec().data());
  tensor.dtype = XrtDType2DLDtype(para.data_type());
  tensor.byte_offset = 0;
  return ret;
}

}  // namespace

TVMExecutable::TVMExecutable(const std::string& name, const int num_inputs,
                             const std::vector<Parameter>& outputs, const std::string& json,
                             const tvm::runtime::Module& built_mod, XrtDevice device)
    : Executable(name, XrtEngine::TVM),
      num_inputs_(num_inputs),
      outputs_(outputs),
      built_mod_(built_mod),
      graph_json_(json),
      device_(device),
      is_inited_(false) {}

bool TVMExecutable::Run(const std::vector<Parameter>& inputs,
                        const ExecutableRunOptions& run_options, bool block_until_done) {
  if (!is_inited_) {
    ctx_.device_type = XrtDev2DLDev(device_);
    ctx_.device_id = run_options.device_ordinal;

    auto create_fn = tvm::runtime::Registry::Get("tvm.graph_runtime.create");
    executor_ = (*create_fn)(graph_json_, built_mod_, (int)ctx_.device_type, (int)ctx_.device_id);
    run_ = executor_->GetFunction("run", false);
    set_input_ = executor_->GetFunction("set_input", false);
    get_output_ = executor_->GetFunction("get_output", false);
    get_num_outputs_ = executor_->GetFunction("get_num_outputs", false);

    for (const auto& output : outputs_) {
      output_dltensors_.emplace_back(XrtParameter2DLManagedTensor(output, ctx_));
    }
  }

  std::vector<DLManagedTensor> dl_managed_tensors;
  for (const auto& input : inputs) {
    dl_managed_tensors.emplace_back(XrtParameter2DLManagedTensor(input, ctx_));
    set_input_(input.name(), tvm::runtime::NDArray::FromDLPack(&dl_managed_tensors.back()));
  }

  run_();

  int num_outputs = get_num_outputs_();
  CHECK_EQ(num_outputs, outputs_.size());
  for (int i = 0; i < outputs_.size(); ++i) {
    get_output_(i, &(output_dltensors_.at(i).dl_tensor));
  }

  if (block_until_done) {
    // TODO(niuchong): tvm async
  }

  this->results_ = run_options.return_params;
  return true;
}  // of_tvm
}  // namespace of_tvm
}  // namespace xrt

}  // namespace oneflow
