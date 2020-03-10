#include "oneflow/xrt/tvm/tvm_executable.h"
#include <dlpack/dlpack.h>
#include <cstdint>

namespace oneflow {

namespace xrt {

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
    default:
      LOG(FATAL) << "Unsupport data type: " << data_type << " for Xrt with TVM";
  }
  return dl_type;
}

DLManagedTensor XrtParameter2DLManagedTensor(const Parameter& para, int32_t device_id) {
  DLManagedTensor ret;
  auto& tensor = ret.dl_tensor;
  CHECK(IsAligned(para.data(), tvm::runtime::kAllocAlignment));
  tensor.data = para.data();
  tensor.ctx.device_type = XrtDev2DLDev(XrtDevice::GPU_CUDA); // TODO(niuchong): support more device
  tensor.ctx.device_id = device_id;
  tensor.ndim = para.shape().NumAxes();
  tensor.shape = const_cast<int64_t*>(para.shape().dim_vec().data());
  tensor.dtype = XrtDType2DLDtype(para.data_type());
  tensor.byte_offset = 0;
  return ret;
}

}

TVMExecutable::TVMExecutable(const std::string& name, const int num_inputs,
    const std::vector<Parameter>& outputs,
    const std::string& json,
    const tvm::runtime::Module& built_mod,
    TVMContext ctx) :
      Executable(XrtEngine::TVM), name_(name), num_inputs_(num_inputs),
      outputs_(), ctx_(ctx) {
  {
    auto create_fn = tvm::runtime::Registry::Get("tvm.graph_runtime.create");
    executor_ = (*create_fn)(json, built_mod, (int)ctx_.device_type, (int)ctx_.device_id);
    run_ = executor_.GetFunction("run", false);
    set_input_zero_copy_ = executor_.GetFunction("set_input_zero_copy", false);
    get_output_ = executor_.GetFunction("get_output", false);
    get_num_outputs_ = executor_.GetFunction("get_num_outputs", false);
  }
  for (const auto& output : outputs) {
    outputs_.emplace_back(XrtParameter2DLManagedTensor(output, ctx_.device_id));
  }
}

bool TVMExecutable::Run(const std::vector<Parameter> &inputs, 
    const ExecutableRunOptions &run_options,
    bool block_until_done) {
  std::vector<DLManagedTensor> dl_managed_tensors;
  for (const auto& input : inputs) {
    dl_managed_tensors.emplace_back(XrtParameter2DLManagedTensor(input, ctx_.device_id));
    set_input_zero_copy_(input.name(),
        tvm::runtime::NDArray::FromDLPack(&dl_managed_tensors.back()));
  }

  run_();

  int num_outputs = get_num_outputs_();
  CHECK_EQ(num_outputs, outputs_.size());
  for (int i = 0;i < outputs_.size(); ++i) {
    get_output_(i, &outputs_.at(i));
  }

  if (block_until_done) {
    // TODO(niuchong): tvm async
  }

  this->results_ = run_options.return_params;
  return true;
}

}

}
