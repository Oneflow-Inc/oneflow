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
#include "oneflow/core/framework/tensor_impl.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/framework/dtype.h"

namespace oneflow {
namespace one {

namespace {

std::shared_ptr<const ParallelDesc> MakeParallelDescByDevice(const Device& device) {
  int64_t machine_id = GlobalProcessCtx::Rank() / GlobalProcessCtx::NumOfProcessPerNode();
  int64_t device_id = device.device_id();
  std::string machine_device_id = std::to_string(machine_id) + ":" + std::to_string(device_id);
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag(device.of_type());
  parallel_conf.add_device_name(machine_device_id);
  return std::make_shared<const ParallelDesc>(parallel_conf);
}

Maybe<const Device> MakeDeviceByParallelDesc(const ParallelDesc& parallel_desc) {
  std::string type = parallel_desc.device_tag();
  if (parallel_desc.device_tag() == "gpu") { type = "cuda"; }
  std::vector<std::string> machine_device_ids;
  for (const auto& item : parallel_desc.parallel_conf().device_name()) {
    machine_device_ids.emplace_back(item);
  }
  CHECK_EQ_OR_RETURN(machine_device_ids.size(), 1);
  const std::string& machine_device_id = machine_device_ids.at(0);
  size_t pos = machine_device_id.find(':');
  CHECK_NE_OR_RETURN(pos, std::string::npos) << "device_name: " << machine_device_id;
  std::string device_id = machine_device_id.substr(pos + 1);
  CHECK_EQ_OR_RETURN(device_id.find('-'), std::string::npos);
  CHECK_OR_RETURN(IsStrInt(device_id));
  return std::make_shared<const Device>(type, std::stoi(device_id));
}

}  // namespace

Maybe<void> TensorImpl::SyncBlobObject2Attributes(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  set_shape(blob_object->op_arg_blob_attr()->shape());
  DataType data_type = static_cast<DataType>(blob_object->op_arg_blob_attr()->get_dtype());
  const std::shared_ptr<DType>& dtype = JUST(DType::GetDTypeByDataType(data_type));
  set_dtype(dtype);
  return Maybe<void>::Ok();
}

void MirroredTensorImpl::set_device(const std::shared_ptr<const Device>& device) {
  device_ = device;
  parallel_desc_ = MakeParallelDescByDevice(*device);
}

Maybe<void> MirroredTensorImpl::set_parallel_desc(
    const std::shared_ptr<const ParallelDesc>& parallel_desc) {
  parallel_desc_ = parallel_desc;
  device_ = JUST(MakeDeviceByParallelDesc(*parallel_desc));
  return Maybe<void>::Ok();
}

Maybe<void> EagerMirroredTensorImpl::set_blob_object(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  blob_object_ = blob_object;
  JUST(SyncBlobObject2Attributes(blob_object));
  return Maybe<void>::Ok();
}

Maybe<void> EagerConsistentTensorImpl::set_blob_object(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  blob_object_ = blob_object;
  JUST(SyncBlobObject2Attributes(blob_object));
  return Maybe<void>::Ok();
}

}  // namespace one
}  // namespace oneflow
