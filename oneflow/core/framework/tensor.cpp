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
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/framework/op_interpreter/eager_mirrored_op_interpreter.h"

namespace oneflow {

namespace one {

/*static*/ Maybe<MirroredTensor> MirroredTensor::MakeTensor(
    const std::shared_ptr<const Shape>& shape, DataType dtype,
    const std::shared_ptr<const Device>& device, bool is_lazy, bool requires_grad, bool is_leaf) {
  const auto& tensor_meta =
      std::make_shared<MirroredTensorMeta>(std::make_shared<Shape>(*shape), dtype, device);
  if (is_lazy) {
    const auto& impl =
        std::make_shared<LazyMirroredTensorImpl>(tensor_meta, requires_grad, is_leaf);
    return std::make_shared<MirroredTensor>(impl);
  } else {
    const auto& impl =
        std::make_shared<EagerMirroredTensorImpl>(tensor_meta, requires_grad, is_leaf);
    const auto& tensor = std::make_shared<MirroredTensor>(impl);
    const auto& outputs = std::make_shared<TensorTuple>();
    outputs->push_back(tensor);
    JUST(RunEmptyOp(outputs.get()));
    return tensor;
  }
}

bool MirroredTensor::is_cuda() const { return CHECK_JUST(device())->type() == "cuda"; }

int64_t MirroredTensor::ndim() const { return shape()->NumAxes(); }

int64_t MirroredTensor::dim(int64_t index) const { return shape()->At(index); }

int64_t MirroredTensor::nelement() const { return shape()->elem_cnt(); }

std::shared_ptr<MirroredTensor> MirroredTensor::data() const {
  std::shared_ptr<MirroredTensor> t = std::make_shared<MirroredTensor>(impl_);
  return t;
}

Maybe<MirroredTensor> MirroredTensor::api_detach() const {
  return std::make_shared<MirroredTensor>(JUST(impl_->detach()));
}

Maybe<ConsistentTensor> ConsistentTensor::MakeTensor(
    const std::shared_ptr<const Shape>& shape, DataType dtype,
    Symbol<cfg::ParallelDistribution> parallel_distribution, Symbol<ParallelDesc> parallel_desc,
    bool is_lazy, bool requires_grad, bool is_leaf) {
  std::shared_ptr<ConsistentTensorImpl> impl;
  Symbol<ConsistentTensorMeta> consistent_tensor_meta(
      ConsistentTensorMeta(shape, dtype, parallel_distribution, parallel_desc));
  if (is_lazy) {
    impl =
        std::make_shared<LazyConsistentTensorImpl>(consistent_tensor_meta, requires_grad, is_leaf);
  } else {
    impl = JUST(EagerConsistentTensorImpl::New(consistent_tensor_meta, requires_grad, is_leaf));
  }
  return std::make_shared<ConsistentTensor>(impl);
}

bool ConsistentTensor::is_cuda() const {
  return CHECK_JUST(parallel_desc())->device_type() == DeviceType::kGPU;
}

int64_t ConsistentTensor::dim(int64_t index) const { return shape()->At(index); }

int64_t ConsistentTensor::nelement() const { return shape()->elem_cnt(); }

int64_t ConsistentTensor::ndim() const { return shape()->NumAxes(); }

std::shared_ptr<ConsistentTensor> ConsistentTensor::data() const {
  std::shared_ptr<ConsistentTensor> t = std::make_shared<ConsistentTensor>(impl_);
  return t;
}

Maybe<ConsistentTensor> ConsistentTensor::api_detach() const {
  std::shared_ptr<ConsistentTensor> t = std::make_shared<ConsistentTensor>(impl_);
  return t;
}

}  // namespace one

}  // namespace oneflow
