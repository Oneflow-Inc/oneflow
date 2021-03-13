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
#include "oneflow/core/autograd/autograd_engine.h"

namespace oneflow {

namespace one {

std::shared_ptr<MirroredTensor> MirroredTensor::MakeTensor(
    const std::shared_ptr<const Shape>& shape, const std::shared_ptr<const DType>& dtype,
    const std::shared_ptr<const Device>& device, bool is_lazy, bool requires_grad, bool is_leaf,
    bool retain_grad) {
  std::shared_ptr<MirroredTensorImpl> impl;
  if (is_lazy) {
    impl = std::make_shared<LazyMirroredTensorImpl>(shape, dtype, device, requires_grad, is_leaf,
                                                    retain_grad);
  } else {
    impl = std::make_shared<EagerMirroredTensorImpl>(shape, dtype, device, requires_grad, is_leaf,
                                                     retain_grad);
  }
  return std::make_shared<MirroredTensor>(impl);
}

bool MirroredTensor::is_cuda() const { return device()->type() == "cuda"; }

int64_t MirroredTensor::ndim() const { return shape()->NumAxes(); }

int64_t MirroredTensor::dim(int64_t index) const { return shape()->At(index); }

int64_t MirroredTensor::nelement() const { return shape()->elem_cnt(); }

std::shared_ptr<MirroredTensor> MirroredTensor::data() const {
  std::shared_ptr<MirroredTensor> t =
      MakeTensor(shape(), dtype(), device(), is_lazy(), false, is_leaf(), false);
  t->set_blob_object(blob_object());
  return t;
}

std::shared_ptr<MirroredTensor> MirroredTensor::detach() const {
  std::shared_ptr<MirroredTensor> t = std::make_shared<MirroredTensor>(impl_);
  return t;
}

Maybe<void> MirroredTensor::set_parallel_desc(const std::shared_ptr<const ParallelDesc>& parallel_desc) { 
  JUST(impl_->set_parallel_desc(parallel_desc)); 
  return Maybe<void>::Ok();
} 

std::shared_ptr<ConsistentTensor> ConsistentTensor::MakeTensor(
    const std::shared_ptr<const Shape>& shape, const std::shared_ptr<const DType>& dtype,
    const std::shared_ptr<const compatible_py::Distribute>& distribute,
    const std::shared_ptr<const ParallelDesc>& parallel_desc, bool is_lazy, bool requires_grad,
    bool is_leaf, bool retain_grad) {
  std::shared_ptr<ConsistentTensorImpl> impl;
  if (is_lazy) {
    impl = std::make_shared<LazyConsistentTensorImpl>(shape, dtype, distribute, parallel_desc,
                                                      requires_grad, is_leaf, retain_grad);
  } else {
    impl = std::make_shared<EagerConsistentTensorImpl>(shape, dtype, distribute, parallel_desc,
                                                       requires_grad, is_leaf, retain_grad);
  }
  return std::make_shared<ConsistentTensor>(impl);
}

bool ConsistentTensor::is_cuda() const {
  return parallel_desc()->device_type() == DeviceType::kGPU;
}

int64_t ConsistentTensor::dim(int64_t index) const { return shape()->At(index); }

int64_t ConsistentTensor::nelement() const { return shape()->elem_cnt(); }

int64_t ConsistentTensor::ndim() const { return shape()->NumAxes(); }

std::shared_ptr<ConsistentTensor> ConsistentTensor::data() const {
  std::shared_ptr<ConsistentTensor> t = MakeTensor(shape(), dtype(), distribute(), parallel_desc(),
                                                   is_lazy(), false, is_leaf(), false);
  t->set_blob_object(blob_object());
  return t;
}

std::shared_ptr<ConsistentTensor> ConsistentTensor::detach() const {
  std::shared_ptr<ConsistentTensor> t = std::make_shared<ConsistentTensor>(impl_);
  return t;
}

}  // namespace one

}  // namespace oneflow
