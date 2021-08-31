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
#include <Python.h>

#include "oneflow/api/python/utils/tensor_utils.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/functional/tensor_api.yaml.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/scalar.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/framework/nd_sbp.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class TensorWithDataFunctor {
 public:
  Maybe<Tensor> operator()(PyObject* data, const Optional<Symbol<DType>>& dtype,
                           const Optional<Symbol<Device>>& device,
                           const Optional<Symbol<ParallelDesc>>& placement,
                           const Optional<std::vector<Symbol<cfg::SbpParallel>>>& sbp_tuple,
                           const bool& requires_grad) const {
    // NOTE(chengcheng): flow.Tensor or flow.tensor ONLY created by EagerTensor now.
    //  even if in nn.Graph build (module forward function), if you create a flow.Tensor,
    //  its a eager tensor by Run functional::Empty() in LazyMode::Grad(false)
    LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);

    if (PyTensorCheck(data)) {
      // Throw warnings like pytorch.
      auto ret = PyErr_WarnEx(
          PyExc_UserWarning,
          "To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() "
          "or sourceTensor.clone().detach().requires_grad_(True), rather than "
          "oneflow.tensor(sourceTensor).",
          1);
      if (ret != 0) { return Error::RuntimeError(); }

      const auto& other = JUST(PyUnpackTensor(data));
      return MakeTensorFromOtherTensor(other, dtype, device, placement, sbp_tuple, requires_grad);
    }

    // TODO(): Construct consistent tensor from sequence or numpy ndarray.
    if (placement.has_value() || sbp_tuple.has_value()) {
      return Error::RuntimeError()
             << "Can not construct consistent tensor from sequence or numpy array currently.";
    }
    // Make tensor from python sequence or numpy array.
    return MakeLocalTensorFromData(data, dtype, device, requires_grad);
  }
};

class TensorEmptyCtorFunctor {
 public:
  Maybe<Tensor> operator()(const Optional<Symbol<Device>>& device,
                           const Optional<Symbol<ParallelDesc>>& placement,
                           const Optional<std::vector<Symbol<cfg::SbpParallel>>>& sbp_tuple) const {
    Shape shape(DimVector{0});
    return TensorWithShapeCtor(shape, device, placement, sbp_tuple);
  }
};

class TensorWithOtherCtor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& other) const {
    // NOTE(chengcheng): flow.Tensor or flow.tensor ONLY created by EagerTensor now.
    //  even if in nn.Graph build (module forward function), if you create a flow.Tensor,
    //  its a eager tensor by Run functional::Empty() in LazyMode::Grad(false)
    LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);
    return MakeTensorFromOtherTensor(other);
  }
};

class TensorWithDataCtorFunctor {
 public:
  Maybe<Tensor> operator()(PyObject* data, const Optional<Symbol<Device>>& device,
                           const Optional<Symbol<ParallelDesc>>& placement,
                           const Optional<std::vector<Symbol<cfg::SbpParallel>>>& sbp_tuple) const {
    // Treat the single long as shape.
    if (PyLong_Check(data)) {
      int64_t size = PyLong_AsLongLong(data);
      Shape shape(DimVector{size});
      return TensorWithShapeCtor(shape, device, placement, sbp_tuple);
    }
    return TensorWithData(data, Optional<Symbol<DType>>(), device, placement, sbp_tuple, false);
  }
};

class TensorWithShapeCtorFunctor {
 public:
  Maybe<Tensor> operator()(const Shape& shape, const Optional<Symbol<Device>>& device,
                           const Optional<Symbol<ParallelDesc>>& placement,
                           const Optional<std::vector<Symbol<cfg::SbpParallel>>>& sbp_tuple) const {
    // NOTE(chengcheng): flow.Tensor or flow.tensor ONLY created by EagerTensor now.
    //  even if in nn.Graph build (module forward function), if you create a flow.Tensor,
    //  its a eager tensor by Run functional::Empty() in LazyMode::Grad(false)
    LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);
    if (placement) {
      if (!sbp_tuple) {
        return Error::RuntimeError()
               << "Sbp tuple is expected while constructing consistent tensor.";
      }
      return functional::ConsistentEmpty(shape, DType::Float(), JUST(placement.value()),
                                         *JUST(sbp_tuple.value()));
    } else {
      Symbol<Device> device_;
      if (device) {
        device_ = JUST(device.value());
      } else {
        device_ = JUST(Device::New("cpu"));
      }
      return functional::Empty(shape, DType::Float(), device_);
    }
  }
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::TensorWithDataFunctor>("TensorWithData");
  m.add_functor<impl::TensorEmptyCtorFunctor>("TensorEmptyCtor");
  m.add_functor<impl::TensorWithDataCtorFunctor>("TensorWithDataCtor");
  m.add_functor<impl::TensorWithShapeCtorFunctor>("TensorWithShapeCtor");
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
