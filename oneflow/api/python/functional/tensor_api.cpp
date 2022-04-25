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
#include <memory>

#include "oneflow/api/python/utils/tensor_utils.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/functional/tensor_api.yaml.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/common/foreign_lock_helper.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/autograd/autograd_mode.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class TensorWithDataFunctor {
 public:
  Maybe<Tensor> operator()(PyObject* data, const Optional<Symbol<DType>>& dtype,
                           const Optional<Symbol<Device>>& device,
                           const bool& requires_grad) const {
    // NOTE(chengcheng): flow.Tensor or flow.tensor ONLY created by EagerTensor now.
    //  even if in nn.Graph build (module forward function), if you create a flow.Tensor,
    //  its a eager tensor by Run functional::Empty() in LazyMode::Grad(false)
    LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);

    if (PyTensor_Check(data)) {
      // Throw warnings like pytorch.
      auto ret = PyErr_WarnEx(
          PyExc_UserWarning,
          "To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() "
          "or sourceTensor.clone().detach().requires_grad_(True), rather than "
          "oneflow.tensor(sourceTensor).",
          1);
      if (ret != 0) { return Error::RuntimeError(); }

      const auto& other = PyTensor_Unpack(data);
      return MakeTensorFromOtherTensor(other, dtype, device, requires_grad);
    } else {
      // Make tensor from python sequence or numpy array.
      return MakeLocalTensorFromData(data, dtype, device, requires_grad);
    }
  }
};

class ConsistentTensorWithDataFunctor {
 public:
  Maybe<Tensor> operator()(PyObject* data, const Optional<Symbol<DType>>& dtype,
                           const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple,
                           const bool& requires_grad) const {
    // NOTE(chengcheng): flow.Tensor or flow.tensor ONLY created by EagerTensor now.
    LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);
    JUST(CheckDeviceIdsIsValid(placement));

    if (PyTensor_Check(data)) {
      // Throw warnings like pytorch.
      auto ret = PyErr_WarnEx(
          PyExc_UserWarning,
          "To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() "
          "or sourceTensor.clone().detach().requires_grad_(True), rather than "
          "oneflow.tensor(sourceTensor).",
          1);
      if (ret != 0) { return Error::RuntimeError(); }

      const auto& other = PyTensor_Unpack(data);
      return MakeTensorFromOtherTensor(other, dtype, placement, sbp_tuple, requires_grad);
    }
    // Make consistent tensor from python sequence or numpy array.
    return MakeConsistentTensorFromData(data, dtype, placement, sbp_tuple, requires_grad);
  }
};

class TensorEmptyCtorFunctor {
 public:
  Maybe<Tensor> operator()(const Optional<Symbol<Device>>& device) const {
    Shape shape(DimVector{0});
    return TensorWithShapeCtor(shape, device);
  }
};

class ConsistentTensorEmptyCtorFunctor {
 public:
  Maybe<Tensor> operator()(const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple) const {
    Shape shape(DimVector{0});
    JUST(CheckDeviceIdsIsValid(placement));
    return ConsistentTensorWithShapeCtor(shape, placement, sbp_tuple);
  }
};

class TensorWithOtherCtorFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& other) const {
    // NOTE(chengcheng): flow.Tensor or flow.tensor ONLY created by EagerTensor now.
    LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);
    return MakeTensorFromOtherTensor(other);
  }
};

class TensorWithDataCtorFunctor {
 public:
  Maybe<Tensor> operator()(PyObject* data, const Optional<Symbol<Device>>& device) const {
    // Treat the single long as shape.
    if (PyLong_Check(data)) {
      int64_t size = PyLong_AsLongLong(data);
      Shape shape(DimVector{size});
      return TensorWithShapeCtor(shape, device);
    }

    // NOTE(chengcheng): flow.Tensor or flow.tensor ONLY created by EagerTensor now.
    LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);

    const auto& dtype = DType::Float();
    if (PyTensor_Check(data)) {
      const auto& other = PyTensor_Unpack(data);
      return MakeTensorFromOtherTensor(other, dtype, device,
                                       /*requires_grad=*/false);
    }
    // Make tensor from python sequence or numpy array.
    return MakeLocalTensorFromData(data, dtype, device, /*requires_grad=*/false);
  }
};

class ConsistentTensorWithDataCtorFunctor {
 public:
  Maybe<Tensor> operator()(PyObject* data, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple) const {
    JUST(CheckDeviceIdsIsValid(placement));
    // Treat the single long as shape.
    if (PyLong_Check(data)) {
      int64_t size = PyLong_AsLongLong(data);
      Shape shape(DimVector{size});
      return ConsistentTensorWithShapeCtor(shape, placement, sbp_tuple);
    }

    // NOTE(chengcheng): flow.Tensor or flow.tensor ONLY created by EagerTensor now.
    LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);

    const auto& dtype = DType::Float();
    if (PyTensor_Check(data)) {
      const auto& other = PyTensor_Unpack(data);
      return MakeTensorFromOtherTensor(other, dtype, placement, sbp_tuple,
                                       /*requires_grad=*/false);
    }
    // Make consistent tensor from python sequence or numpy array.
    return MakeConsistentTensorFromData(data, dtype, placement, sbp_tuple, /*requires_grad=*/false);
  }
};

class TensorWithShapeCtorFunctor {
 public:
  Maybe<Tensor> operator()(const Shape& shape, const Optional<Symbol<Device>>& device) const {
    // NOTE(chengcheng): flow.Tensor or flow.tensor ONLY created by EagerTensor now.
    LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);
    Symbol<Device> device_;
    if (device) {
      device_ = JUST(device);
    } else {
      device_ = JUST(Device::New("cpu"));
    }
    return functional::Empty(shape, DType::Float(), device_, /*pin_memory=*/false);
  }
};

class ConsistentTensorWithShapeCtorFunctor {
 public:
  Maybe<Tensor> operator()(const Shape& shape, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple) const {
    // NOTE(chengcheng): flow.Tensor or flow.tensor ONLY created by EagerTensor now.
    LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);
    JUST(CheckDeviceIdsIsValid(placement));
    return functional::ConsistentEmpty(shape, DType::Float(), placement, sbp_tuple);
  }
};

class AssignLocalTensorFunctor {
 public:
  AssignLocalTensorFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("assign").Input("ref").Input("value").Build());
  }
  Maybe<void> operator()(const std::shared_ptr<one::Tensor>& ref,
                         const std::shared_ptr<one::Tensor>& value) const {
    // JUST(CheckInplaceValid(ref)); // align check to torch
    CHECK_OR_RETURN(ref->is_local() && value->is_local())
        << "Both ref and value must be local tensor.";
    JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_, {ref, value}));
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class LocalTensorSharedNumpyDataFunctor {
 public:
  LocalTensorSharedNumpyDataFunctor() {}
  Maybe<Tensor> operator()(PyObject* obj) const {
    if (!PyArray_Check(obj)) {
      return Error::TypeError() << "expected np.ndarray, but got " << Py_TYPE(obj)->tp_name;
    }
    auto* array = reinterpret_cast<PyArrayObject*>(obj);
    // TODO(wyg): support non-contiguous array.
    if (!PyArray_IS_C_CONTIGUOUS(array)) {
      OF_LOG_ONCE(LOG(WARNING) << "OneFlow don't support non-contiguous array now, "
                                  "and we will copy the array to a contiguous one.");
      array = PyArray_GETCONTIGUOUS(array);
    }

    // Build TensorMeta
    int32_t dim = PyArray_NDIM(array);
    const npy_intp* dims_ptr = PyArray_SHAPE(array);
    const auto shape = std::make_shared<Shape>(DimVector(dims_ptr, dims_ptr + dim));
    DataType data_type = JUST(numpy::GetOFDataTypeFromNpArray(array));
    Symbol<Device> device = JUST(Device::New("cpu"));
    const npy_intp* stride_ptr = PyArray_STRIDES(array);
    // stride
    auto strides_vec = DimVector(stride_ptr, stride_ptr + dim);
    auto element_size_in_bytes = PyArray_ITEMSIZE(array);
    // NumPy strides use bytes. OneFlow strides use element counts.
    for (auto& stride : strides_vec) {
      if (stride % element_size_in_bytes != 0) {
        return Error::RuntimeError() << "given numpy array strides not a multiple of the element "
                                        "byte size. Copy the numpy array to reallocate the memory.";
      }
      stride /= element_size_in_bytes;
    }
    const auto strides = std::make_shared<Stride>(strides_vec);
    auto tensor_meta = std::make_shared<MirroredTensorMeta>(shape, data_type, device, strides, 0);

    // Build TensorBuffer
    const auto& Free = [obj](char* dptr) {
      CHECK_JUST(Global<ForeignLockHelper>::Get()->WithScopedAcquire([&]() -> Maybe<void> {
        Py_DECREF(obj);
        return Maybe<void>::Ok();
      }));
    };
    Py_INCREF(obj);  // make TensorBuffer hold ndarray
    void* data_ptr = PyArray_DATA(array);
    auto array_size_in_bytes = PyArray_NBYTES(array);
    auto tensor_data = std::make_shared<vm::TensorStorage>();
    tensor_data->set_blob_dptr(
        std::unique_ptr<char, std::function<void(char*)>>(static_cast<char*>(data_ptr), Free),
        array_size_in_bytes);

    // Build TensorStorage: decrease ndarray reference count before releasing
    auto tensor_storage = std::make_shared<TensorStorage>(tensor_data);

    // Build Tensor
    auto tensor_impl = std::make_shared<EagerMirroredTensorImpl>(tensor_meta, tensor_storage,
                                                                 /*requires_grad=*/false,
                                                                 /*ls_leaf=*/true);

    // Init blob
    JUST(tensor_impl->InitEagerBlobObject(NewLocalDepObject(), /*pin_memory=*/false));
    const auto& stream = GetDefaultStreamByDevice(device);
    JUST(tensor_impl->eager_blob_object())->set_last_used_stream(stream);
    JUST(JUST(tensor_impl->eager_blob_object())->TryInitBlob());
    JUST(tensor_impl->eager_blob_object())->mut_blob()->reset_dptr(static_cast<char*>(data_ptr));
    std::shared_ptr<Tensor> out(new MirroredTensor(tensor_impl));
    return out;
  }
};

class PinMemoryFunctor {
 public:
  PinMemoryFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("slice_update").Input("x").Input("update").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input) const {
    // TODO:(zhaoluyang) support consistent tensor.pin_memory()
    CHECK_OR_RETURN(input->is_local())
        << Error::RuntimeError() << "Tensor.pin_memory() only support local tensor for now!";
    // if tensor already pinned, then just return
    if (JUST(JUST(input->AsMirroredTensor())->eager_blob_object())->pin_memory()) { return input; }
    auto shape = input->shape();
    auto device = JUST(input->device());
    const bool requires_grad = input->requires_grad();
    CHECK_EQ_OR_RETURN(device->enum_type(), DeviceType::kCPU)
        << Error::RuntimeError() << "cannot pin tensor with device: " << device->ToString()
        << ", only dense CPU tensors can be pinned.";

    auto empty = JUST(functional::Empty(*shape.get(), input->dtype(), device, /*pin_memory=*/true));
    // TODO: remove this requires_grad
    JUST(empty->set_requires_grad(requires_grad));
    const int32_t ndim = input->ndim();
    if(ndim == 0){
      //for 0-dim case, use assign, other use slice_update
      JUST(functional::AssignLocalTensor(empty, input));
      // if requires_grad, set backward function-node 'copy_backward'
      if(autograd::GradMode::is_enabled() && requires_grad){
        auto backward_fn = std::make_shared<BackwardFunction>();
        backward_fn->body = [=](const TensorTuple& out_grads, TensorTuple* in_grads,
                                bool create_graph) -> Maybe<void> {
          autograd::AutoGradMode mode(create_graph);
          CHECK_EQ_OR_RETURN(out_grads.size(), 1);  // NOLINT(maybe-need-error-msg)
          in_grads->resize(1);
          const Symbol<Device>& device = JUST(out_grads[0]->device());
          (*in_grads)[0] = JUST(functional::Copy(out_grads[0], device->type(), device->device_id()););
          return Maybe<void>::Ok();
        };
        backward_fn->status = []() { return true; };
        TensorTuple outputs{empty};
        JUST(GetThreadLocalAutogradEngine()->AddNode("copy_backward", backward_fn, {input},
                                                    &outputs));
      }
      return empty;
    } else {
      MutableAttrMap attrs;
      std::vector<int64_t> starts(ndim, 0);
      std::vector<int64_t> stops(ndim);
      std::vector<int64_t> steps(ndim, 1);
      for (int i = 0; i < ndim; ++i) { stops[i] = input->shape()->At(i); }
      JUST(attrs.SetAttr<std::vector<int64_t>>("start", starts));
      JUST(attrs.SetAttr<std::vector<int64_t>>("stop", stops));
      JUST(attrs.SetAttr<std::vector<int64_t>>("step", steps));
      JUST(empty->set_requires_grad(requires_grad));
      auto outputs = TensorTuple{empty};
      JUST(OpInterpUtil::Dispatch(*op_, TensorTuple{empty, input}, &outputs, attrs));
      return outputs[0];
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::TensorWithDataFunctor>("TensorWithData");
  m.add_functor<impl::ConsistentTensorWithDataFunctor>("ConsistentTensorWithData");
  m.add_functor<impl::TensorEmptyCtorFunctor>("TensorEmptyCtor");
  m.add_functor<impl::ConsistentTensorEmptyCtorFunctor>("ConsistentTensorEmptyCtor");
  m.add_functor<impl::TensorWithOtherCtorFunctor>("TensorWithOtherCtor");
  m.add_functor<impl::TensorWithDataCtorFunctor>("TensorWithDataCtor");
  m.add_functor<impl::ConsistentTensorWithDataCtorFunctor>("ConsistentTensorWithDataCtor");
  m.add_functor<impl::TensorWithShapeCtorFunctor>("TensorWithShapeCtor");
  m.add_functor<impl::ConsistentTensorWithShapeCtorFunctor>("ConsistentTensorWithShapeCtor");
  m.add_functor<impl::AssignLocalTensorFunctor>("AssignLocalTensor");
  m.add_functor<impl::LocalTensorSharedNumpyDataFunctor>("LocalTensorSharedNumpyData");
  m.add_functor<impl::PinMemoryFunctor>("PinMemory");
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
