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

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class TensorWithDataFunctor {
 public:
  Maybe<Tensor> operator()(PyObject* data, const Optional<Symbol<DType>>& dtype,
                           const Optional<Symbol<Device>>& device, const bool requires_grad,
                           const bool pin_memory) const {
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
      return MakeTensorFromOtherTensor(other, dtype, device, requires_grad, pin_memory);
    } else {
      // Make tensor from python sequence or numpy array.
      return MakeLocalTensorFromData(data, dtype, device, requires_grad, pin_memory);
    }
  }
};

class ConsistentTensorWithDataFunctor {
 public:
  Maybe<Tensor> operator()(PyObject* data, const Optional<Symbol<DType>>& dtype,
                           const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple,
                           const bool requires_grad) const {
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
    bool pin_memory = false;
    if (other->is_local()) {
      pin_memory = JUST(JUST(other->AsMirroredTensor())->eager_blob_object())->pin_memory();
    }
    return MakeTensorFromOtherTensor(other, pin_memory);
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
      const bool pin_memory =
          other->is_local()
              ? JUST(JUST(other->AsMirroredTensor())->eager_blob_object())->pin_memory()
              : false;
      return MakeTensorFromOtherTensor(other, dtype, device,
                                       /*requires_grad=*/false, /*pin_memory=*/pin_memory);
    }
    // Make tensor from python sequence or numpy array.
    return MakeLocalTensorFromData(data, dtype, device, /*requires_grad=*/false,
                                   /*pin_memory=*/false);
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
    std::shared_ptr<Tensor> out(new MirroredTensor(tensor_impl));
    return out;
  }
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
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
