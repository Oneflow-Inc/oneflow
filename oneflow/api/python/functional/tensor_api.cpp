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
#include "oneflow/api/python/framework/size.h"
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

class GlobalTensorWithDataFunctor {
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
    // Make global tensor from python sequence or numpy array.
    return MakeGlobalTensorFromData(data, dtype, placement, sbp_tuple, requires_grad);
  }
};

class TensorEmptyCtorFunctor {
 public:
  Maybe<Tensor> operator()(const Optional<Symbol<Device>>& device) const {
    Shape shape(DimVector{0});
    return TensorWithShapeCtor(shape, device);
  }
};

class GlobalTensorEmptyCtorFunctor {
 public:
  Maybe<Tensor> operator()(const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple) const {
    Shape shape(DimVector{0});
    JUST(CheckDeviceIdsIsValid(placement));
    return GlobalTensorWithShapeCtor(shape, placement, sbp_tuple);
  }
};

class TensorWithOtherCtorFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& other) const {
    // NOTE(chengcheng): flow.Tensor or flow.tensor ONLY created by EagerTensor now.
    LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);
    bool is_pinned = false;
    if (other->is_local()) { is_pinned = JUST(CHECK_JUST(other->AsLocalTensor())->is_pinned()); }
    return MakeTensorFromOtherTensor(other, is_pinned);
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
    if (TensorSize_Check(data)) { return TensorWithShapeCtor(TensorSize_AsShape(data), device); }

    // NOTE(chengcheng): flow.Tensor or flow.tensor ONLY created by EagerTensor now.
    LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);

    const auto& dtype = DType::Float();
    if (PyTensor_Check(data)) {
      const auto& other = PyTensor_Unpack(data);
      const bool pin_memory =
          other->is_local() ? JUST(JUST(other->AsLocalTensor())->is_pinned()) : false;
      return MakeTensorFromOtherTensor(other, dtype, device,
                                       /*requires_grad=*/false, /*pin_memory=*/pin_memory);
    }
    // Make tensor from python sequence or numpy array.
    return MakeLocalTensorFromData(data, dtype, device, /*requires_grad=*/false,
                                   /*pin_memory=*/false);
  }
};

class GlobalTensorWithDataCtorFunctor {
 public:
  Maybe<Tensor> operator()(PyObject* data, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple) const {
    JUST(CheckDeviceIdsIsValid(placement));
    // Treat the single long as shape.
    if (PyLong_Check(data)) {
      int64_t size = PyLong_AsLongLong(data);
      Shape shape(DimVector{size});
      return GlobalTensorWithShapeCtor(shape, placement, sbp_tuple);
    }
    if (TensorSize_Check(data)) {
      return GlobalTensorWithShapeCtor(TensorSize_AsShape(data), placement, sbp_tuple);
    }

    // NOTE(chengcheng): flow.Tensor or flow.tensor ONLY created by EagerTensor now.
    LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);

    const auto& dtype = DType::Float();
    if (PyTensor_Check(data)) {
      const auto& other = PyTensor_Unpack(data);
      return MakeTensorFromOtherTensor(other, dtype, placement, sbp_tuple,
                                       /*requires_grad=*/false);
    }
    // Make global tensor from python sequence or numpy array.
    return MakeGlobalTensorFromData(data, dtype, placement, sbp_tuple, /*requires_grad=*/false);
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

class GlobalTensorWithShapeCtorFunctor {
 public:
  Maybe<Tensor> operator()(const Shape& shape, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple) const {
    // NOTE(chengcheng): flow.Tensor or flow.tensor ONLY created by EagerTensor now.
    LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);
    JUST(CheckDeviceIdsIsValid(placement));
    return functional::GlobalEmpty(shape, DType::Float(), placement, sbp_tuple);
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

static std::vector<int64_t> get_shape_or_stride_from_numpy(size_t ndim, npy_intp* values) {
  auto result = std::vector<int64_t>(ndim);
  for (size_t i = 0; i < ndim; ++i) { result[i] = static_cast<int64_t>(values[i]); }
  return result;
}

class LocalTensorSharedNumpyDataFunctor {
 public:
  LocalTensorSharedNumpyDataFunctor() {}
  Maybe<Tensor> operator()(PyObject* obj) const {
    if (!PyArray_Check(obj)) {
      return Error::TypeError() << "expected np.ndarray, but got " << Py_TYPE(obj)->tp_name;
    }
    auto* array = reinterpret_cast<PyArrayObject*>(obj);
    const size_t ndim = PyArray_NDIM(array);
    std::vector<int64_t> sizes = get_shape_or_stride_from_numpy(ndim, PyArray_DIMS(array));
    std::vector<int64_t> strides = get_shape_or_stride_from_numpy(ndim, PyArray_STRIDES(array));
    // NumPy strides use bytes. OneFlow strides use element counts.
    // These checks are consistent with pytorch(v1.10.0):
    // https://github.com/pytorch/pytorch/blob/v1.10.0/torch/csrc/utils/tensor_numpy.cpp#L171
    const auto element_size_in_bytes = PyArray_ITEMSIZE(array);
    for (auto& stride : strides) {
      if (stride % element_size_in_bytes != 0) {
        return Error::InvalidValueError()
               << "given numpy array strides not a multiple of the element byte size. "
               << "Copy the numpy array to reallocate the memory.";
      }
      stride /= element_size_in_bytes;
    }
    for (size_t i = 0; i < ndim; ++i) {
      if (strides[i] < 0) {
        return Error::InvalidValueError()
               << "At least one stride in the given numpy array is negative, "
               << "and tensors with negative strides are not currently supported. "
               << "(You can probably work around this by making a copy of your array "
               << " with array.copy().) ";
      }
    }
    void* data_ptr = PyArray_DATA(array);
    if (!PyArray_EquivByteorders(PyArray_DESCR(array)->byteorder, NPY_NATIVE)) {
      return Error::InvalidValueError()
             << "given numpy array has byte order different from the native byte order. "
             << "Conversion between byte orders is currently not supported.";
    }
    Py_INCREF(obj);

    // Build TensorMeta
    const auto shape = std::make_shared<Shape>(DimVector(sizes.begin(), sizes.end()));
    const auto stride = std::make_shared<Stride>(strides.begin(), strides.end());
    DataType data_type = JUST(numpy::GetOFDataTypeFromNpArray(array));
    Symbol<Device> device = JUST(Device::New("cpu"));

    auto tensor_meta = SymbolOf(LocalTensorMeta(shape, stride, data_type, device, 0));

    // Build TensorBuffer
    const auto& Free = [array](char* dptr) {
      CHECK_JUST(Singleton<ForeignLockHelper>::Get()->WithScopedAcquire([&]() -> Maybe<void> {
        Py_DECREF(array);
        return Maybe<void>::Ok();
      }));
    };

    const auto array_size_in_bytes = PyArray_NBYTES(array);
    auto tensor_data = std::make_shared<vm::TensorStorage>();
    tensor_data->set_blob_dptr(
        std::unique_ptr<char, std::function<void(char*)>>(static_cast<char*>(data_ptr), Free),
        array_size_in_bytes, /*is_allocated_in_vm*/ false);

    // Build TensorStorage: decrease ndarray reference count before releasing
    auto tensor_storage = std::make_shared<TensorStorage>(tensor_data);

    // Build Tensor
    auto tensor_impl = std::make_shared<EagerLocalTensorImpl>(tensor_storage,
                                                              /*requires_grad=*/false,
                                                              /*ls_leaf=*/true);

    // Init blob
    JUST(tensor_impl->InitEagerBlobObject(tensor_meta, NewLocalDepObject()));
    const auto& stream = JUST(GetDefaultStreamByDevice(device));
    const auto& eager_blob_object = JUST(tensor_impl->eager_blob_object());
    JUST(eager_blob_object->init_producer_stream(stream));
    eager_blob_object->set_last_used_stream(stream);
    std::shared_ptr<Tensor> out(new LocalTensor(tensor_impl));
    return out;
  }
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::TensorWithDataFunctor>("TensorWithData");
  m.add_functor<impl::GlobalTensorWithDataFunctor>("GlobalTensorWithData");
  m.add_functor<impl::TensorEmptyCtorFunctor>("TensorEmptyCtor");
  m.add_functor<impl::GlobalTensorEmptyCtorFunctor>("GlobalTensorEmptyCtor");
  m.add_functor<impl::TensorWithOtherCtorFunctor>("TensorWithOtherCtor");
  m.add_functor<impl::TensorWithDataCtorFunctor>("TensorWithDataCtor");
  m.add_functor<impl::GlobalTensorWithDataCtorFunctor>("GlobalTensorWithDataCtor");
  m.add_functor<impl::TensorWithShapeCtorFunctor>("TensorWithShapeCtor");
  m.add_functor<impl::GlobalTensorWithShapeCtorFunctor>("GlobalTensorWithShapeCtor");
  m.add_functor<impl::AssignLocalTensorFunctor>("AssignLocalTensor");
  m.add_functor<impl::LocalTensorSharedNumpyDataFunctor>("LocalTensorSharedNumpyData");
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
