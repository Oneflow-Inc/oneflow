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
#include "oneflow/api/python/dlpack/converter.h"
#include "oneflow/api/python/framework/size.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/functional/tensor_api.yaml.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/eager/tensor_storage.h"
#include "oneflow/core/framework/mutable_attr_map.h"
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
    if (GlobalMode::is_enabled()) {
      auto global_mode_gurad = GlobalMode::Guard(false);
      return JUST(
          functional::GlobalTensorWithData(data, dtype, GetGlobalParallelDescFromDevice(device),
                                           *JUST(GetSbpList(GlobalMode::nd_sbp())), requires_grad));
    }

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

class TensorEmptyGenericCtorFunctor {
 public:
  Maybe<Tensor> operator()(const Symbol<DType>& dtype,
                           const Optional<Symbol<Device>>& device) const {
    Shape shape(DimVector{0});
    return TensorWithShapeGenericCtor(shape, dtype, device);
  }
};

class GlobalTensorEmptyGenericCtorFunctor {
 public:
  Maybe<Tensor> operator()(const Symbol<DType>& dtype, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple) const {
    Shape shape(DimVector{0});
    JUST(CheckDeviceIdsIsValid(placement));
    return GlobalTensorWithShapeGenericCtor(shape, dtype, placement, sbp_tuple);
  }
};

class TensorWithOtherGenericCtorFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& other,
                           const Optional<Symbol<DType>>& dtype) const {
    // NOTE(chengcheng): flow.Tensor or flow.tensor ONLY created by EagerTensor now.
    LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);
    bool is_pinned = false;
    if (other->is_local()) { is_pinned = JUST(CHECK_JUST(other->AsLocalTensor())->is_pinned()); }
    return To(JUST(MakeTensorFromOtherTensor(other, is_pinned)), dtype, false);
  }
};

class TensorWithDataGenericCtorFunctor {
 public:
  Maybe<Tensor> operator()(PyObject* data, const Symbol<DType>& dtype,
                           const Optional<Symbol<Device>>& device) const {
    // Treat the single long as shape.
    if (PyLong_Check(data)) {
      int64_t size = PyLong_AsLongLong(data);
      Shape shape(DimVector{size});
      return TensorWithShapeGenericCtor(shape, dtype, device);
    }
    if (TensorSize_Check(data)) {
      return TensorWithShapeGenericCtor(TensorSize_AsShape(data), dtype, device);
    }

    // NOTE(chengcheng): flow.Tensor or flow.tensor ONLY created by EagerTensor now.
    LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);

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

class GlobalTensorWithDataGenericCtorFunctor {
 public:
  Maybe<Tensor> operator()(PyObject* data, const Symbol<DType>& dtype,
                           const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple) const {
    JUST(CheckDeviceIdsIsValid(placement));
    // Treat the single long as shape.
    if (PyLong_Check(data)) {
      int64_t size = PyLong_AsLongLong(data);
      Shape shape(DimVector{size});
      return GlobalTensorWithShapeGenericCtor(shape, dtype, placement, sbp_tuple);
    }
    if (TensorSize_Check(data)) {
      return GlobalTensorWithShapeGenericCtor(TensorSize_AsShape(data), dtype, placement,
                                              sbp_tuple);
    }

    // NOTE(chengcheng): flow.Tensor or flow.tensor ONLY created by EagerTensor now.
    LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);

    if (PyTensor_Check(data)) {
      const auto& other = PyTensor_Unpack(data);
      return MakeTensorFromOtherTensor(other, dtype, placement, sbp_tuple,
                                       /*requires_grad=*/false);
    }
    // Make global tensor from python sequence or numpy array.
    return MakeGlobalTensorFromData(data, dtype, placement, sbp_tuple, /*requires_grad=*/false);
  }
};

class TensorWithShapeGenericCtorFunctor {
 public:
  Maybe<Tensor> operator()(const Shape& shape, const Symbol<DType>& dtype,
                           const Optional<Symbol<Device>>& device) const {
    // NOTE(chengcheng): flow.Tensor or flow.tensor ONLY created by EagerTensor now.
    LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);
    Symbol<Device> device_;
    if (device) {
      device_ = JUST(device);
    } else {
      device_ = JUST(Device::New("cpu"));
    }
    return functional::Empty(shape, dtype, device_, /*requires_grad=*/false, /*pin_memory=*/false);
  }
};

class GlobalTensorWithShapeGenericCtorFunctor {
 public:
  Maybe<Tensor> operator()(const Shape& shape, const Symbol<DType>& dtype,
                           const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple) const {
    // NOTE(chengcheng): flow.Tensor or flow.tensor ONLY created by EagerTensor now.
    LazyMode::Guard lazy_mode_disabled_guard(/*is_enabled*/ false);
    JUST(CheckDeviceIdsIsValid(placement));
    return functional::GlobalEmpty(shape, dtype, placement, sbp_tuple);
  }
};

class AssignLocalTensorFunctor {
 public:
  AssignLocalTensorFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("copy").Input("in").Output("out").Build());
  }
  Maybe<void> operator()(const std::shared_ptr<one::Tensor>& y,
                         const std::shared_ptr<one::Tensor>& x) const {
    // JUST(CheckInplaceValid(y)); // align check to torch
    CHECK_OR_RETURN(y->is_local() && x->is_local()) << "Both x and y must be local tensor.";
    std::shared_ptr<one::Tensor> src = x;
    if (y->dtype() != src->dtype()) { src = JUST(To(src, y->dtype(), false)); }

    auto device = JUST(y->device());
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("device", "pin_memory");
    attrs.SetAllAttrs(device, false);
    TensorTuple outputs{y};
    return OpInterpUtil::Dispatch(*op_, {x}, &outputs, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

static std::vector<int64_t> get_shape_or_stride_from_numpy(size_t ndim, npy_intp* values) {
  auto result = std::vector<int64_t>(ndim);
  for (size_t i = 0; i < ndim; ++i) { result[i] = static_cast<int64_t>(values[i]); }
  return result;
}

class LocalTensorSharedDlPackDataFunctor {
 public:
  LocalTensorSharedDlPackDataFunctor() {}
  Maybe<Tensor> operator()(PyObject* obj) const {
    DLManagedTensor* dlMTensor = (DLManagedTensor*)PyCapsule_GetPointer(obj, "dltensor");
    CHECK_NOTNULL_OR_RETURN(dlMTensor)
        << "from_dlpack received an invalid capsule. "
           "Note that DLTensor capsules can be consumed only once, "
           "so you might have already constructed a tensor from it once.";

    // `tensor` steals the ownership of the underlying storage. It also passes a
    // destructor function that will be called when the underlying storage goes
    // out of scope. When the destructor is called, the dlMTensor is destructed
    // too.
    auto tensor = fromDLPack(dlMTensor);

    // Make sure this capsule will never be used again.
    PyCapsule_SetName(obj, "used_dltensor");

    return tensor;
  }
};

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
    const auto shape = Shape(DimVector(sizes.begin(), sizes.end()));
    const auto stride = Stride(strides.begin(), strides.end());
    DataType data_type = JUST(numpy::GetOFDataTypeFromNpArray(array));
    Symbol<Device> device = JUST(Device::New("cpu"));

    auto tensor_meta = SymbolOf(LocalTensorMeta(shape, stride, data_type, device));

    // Build TensorBuffer
    const auto& Free = [array](char* dptr) {
      CHECK_JUST(Singleton<ForeignLockHelper>::Get()->WithScopedAcquire([&]() -> Maybe<void> {
        Py_DECREF(array);
        return Maybe<void>::Ok();
      }));
    };

    const auto array_size_in_bytes = PyArray_NBYTES(array);
    auto tensor_data = std::make_shared<vm::TensorStorage>(false);
    tensor_data->set_blob_dptr(
        std::unique_ptr<char, std::function<void(char*)>>(static_cast<char*>(data_ptr), Free),
        array_size_in_bytes);

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
  m.add_functor<impl::TensorEmptyGenericCtorFunctor>("TensorEmptyGenericCtor");
  m.add_functor<impl::GlobalTensorEmptyGenericCtorFunctor>("GlobalTensorEmptyGenericCtor");
  m.add_functor<impl::TensorWithOtherGenericCtorFunctor>("TensorWithOtherGenericCtor");
  m.add_functor<impl::TensorWithDataGenericCtorFunctor>("TensorWithDataGenericCtor");
  m.add_functor<impl::GlobalTensorWithDataGenericCtorFunctor>("GlobalTensorWithDataGenericCtor");
  m.add_functor<impl::TensorWithShapeGenericCtorFunctor>("TensorWithShapeGenericCtor");
  m.add_functor<impl::GlobalTensorWithShapeGenericCtorFunctor>("GlobalTensorWithShapeGenericCtor");
  m.add_functor<impl::AssignLocalTensorFunctor>("AssignLocalTensor");
  m.add_functor<impl::LocalTensorSharedNumpyDataFunctor>("LocalTensorSharedNumpyData");
  m.add_functor("TensorEmptyCtor", [](const Optional<Symbol<Device>>& device) -> Maybe<Tensor> {
    return TensorEmptyGenericCtor(GetDefaultDType(), device);
  });
  m.add_functor("GlobalTensorEmptyCtor",
                [](const Symbol<ParallelDesc>& placement,
                   const std::vector<Symbol<SbpParallel>>& sbp_tuple) -> Maybe<Tensor> {
                  return GlobalTensorEmptyGenericCtor(GetDefaultDType(), placement, sbp_tuple);
                });
  m.add_functor("TensorWithOtherCtor", [](const std::shared_ptr<Tensor>& other) -> Maybe<Tensor> {
    return TensorWithOtherGenericCtor(other, NullOpt);
  });
  m.add_functor("TensorWithDataCtor",
                [](PyObject* data, const Optional<Symbol<Device>>& device) -> Maybe<Tensor> {
                  return TensorWithDataGenericCtor(data, GetDefaultDType(), device);
                });
  m.add_functor("GlobalTensorWithDataCtor",
                [](PyObject* data, const Symbol<ParallelDesc>& placement,
                   const std::vector<Symbol<SbpParallel>>& sbp_tuple) -> Maybe<Tensor> {
                  return GlobalTensorWithDataGenericCtor(data, GetDefaultDType(), placement,
                                                         sbp_tuple);
                });
  m.add_functor("TensorWithShapeCtor",
                [](const Shape& shape, const Optional<Symbol<Device>>& device) -> Maybe<Tensor> {
                  return TensorWithShapeGenericCtor(shape, GetDefaultDType(), device);
                });
  m.add_functor("GlobalTensorWithShapeCtor",
                [](const Shape& shape, const Symbol<ParallelDesc>& placement,
                   const std::vector<Symbol<SbpParallel>>& sbp_tuple) -> Maybe<Tensor> {
                  return GlobalTensorWithShapeGenericCtor(shape, GetDefaultDType(), placement,
                                                          sbp_tuple);
                });
  m.add_functor<impl::LocalTensorSharedDlPackDataFunctor>("LocalTensorSharedDlPackData");
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
