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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "oneflow/core/common/throw.h"
#include "oneflow/api/python/framework/size.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/ofblob/ofblob.e.h"
#include "oneflow/api/python/utils/tensor_utils.h"
#include "oneflow/api/python/functional/tensor_api.yaml.pybind.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/framework/tensor_methods.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stride.h"
#include "oneflow/core/framework/py_distribute.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/placement_utils.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/common/decorator.h"

namespace py = pybind11;

namespace oneflow {

namespace one {

namespace {

const Symbol<DType>* GetTensorDType(const Tensor& tensor) {
  return &CHECK_JUST(DType::Get(tensor.dtype()->data_type()));
}

py::array ApiEagerTensorToNumpy(const py::handle& py_tensor) {
  const std::shared_ptr<Tensor> tensor = py::cast<const std::shared_ptr<Tensor>>(py_tensor);
  DataType data_type = tensor->dtype()->data_type();
  switch (data_type) {
#define SWITCH_EAGER_TENSOR_TO_NUMPY(cpp_type, of_type) \
  case of_type: return EagerTensorToNumpy<cpp_type>(py_tensor).GetOrThrow();
    OF_PP_FOR_EACH_TUPLE(SWITCH_EAGER_TENSOR_TO_NUMPY, POD_DATA_TYPE_SEQ)
    case DataType::kFloat16: return EagerTensorToNumpy<float16>(py_tensor).GetOrThrow();
    default:
      return Maybe<py::array>(Error::UnimplementedError() << "Invalid datatype").GetOrThrow();
  }
}

void ApiEagerMirroredTensorZeros(const std::shared_ptr<Tensor>& tensor) {
  return EagerMirroredTensorZeros(tensor).GetOrThrow();
}

template<typename T>
void ApiCopyMirroredTensorToNumpy(const std::shared_ptr<Tensor>& tensor, py::array_t<T> array) {
  CopyBetweenMirroredTensorAndNumpy<T>(tensor, array.ptr(), BlobNumpyCopyUtil<T>::To, "const",
                                       /*block_host_until_done=*/true)
      .GetOrThrow();
}

template<typename T>
void ApiCopyMirroredTensorFromNumpy(const std::shared_ptr<Tensor>& tensor, py::array_t<T> array) {
  // When asynchronously copying array data to tensor, we need to back up the
  // array at the same time.
  // Only NPY_CORDER is supported, and it makes sure that the array is C-style contiguous.
  auto* copied_array = PyArray_NewCopy((PyArrayObject*)array.ptr(), NPY_CORDER);
  CopyBetweenMirroredTensorAndNumpy<T>(tensor, copied_array, BlobNumpyCopyUtil<T>::From, "mut",
                                       /*block_host_until_done=*/false)
      .GetOrThrow();

  Py_DECREF(copied_array);
}

const std::string& ApiGetCopyMirroredTensorToNumpyFuncName(const Tensor& tensor) {
  return *GetCopyMirroredTensorToNumpyFuncName(tensor.dtype()->data_type()).GetPtrOrThrow();
}

const std::string& ApiGetCopyMirroredTensorFromNumpyFuncName(const Tensor& tensor) {
  return *GetCopyMirroredTensorFromNumpyFuncName(tensor.dtype()->data_type()).GetPtrOrThrow();
}

Symbol<Device> TensorGetDevice(const Tensor& tensor) { return tensor.device().GetOrThrow(); }

Symbol<ParallelDesc> TensorGetParallelDesc(const Tensor& tensor) {
  return tensor.parallel_desc().GetOrThrow();
}

std::tuple<std::vector<Shape>, std::vector<Symbol<DType>>> GetTensorBufferShapesAndDTypes(
    const std::shared_ptr<Tensor>& tensor) {
  return MaybeGetTensorBufferShapesAndDTypes(tensor).GetOrThrow();
}

void ApiRegisterTensorHook(const std::shared_ptr<Tensor>& self, const AutogradMeta::Hook& hook) {
  return RegisterTensorHook(self, hook).GetOrThrow();
}

void ApiRegisterTensorPostGradAccumulationHook(const std::shared_ptr<Tensor>& self,
                                               const AutogradMeta::Hook& hook) {
  return RegisterTensorPostGradAccumulationHook(self, hook).GetOrThrow();
}

bool ApiIsContiguous(const std::shared_ptr<Tensor>& tensor) {
  return IsContiguous(tensor).GetOrThrow();
}

py::tuple ApiTensorGetPyTupleOfSbp(const Tensor& tensor) {
  return *TensorGetPyTupleOfSbp(tensor).GetPtrOrThrow();
}

std::shared_ptr<Tensor> ApiNewTensor(py::args args, py::kwargs kwargs) {
  return py::cast<std::shared_ptr<Tensor>>(functional::_legacy_tensor_ctor(args, kwargs));
}

void ApiSetRequiresGrad(Tensor& tensor, bool requires_grad) {
  if (tensor.is_leaf()) {
    tensor.set_requires_grad(requires_grad).GetOrThrow();
  } else {
    throw std::runtime_error("You can only change requires_grad flags of leaf tensors.");
  }
}

std::shared_ptr<Parameter> ApiNewParameter(const std::shared_ptr<Tensor>& data,
                                           bool requires_grad) {
  return std::make_shared<Parameter>(data, requires_grad);
}

void ApiRegisterStorageDeleteHook(const std::shared_ptr<Tensor>& tensor,
                                  const std::function<void()>& hook) {
  CHECK_JUST(tensor->RegisterStorageDeleteHook(hook));
}

}  // namespace

using namespace pybind11::literals;

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
      .def(py::init(&ApiNewTensor))
      // Properties of pytorch
      .def_property_readonly("ndim", &Tensor::ndim)
      .def_property_readonly("shape", &Tensor::shape)
      .def_property_readonly("dtype", &GetTensorDType)
      .def_property_readonly("is_cuda", &Tensor::is_cuda)
      .def_property(
          "grad",
          [](const Tensor& t) -> std::shared_ptr<Tensor> { return t.acc_grad().GetPtrOrThrow(); },
          [](Tensor& t, const std::shared_ptr<Tensor>& grad) {
            if (t.is_leaf()) {
              if (grad != nullptr) {
                t.set_acc_grad(grad->detach().GetPtrOrThrow()).GetOrThrow();
              } else {
                t.set_acc_grad(nullptr).GetOrThrow();
              }
            } else {
              throw std::runtime_error("You can only change gradient of leaf tensors.");
            }
          })
      .def_property(
          "_is_grad_acc_inplace",
          [](const Tensor& t) -> bool { return t.autograd_meta()->is_grad_acc_inplace(); },
          [](Tensor& t, bool is_inplace) {
            t.mut_autograd_meta()->set_is_grad_acc_inplace(is_inplace);
          })
      .def_property(
          "data", [](Tensor& t) { return t.data().GetPtrOrThrow(); },
          [](Tensor& t, const std::shared_ptr<Tensor>& other) { t.set_data(other).GetOrThrow(); })
      .def("storage_offset", [](const Tensor& t) { return t.storage_offset().GetOrThrow(); })
      .def("stride",
           [](const Tensor& t) {
             const auto& stride = t.stride().GetPtrOrThrow()->StrideVec();
             return py::tuple(py::make_iterator(stride.begin(), stride.end()));
           })
      .def("is_contiguous", &ApiIsContiguous)
      .def_property_readonly("grad_fn", &Tensor::grad_fn_node)
      .def_property_readonly("is_leaf", &Tensor::is_leaf)
      .def_property("requires_grad", &Tensor::requires_grad, &ApiSetRequiresGrad)
      // Methods of pytorch
      .def(
          "requires_grad_",
          [](Tensor& t, bool requires_grad) -> Tensor& {
            ApiSetRequiresGrad(t, requires_grad);
            return t;
          },
          "requires_grad"_a = true)
      .def("retain_grad",
           [](Tensor& t) {
             if (!t.is_leaf()) { t.set_retain_grad(true).GetOrThrow(); }
           })
      .def("detach", [](const Tensor& t) { return t.detach().GetPtrOrThrow(); })
      .def("clone", [](const Tensor& t) { return t.clone().GetPtrOrThrow(); })
      // OneFlow tensor properties other than pytorch tensor
      .def_property_readonly("is_lazy", &Tensor::is_lazy)
      .def_property_readonly("is_eager", &Tensor::is_eager)
      .def_property_readonly("is_global", &Tensor::is_consistent)
      .def_property_readonly("is_local", &Tensor::is_local)
      .def("zeros_", &ApiEagerMirroredTensorZeros)
      .def("register_hook", &ApiRegisterTensorHook)
      .def("_register_post_grad_accumulation_hook", &ApiRegisterTensorPostGradAccumulationHook)
      // local tensor only
      .def_property_readonly("_tensor_buffer_shapes_and_dtypes", &GetTensorBufferShapesAndDTypes)
      .def_property_readonly("device", &TensorGetDevice)
      .def("global_id",
           [](const one::Tensor& tensor) -> int64_t {
             return static_cast<uint64_t>(tensor.transport_token().GetOrThrow());
           })
      .def("check_meta_consistency",
           [](const std::shared_ptr<one::Tensor>& tensor) {
             return CheckMetaConsistency(tensor).GetOrThrow();
           })
      .def("to_numpy", &ApiEagerTensorToNumpy, py::return_value_policy::move)
#define DEFINE_TENSOR_METHOD(T, type_proto)                    \
  .def("_copy_to_numpy_" #T, &ApiCopyMirroredTensorToNumpy<T>) \
      .def("_copy_from_numpy_" #T, &ApiCopyMirroredTensorFromNumpy<T>)
          OF_PP_FOR_EACH_TUPLE(DEFINE_TENSOR_METHOD, POD_DATA_TYPE_SEQ)
#undef DEFINE_TENSOR_METHOD
      .def("_get_copy_mirrored_tensor_to_numpy_func_name", &ApiGetCopyMirroredTensorToNumpyFuncName)
      .def("_get_copy_mirrored_tensor_from_numpy_func_name",
           &ApiGetCopyMirroredTensorFromNumpyFuncName)
      .def("_register_storage_delete_hook", &ApiRegisterStorageDeleteHook)
      // consistent tensor only
      .def_property_readonly("placement", &TensorGetParallelDesc)
      .def_property_readonly("sbp", &ApiTensorGetPyTupleOfSbp);

  auto nn = m.def_submodule("nn");
  py::class_<Parameter, std::shared_ptr<Parameter>, Tensor>(nn, "Parameter")
      .def(py::init(&ApiNewParameter), "data"_a, "requires_grad"_a = true);
}

}  // namespace one

}  // namespace oneflow
