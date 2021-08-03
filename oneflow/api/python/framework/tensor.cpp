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

#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/ofblob/ofblob.e.h"
#include "oneflow/api/python/framework/device.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/framework/tensor_method.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stride.h"
#include "oneflow/core/framework/py_distribute.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/autograd/autograd_meta.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/extension/python/numpy.h"

namespace py = pybind11;

namespace oneflow {

namespace one {

namespace {

const DType* GetTensorDType(const Tensor& tensor) {
  return DType::Get(tensor.dtype()).GetOrThrow().get();
}

std::shared_ptr<Tensor> MakeLocalTensor(const std::shared_ptr<const Shape>& shape,
                                        const DType* dtype, const Symbol<Device>& device,
                                        bool is_lazy, bool requires_grad, bool is_leaf) {
  return MirroredTensor::MakeTensor(shape, dtype->data_type(), device, is_lazy, requires_grad,
                                    is_leaf)
      .GetPtrOrThrow();
}

std::shared_ptr<Tensor> MakeConsistentTensor(
    const std::shared_ptr<const Shape>& shape, const DType* dtype,
    Symbol<cfg::ParallelDistribution>& parallel_distribution, Symbol<ParallelDesc> parallel_desc,
    bool is_lazy, bool requires_grad, bool is_leaf) {
  return ConsistentTensor::MakeTensor(shape, dtype->data_type(), parallel_distribution,
                                      parallel_desc, is_lazy, requires_grad, is_leaf)
      .GetPtrOrThrow();
}

Maybe<void> EagerMirroredTensorZeros(const std::shared_ptr<Tensor>& t) {
  const auto& tensor = JUST(t->AsMirroredTensor());
  CHECK_OR_RETURN(tensor->is_eager()) << "eager tensors supported only";
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    JUST(builder->AccessBlobByCallback(
        tensor,
        [](uint64_t of_blob_ptr) {
          auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
          of_blob->AsyncAutoMemset(0);
        },
        "mut"));
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

void ApiEagerMirroredTensorZeros(const std::shared_ptr<Tensor>& tensor) {
  return EagerMirroredTensorZeros(tensor).GetOrThrow();
}

template<typename T>
Maybe<void> CopyBetweenMirroredTensorAndNumpy(const std::shared_ptr<Tensor>& t,
                                              py::array_t<T> array,
                                              void (*Copy)(uint64_t, py::array_t<T>),
                                              const std::string& modifier) {
  auto tensor = JUST(t->AsMirroredTensor());
  CHECK_OR_RETURN(tensor->is_eager()) << "eager tensors supported only";
  std::atomic<bool> synced(false);

  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    JUST(builder->AccessBlobByCallback(
        tensor,
        [&array, &synced, &Copy](uint64_t ofblob_ptr) {
          Copy(ofblob_ptr, array);
          synced = true;
        },
        modifier));
    return Maybe<void>::Ok();
  }));

  Global<ForeignLockHelper>::Get()->WithScopedRelease([&synced]() { /* spin wait */
                                                                    while (!synced) {}
  });

  return Maybe<void>::Ok();
}

template<typename T>
void ApiCopyMirroredTensorToNumpy(const std::shared_ptr<Tensor>& tensor, py::array_t<T> array) {
  return CopyBetweenMirroredTensorAndNumpy(tensor, array, OfBlob_CopyToBuffer, "const")
      .GetOrThrow();
}

template<typename T>
void ApiCopyMirroredTensorFromNumpy(const std::shared_ptr<Tensor>& tensor, py::array_t<T> array) {
  return CopyBetweenMirroredTensorAndNumpy(tensor, array, OfBlob_CopyFromBuffer, "mut")
      .GetOrThrow();
}

template<typename T>
Maybe<void> CopyMirroredTensorFromUntypedArray(const std::shared_ptr<Tensor>& tensor,
                                               py::object array) {
  return CopyBetweenMirroredTensorAndNumpy(tensor, array.cast<py::array_t<T>>(),
                                           OfBlob_CopyFromBuffer, "mut");
}

#define MAKE_SWITCH_ENTRY(func_name, dtype) func_name<dtype>
DEFINE_STATIC_SWITCH_FUNC(Maybe<void>, CopyMirroredTensorFromUntypedArray, MAKE_SWITCH_ENTRY,
                          MAKE_DATA_TYPE_CTRV_SEQ(POD_DATA_TYPE_SEQ));

Maybe<std::string> GetCopyMirroredTensorToNumpyFuncName(DataType dtype) {
  using namespace oneflow;
  static const HashMap<int64_t, std::shared_ptr<std::string>> data_type2func_name{
#define DATA_TYPE_FUNC_NAME_PAIR(type_cpp, type_proto) \
  {type_proto, std::make_shared<std::string>("_copy_to_numpy_" #type_cpp)},
      OF_PP_FOR_EACH_TUPLE(DATA_TYPE_FUNC_NAME_PAIR, POD_DATA_TYPE_SEQ)
#undef DATA_TYPE_FUNC_NAME_PAIR
  };
  return JUST(MapAt(data_type2func_name, static_cast<int64_t>(dtype)));
}

const std::string& ApiGetCopyMirroredTensorToNumpyFuncName(const Tensor& tensor) {
  return *GetCopyMirroredTensorToNumpyFuncName(tensor.dtype()).GetPtrOrThrow();
}

Maybe<std::string> GetCopyMirroredTensorFromNumpyFuncName(DataType dtype) {
  using namespace oneflow;
  static const HashMap<int64_t, std::shared_ptr<std::string>> data_type2func_name{
#define DATA_TYPE_FUNC_NAME_PAIR(type_cpp, type_proto) \
  {type_proto, std::make_shared<std::string>("_copy_from_numpy_" #type_cpp)},
      OF_PP_FOR_EACH_TUPLE(DATA_TYPE_FUNC_NAME_PAIR, POD_DATA_TYPE_SEQ)
#undef DATA_TYPE_FUNC_NAME_PAIR
  };
  return JUST(MapAt(data_type2func_name, static_cast<int64_t>(dtype)));
}

const std::string& ApiGetCopyMirroredTensorFromNumpyFuncName(const Tensor& tensor) {
  return *GetCopyMirroredTensorFromNumpyFuncName(tensor.dtype()).GetPtrOrThrow();
}

Maybe<Tensor> MakeLocalTensorByNumpy(py::object array, const DType* desired_dtype,
                                     const Symbol<Device>& device, bool requires_grad) {
  // Executing any numpy c api before _import_array() results in segfault
  if (PyArray_API == nullptr) { _import_array(); }
  auto* np_arr_pyobject = PyArray_FromAny(array.ptr(), nullptr, 0, 0, NPY_ARRAY_DEFAULT, nullptr);
  CHECK_NOTNULL_OR_RETURN(np_arr_pyobject) << "input data cannot convert to a numpy array";
  // transfer the ownership to np_arr_raii so that the ref count
  // can be decreased automatically when function exits either normally or abnormally
  auto np_arr_raii = py::reinterpret_steal<py::array>(np_arr_pyobject);
  auto* np_arr = reinterpret_cast<PyArrayObject*>(np_arr_pyobject);
  bool init_from_numpy = py::isinstance<py::array>(array);
  const npy_intp* dims_ptr = PyArray_SHAPE(np_arr);
  const auto shape = std::make_shared<Shape>(DimVector(dims_ptr, dims_ptr + PyArray_NDIM(np_arr)));
  DataType flow_dtype = JUST(numpy::GetOFDataTypeFromNpArray(np_arr));
  std::shared_ptr<Tensor> tensor =
      MirroredTensor::MakeTensor(shape, flow_dtype, device, /* is_lazy */ false, requires_grad,
                                 /* is_leaf */ true)
          .GetPtrOrThrow();
  JUST(SwitchCopyMirroredTensorFromUntypedArray(SwitchCase(flow_dtype), tensor, np_arr_raii));
  if (flow_dtype == DataType::kDouble && !init_from_numpy && desired_dtype == nullptr) {
    desired_dtype = DType::Float().get();
  }
  if (desired_dtype != nullptr) {
    autograd::NoGradGuard no_grad;
    tensor = JUST(functional::Cast(tensor, desired_dtype->data_type()));
    tensor->set_requires_grad(requires_grad);
  }
  return tensor;
}

Symbol<Device> TensorGetDevice(const Tensor& tensor) { return tensor.device().GetOrThrow(); }

Symbol<ParallelDesc> TensorGetParallelDesc(const Tensor& tensor) {
  return tensor.parallel_desc().GetOrThrow();
}

Maybe<std::tuple<std::vector<Shape>, std::vector<const DType*>>>
MaybeGetTensorBufferShapesAndDTypes(const std::shared_ptr<Tensor>& t) {
  const auto& tensor = JUST(t->AsMirroredTensor());
  CHECK_OR_RETURN(tensor->is_eager()) << "eager tensors supported only";
  std::vector<Shape> shapes;
  std::vector<const DType*> dtypes;
  std::atomic<bool> synced(false);

  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    JUST(builder->AccessBlobByCallback(
        tensor, [&synced](uint64_t of_blob_ptr) { synced = true; }, "const"));
    return Maybe<void>::Ok();
  }));

  Global<ForeignLockHelper>::Get()->WithScopedRelease([&synced]() {
    while (!synced) {}
  });

  const Blob& blob = JUST(tensor->eager_blob_object())->blob();
  const Shape& blob_shape = blob.static_shape();
  const auto* tensor_buffer_ptr = blob.dptr<TensorBuffer>();
  for (int64_t i = 0; i < blob_shape.elem_cnt(); ++i) {
    const TensorBuffer* tensor_buffer = tensor_buffer_ptr + i;
    shapes.push_back(tensor_buffer->shape());
    dtypes.push_back(DType::Get(tensor_buffer->data_type()).GetOrThrow().get());
  }
  return std::make_tuple(shapes, dtypes);
}

std::tuple<std::vector<Shape>, std::vector<const DType*>> GetTensorBufferShapesAndDTypes(
    const std::shared_ptr<Tensor>& tensor) {
  return MaybeGetTensorBufferShapesAndDTypes(tensor).GetOrThrow();
}

Maybe<void> RegisterTensorHook(const std::shared_ptr<Tensor>& self,
                               const AutogradMeta::Hook& hook) {
  if (!self->grad_fn_node()) { JUST(AddAccumulateFunctionNode(self)); }
  self->mut_autograd_meta()->add_hook(hook);
  return Maybe<void>::Ok();
}
void ApiRegisterTensorHook(const std::shared_ptr<Tensor>& self, const AutogradMeta::Hook& hook) {
  return RegisterTensorHook(self, hook).GetOrThrow();
}

Maybe<void> CheckConsistentTensorMeta(const one::Tensor& tensor, int64_t seconds) {
  const auto& ctx = JUST(LaunchTensorMetaConsistencyCheck(tensor));
  JUST(RpcUtil::WaitUntilDoneOrTimeout(*ctx, seconds));
  JUST(ctx->Check());
  return Maybe<void>::Ok();
}

bool ApiIsContiguous(const std::shared_ptr<Tensor>& tensor) {
  return IsContiguous(tensor).GetOrThrow();
}

Maybe<py::tuple> TensorGetPyTupleOfSbp(const Tensor& tensor) {
  const auto& nd_sbp = JUST(tensor.parallel_distribution());
  const auto& tuple = std::make_shared<py::tuple>(nd_sbp->sbp_parallel_size());
  for (int i = 0; i < nd_sbp->sbp_parallel_size(); ++i) {
    (*tuple)[i] = SymbolOf(nd_sbp->sbp_parallel(i));
  }
  return tuple;
}

py::tuple ApiTensorGetPyTupleOfSbp(const Tensor& tensor) {
  return *TensorGetPyTupleOfSbp(tensor).GetPtrOrThrow();
}

Maybe<Tensor> NewTensor(py::args args, py::kwargs kwargs, const DType* desired_dtype,
                        bool treat_single_int_as_size) {
  Symbol<Device> device;
  if (kwargs.contains("device")) {
    const auto& device_kwarg = kwargs["device"];
    CHECK_OR_RETURN(py::isinstance<Symbol<Device>>(device_kwarg)
                    || py::isinstance<py::str>(device_kwarg));

    if (py::isinstance<py::str>(device_kwarg)) {
      device = DeviceExportUtil::MakeDevice(py::cast<std::string>(device_kwarg));
    } else {
      device = py::cast<Symbol<Device>>(device_kwarg);
    }
  } else {
    device = JUST(Device::New("cpu"));
  }

  desired_dtype = kwargs.contains("dtype") ? kwargs["dtype"].cast<const DType*>() : desired_dtype;

  bool requires_grad = false;
  if (kwargs.contains("requires_grad")) { requires_grad = kwargs["requires_grad"].cast<bool>(); }

  if (args.size() == 1) {
    const auto& arg = args[0];
    if (py::isinstance<Tensor>(arg)) {
      std::shared_ptr<Tensor> other_tensor = py::cast<std::shared_ptr<Tensor>>(arg);
      std::shared_ptr<Tensor> tensor =
          JUST(functional::Copy(other_tensor, device->type(), device->device_id()));
      if (desired_dtype != nullptr && desired_dtype->data_type() != tensor->dtype()) {
        tensor = JUST(functional::Cast(tensor, desired_dtype->data_type()));
      }
      return tensor;
    } else {
      if (!treat_single_int_as_size || !py::isinstance<py::int_>(arg)) {
        return MakeLocalTensorByNumpy(arg, desired_dtype, device, requires_grad);
      }
    }
  }
  DimVector dim_vector;
  for (const auto& arg : args) {
    try {
      dim_vector.push_back(py::cast<int64_t>(arg));
    } catch (const py::cast_error& e) {
      return Error::ValueError("invalid arg: " + py::str(arg).cast<std::string>());
    }
  }
  CHECK_NOTNULL_OR_RETURN(desired_dtype);
  std::shared_ptr<MirroredTensor> tensor = JUST(
      MirroredTensor::MakeTensor(std::make_shared<Shape>(dim_vector), desired_dtype->data_type(),
                                 device, /* is_lazy */ false, requires_grad, /* is_leaf */ true));
  return std::static_pointer_cast<Tensor>(tensor);
}

std::shared_ptr<Tensor> ApiNewTensor(py::args args, py::kwargs kwargs) {
  return NewTensor(args, kwargs, DType::Float().get(), true).GetPtrOrThrow();
}

void ApiSetRequiresGrad(Tensor& tensor, bool requires_grad) {
  if (tensor.is_leaf()) {
    tensor.set_requires_grad(requires_grad);
  } else {
    throw std::runtime_error("You can only change requires_grad flags of leaf tensors.");
  }
}

std::shared_ptr<Parameter> ApiNewParameter(const std::shared_ptr<Tensor>& data,
                                           bool requires_grad) {
  return std::make_shared<Parameter>(data, requires_grad);
}

}  // namespace

using namespace pybind11::literals;

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("tensor", [](py::args args, py::kwargs kwargs) -> std::shared_ptr<Tensor> {
    return NewTensor(args, kwargs, nullptr, false).GetPtrOrThrow();
  });
  py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
      .def(py::init(&ApiNewTensor))
      // Properties of pytorch
      .def_property_readonly("ndim", &Tensor::ndim)
      .def_property_readonly("shape", &Tensor::shape)
      .def_property_readonly("dtype", &GetTensorDType)
      .def_property_readonly("is_cuda", &Tensor::is_cuda)
      .def_property(
          "grad",
          [](const Tensor& t) -> std::shared_ptr<Tensor> {
            if (t.has_autograd_meta()) {
              return t.acc_grad().GetPtrOrThrow();
            } else {
              return std::shared_ptr<Tensor>();
            }
          },
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
      .def_property_readonly("is_consistent", &Tensor::is_consistent)
      .def_property_readonly("is_local", &Tensor::is_local)
      .def("zeros_", &ApiEagerMirroredTensorZeros)
      .def("register_hook", &ApiRegisterTensorHook)
      // local tensor only
      .def_property_readonly("_tensor_buffer_shapes_and_dtypes", &GetTensorBufferShapesAndDTypes)
      .def_property_readonly("device", &TensorGetDevice)
      .def_property_readonly("data", &Tensor::data)
      .def("rpc_token",
           [](const one::Tensor& tensor) -> int64_t {
             return static_cast<uint64_t>(tensor.rpc_token().GetOrThrow());
           })
      .def("check_meta_consistency",
           [](const one::Tensor& tensor) {
             return CheckConsistentTensorMeta(tensor, 60 * 5).GetOrThrow();
           })
      .def("check_meta_consistency",
           [](const one::Tensor& tensor, int64_t seconds) {
             return CheckConsistentTensorMeta(tensor, seconds).GetOrThrow();
           })
#define DEFINE_TENSOR_METHOD(T, type_proto)                    \
  .def("_copy_to_numpy_" #T, &ApiCopyMirroredTensorToNumpy<T>) \
      .def("_copy_from_numpy_" #T, &ApiCopyMirroredTensorFromNumpy<T>)
          OF_PP_FOR_EACH_TUPLE(DEFINE_TENSOR_METHOD, POD_DATA_TYPE_SEQ)
#undef DEFINE_TENSOR_METHOD
      .def("_get_copy_mirrored_tensor_to_numpy_func_name", &ApiGetCopyMirroredTensorToNumpyFuncName)
      .def("_get_copy_mirrored_tensor_from_numpy_func_name",
           &ApiGetCopyMirroredTensorFromNumpyFuncName)
      // consistent tensor only
      .def_property_readonly("placement", &TensorGetParallelDesc)
      .def_property_readonly("sbp", &ApiTensorGetPyTupleOfSbp);

  auto nn = m.def_submodule("nn");
  py::class_<Parameter, std::shared_ptr<Parameter>, Tensor>(nn, "Parameter")
      .def(py::init(&ApiNewParameter), "data"_a, "requires_grad"_a = true);
}

}  // namespace one

}  // namespace oneflow
