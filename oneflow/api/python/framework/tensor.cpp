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
#include "oneflow/api/python/framework/tensor.h"

#include <pybind11/pybind11.h>
#include <Python.h>
#include "oneflow/api/python/exception/exception.h"
#include "oneflow/api/python/framework/size.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/functional/functional_api.yaml.pybind.h"
#include "oneflow/api/python/functional/tensor_api.yaml.pybind.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/ofblob/ofblob.e.h"
#include "oneflow/api/python/utils/tensor_utils.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stride.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/placement_utils.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/tensor_index.h"

namespace py = pybind11;

namespace oneflow {
namespace one {

#define ASSERT(x) (x).GetOrThrow()
#define ASSERT_PTR(x) (x).GetPtrOrThrow()

static int PyTensorObject_init(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  if (self) {
    PyObject* obj = functional::_legacy_tensor_ctor(NULL, args, kwargs);
    ((PyTensorObject*)self)->data = PyTensor_Unpack(obj);
    Py_XDECREF(obj);
  }
  return 0;
  END_HANDLE_ERRORS_RET(-1)
}

static PyObject* PyTensorObject_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  auto* self = (PyTensorObject*)type->tp_alloc(type, 0);
  if (PyTensorObject_init((PyObject*)self, args, kwargs) != 0) { return NULL; }
  return (PyObject*)self;
}

static int PyParameterObject_init(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* data = NULL;
  bool requires_grad = true;
  static const char* keywords[2] = {"data", "requires_grad"};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p:__init__", const_cast<char**>(keywords),
                                   &data, &requires_grad)) {
    return -1;
  }
  if (self) {
    ((PyTensorObject*)self)->data =
        ASSERT_PTR(Parameter::MakeTensor(PyTensor_Unpack(data), requires_grad));
  }
  return 0;
  END_HANDLE_ERRORS_RET(-1)
}

static PyObject* PyParameterObject_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* data = NULL;
  bool requires_grad = true;
  static const char* keywords[2] = {"data", "requires_grad"};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p:__new__", const_cast<char**>(keywords), &data,
                                   &requires_grad)) {
    return NULL;
  }
  auto* self = (PyTensorObject*)PyTensorObject_Type->tp_alloc(type, 0);
  if (self) {
    self->data = ASSERT_PTR(Parameter::MakeTensor(PyTensor_Unpack(data), requires_grad));
  }
  return (PyObject*)self;
  END_HANDLE_ERRORS
}

static Py_ssize_t PyTensorObject_length(PyTensorObject* self) {
  if (self->data->ndim() == 0) { return 0; }
  return self->data->dim(0);
}

static PyObject* PyTensorObject_getitem(PyObject* self, Py_ssize_t item) {
  HANDLE_ERRORS
  const auto& p = PyTensor_Unpack(self);
  return PyTensor_New(
      ASSERT_PTR(functional::TensorGetItem(p, {functional::detail::IndexItem(item)})));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_subscript(TensorSize* self, PyObject* item) {
  HANDLE_ERRORS
  PyObject* args = PyTuple_Pack(2, (PyObject*)self, item);
  PyObject* result = functional::tensor_getitem(NULL, args, NULL);
  Py_XDECREF(args);
  return result;
  END_HANDLE_ERRORS
}

static int PyTensorObject_ass_subscript(TensorSize* self, PyObject* item, PyObject* value) {
  HANDLE_ERRORS
  PyObject* args = PyTuple_Pack(3, (PyObject*)self, item, value);
  functional::tensor_setitem(NULL, args, NULL);
  Py_XDECREF(args);
  return 0;
  END_HANDLE_ERRORS_RET(-1)
}

static PySequenceMethods PyTensorObject_as_sequence = {
    (lenfunc)PyTensorObject_length, NULL, /*sq_concat*/
    NULL,                                 /*sq_repeat*/
    (ssizeargfunc)PyTensorObject_getitem, /*sq_item*/
};

static PyMappingMethods PyTensorObject_as_mapping = {
    (lenfunc)PyTensorObject_length,
    (binaryfunc)PyTensorObject_subscript,
    (objobjargproc)PyTensorObject_ass_subscript,
};

static PyObject* PyTensorObject_storage_offset(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  return functional::CastToPyObject(PyTensor_Unpack(self)->storage_offset());
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_stride(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  return functional::CastToPyObject(PyTensor_Unpack(self)->stride());
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_is_contiguous(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  return functional::CastToPyObject(PyTensor_Unpack(self)->is_contiguous());
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_contiguous(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  return PyTensor_New(PyTensor_Unpack(self)->contiguous());
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_requires_grad_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  bool requires_grad = true;
  static const char* keywords[1] = {"requires_grad"};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|p:requires_grad_", const_cast<char**>(keywords),
                                   &requires_grad)) {
    return NULL;
  }
  ASSERT(PyTensor_Unpack(self)->set_requires_grad(requires_grad));
  return self;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_retain_grad(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  const auto& t = PyTensor_Unpack(self);
  if (!t->requires_grad()) {
    return PyErr_Format(PyExc_RuntimeError,
                        "can't retain_grad on Tensor that has requires_grad=False");
  }
  ASSERT(t->set_retain_grad(true));
  return Py_None;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_detach(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  return PyTensor_New(ASSERT_PTR(PyTensor_Unpack(self)->detach()));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_clone(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  return PyTensor_New(ASSERT_PTR(PyTensor_Unpack(self)->clone()));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_zeros_(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  ASSERT(EagerMirroredTensorZeros(PyTensor_Unpack(self)));
  return self;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_register_hook(PyObject* self, PyObject* hook) {
  HANDLE_ERRORS
  const auto& _hook = py::cast<AutogradMeta::Hook>(py::reinterpret_borrow<py::object>(hook));
  ASSERT(RegisterTensorHook(PyTensor_Unpack(self), _hook));
  return Py_None;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject__register_post_grad_accumulation_hook(PyObject* self,
                                                                      PyObject* hook) {
  HANDLE_ERRORS
  const auto& _hook = py::cast<AutogradMeta::Hook>(py::reinterpret_borrow<py::object>(hook));
  ASSERT(RegisterTensorPostGradAccumulationHook(PyTensor_Unpack(self), _hook));
  return Py_None;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_global_id(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  uint64_t global_id = static_cast<uint64_t>(ASSERT(PyTensor_Unpack(self)->transport_token()));
  return functional::CastToPyObject(global_id);
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_check_meta_consistency(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  ASSERT(CheckMetaConsistency(PyTensor_Unpack(self)));
  return Py_None;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_to_numpy(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  const auto& t = PyTensor_Unpack(self);
  DataType data_type = t->dtype()->data_type();
  switch (data_type) {
#define SWITCH_EAGER_TENSOR_TO_NUMPY(cpp_type, of_type) \
  case of_type: return ASSERT(EagerMirroredTensorToNumpy<cpp_type>(self));
    OF_PP_FOR_EACH_TUPLE(SWITCH_EAGER_TENSOR_TO_NUMPY, POD_DATA_TYPE_SEQ)
    case DataType::kFloat16: return ASSERT(EagerMirroredTensorToNumpy<float16>(self));
    default: {
      return PyErr_Format(PyExc_RuntimeError, "Invalid datatype");
    }
  }
#undef SWITCH_EAGER_TENSOR_TO_NUMPY
  END_HANDLE_ERRORS
}

#define DEFINE_TENSOR_METHOD(T, type_proto)                                               \
  static PyObject* PyTensorObject__copy_to_numpy_##T(PyObject* self, PyObject* array) {   \
    HANDLE_ERRORS                                                                         \
    ASSERT(CopyBetweenMirroredTensorAndNumpy<T>(PyTensor_Unpack(self), array,             \
                                                BlobNumpyCopyUtil<T>::To, "const",        \
                                                /*block_host_until_done=*/true));         \
    return Py_None;                                                                       \
    END_HANDLE_ERRORS                                                                     \
  }                                                                                       \
  static PyObject* PyTensorObject__copy_from_numpy_##T(PyObject* self, PyObject* array) { \
    HANDLE_ERRORS                                                                         \
    auto* copied = PyArray_NewCopy((PyArrayObject*)array, NPY_CORDER);                    \
    ASSERT(CopyBetweenMirroredTensorAndNumpy<T>(PyTensor_Unpack(self), copied,            \
                                                BlobNumpyCopyUtil<T>::From, "mut",        \
                                                /*block_host_until_done=*/false));        \
    return Py_None;                                                                       \
    END_HANDLE_ERRORS                                                                     \
  }
OF_PP_FOR_EACH_TUPLE(DEFINE_TENSOR_METHOD, POD_DATA_TYPE_SEQ)
#undef DEFINE_TENSOR_METHOD

static PyObject* PyTensorObject__get_copy_mirrored_tensor_to_numpy_func_name(PyObject* self,
                                                                             PyObject* unused) {
  HANDLE_ERRORS
  return functional::CastToPyObject(
      GetCopyMirroredTensorToNumpyFuncName(PyTensor_Unpack(self)->dtype()->data_type()));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject__get_copy_mirrored_tensor_from_numpy_func_name(PyObject* self,
                                                                               PyObject* unused) {
  HANDLE_ERRORS
  return functional::CastToPyObject(
      GetCopyMirroredTensorFromNumpyFuncName(PyTensor_Unpack(self)->dtype()->data_type()));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject__register_storage_delete_hook(PyObject* self, PyObject* hook) {
  HANDLE_ERRORS
  auto _hook = py::cast<std::function<void()>>(py::reinterpret_borrow<py::object>(hook));
  ASSERT(PyTensor_Unpack(self)->RegisterStorageDeleteHook(_hook));
  return Py_None;
  END_HANDLE_ERRORS
}

static PyMethodDef PyTensorObject_methods[] = {
    {"storage_offset", PyTensorObject_storage_offset, METH_NOARGS, NULL},
    {"stride", PyTensorObject_stride, METH_NOARGS, NULL},
    {"is_contiguous", PyTensorObject_is_contiguous, METH_NOARGS, NULL},
    {"contiguous", PyTensorObject_contiguous, METH_NOARGS, NULL},
    {"requires_grad_", (PyCFunction)PyTensorObject_requires_grad_, METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"retain_grad", PyTensorObject_retain_grad, METH_NOARGS, NULL},
    {"detach", PyTensorObject_detach, METH_NOARGS, NULL},
    {"clone", PyTensorObject_clone, METH_NOARGS, NULL},
    {"zeros_", PyTensorObject_zeros_, METH_NOARGS, NULL},
    {"register_hook", PyTensorObject_register_hook, METH_O, NULL},
    {"_register_post_grad_accumulation_hook", PyTensorObject__register_post_grad_accumulation_hook,
     METH_VARARGS, NULL},
    {"global_id", PyTensorObject_global_id, METH_NOARGS, NULL},
    {"check_meta_consistency", PyTensorObject_check_meta_consistency, METH_NOARGS, NULL},
    {"to_numpy", PyTensorObject_to_numpy, METH_VARARGS, NULL},
#define DEFINE_TENSOR_METHOD(T, type_proto)                                \
  {"_copy_to_numpy_" #T, PyTensorObject__copy_to_numpy_##T, METH_O, NULL}, \
      {"_copy_from_numpy_" #T, PyTensorObject__copy_from_numpy_##T, METH_O, NULL},
    OF_PP_FOR_EACH_TUPLE(DEFINE_TENSOR_METHOD, POD_DATA_TYPE_SEQ)
#undef DEFINE_TENSOR_METHOD
        {"_get_copy_mirrored_tensor_to_numpy_func_name",
         PyTensorObject__get_copy_mirrored_tensor_to_numpy_func_name, METH_NOARGS, NULL},
    {"_get_copy_mirrored_tensor_from_numpy_func_name",
     PyTensorObject__get_copy_mirrored_tensor_from_numpy_func_name, METH_NOARGS, NULL},
    {"_register_storage_delete_hook", PyTensorObject__register_storage_delete_hook, METH_VARARGS,
     NULL},
    {NULL}};

static PyObject* PyTensorObject_ndim(PyObject* self, void* unused) {
  return functional::CastToPyObject(PyTensor_Unpack(self)->ndim());
}

static PyObject* PyTensorObject_shape(PyObject* self, void* unused) {
  return functional::CastToPyObject(PyTensor_Unpack(self)->shape());
}

static PyObject* PyTensorObject_dtype(PyObject* self, void* unused) {
  HANDLE_ERRORS
  const Symbol<DType>* dtype = &ASSERT(DType::Get(PyTensor_Unpack(self)->dtype()->data_type()));
  return functional::CastToPyObject(dtype);
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_is_cuda(PyObject* self, void* unused) {
  return functional::CastToPyObject(PyTensor_Unpack(self)->is_cuda());
}

static PyObject* PyTensorObject_grad(PyObject* self, void* unused) {
  HANDLE_ERRORS
  return PyTensor_New(ASSERT_PTR(PyTensor_Unpack(self)->acc_grad()));
  END_HANDLE_ERRORS
}

static int PyTensorObject_set_grad(PyObject* self, PyObject* grad, void* unused) {
  HANDLE_ERRORS
  const auto& t = PyTensor_Unpack(self);
  if (self == grad) { PyErr_Format(PyExc_RuntimeError, "can't assign Tensor as its own grad"); }
  if (grad && grad != Py_None) {
    ASSERT(t->set_acc_grad(ASSERT_PTR(PyTensor_Unpack(grad)->detach())));
  } else {
    ASSERT(t->set_acc_grad(NULL));
  }
  return 0;
  END_HANDLE_ERRORS_RET(-1)
}

static PyObject* PyTensorObject__is_grad_acc_inplace(PyObject* self, void* unused) {
  return functional::CastToPyObject(PyTensor_Unpack(self)->autograd_meta()->is_grad_acc_inplace());
}

static int PyTensorObject_set__is_grad_acc_inplace(PyObject* self, PyObject* is_inplace,
                                                   void* unused) {
  PyTensor_Unpack(self)->mut_autograd_meta()->set_is_grad_acc_inplace(is_inplace);
  return 0;
}

static PyObject* PyTensorObject_data(PyObject* self, void* unused) {
  HANDLE_ERRORS
  return PyTensor_New(ASSERT_PTR(PyTensor_Unpack(self)->data()));
  END_HANDLE_ERRORS
}

static int PyTensorObject_set_data(PyObject* self, PyObject* data, void* unused) {
  HANDLE_ERRORS
  const auto& t = PyTensor_Unpack(self);
  auto hooks = t->autograd_meta()->hooks();
  ASSERT(t->set_data(PyTensor_Unpack(data)));
  // Re-register hooks
  for (const auto& hook : hooks) { ASSERT(RegisterTensorHook(t, hook)); }
  return 0;
  END_HANDLE_ERRORS_RET(-1)
}

static PyObject* PyTensorObject_grad_fn(PyObject* self, void* unused) {
  return functional::CastToPyObject(PyTensor_Unpack(self)->grad_fn_node());
}

static PyObject* PyTensorObject_is_leaf(PyObject* self, void* unused) {
  return functional::CastToPyObject(PyTensor_Unpack(self)->is_leaf());
}

static PyObject* PyTensorObject_requires_grad(PyObject* self, void* unused) {
  return functional::CastToPyObject(PyTensor_Unpack(self)->requires_grad());
}

static int PyTensorObject_set_requires_grad(PyObject* self, PyObject* requires_grad, void* unused) {
  HANDLE_ERRORS
  const auto& t = PyTensor_Unpack(self);
  CHECK_OR_THROW(t->is_leaf()) << Error::RuntimeError()
                               << "You can only change requires_grad flags of leaf tensors.";
  ASSERT(t->set_requires_grad(requires_grad == Py_True));
  return 0;
  END_HANDLE_ERRORS_RET(-1)
}

static PyObject* PyTensorObject_is_lazy(PyObject* self, void* unused) {
  return functional::CastToPyObject(PyTensor_Unpack(self)->is_lazy());
}

static PyObject* PyTensorObject_is_eager(PyObject* self, void* unused) {
  return functional::CastToPyObject(PyTensor_Unpack(self)->is_eager());
}

static PyObject* PyTensorObject_is_global(PyObject* self, void* unused) {
  return functional::CastToPyObject(PyTensor_Unpack(self)->is_consistent());
}

static PyObject* PyTensorObject_is_local(PyObject* self, void* unused) {
  return functional::CastToPyObject(PyTensor_Unpack(self)->is_local());
}

static PyObject* PyTensorObject__tensor_buffer_shapes_and_dtypes(PyObject* self, void* unused) {
  HANDLE_ERRORS
  return functional::CastToPyObject(MaybeGetTensorBufferShapesAndDTypes(PyTensor_Unpack(self)));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_device(PyObject* self, void* unused) {
  HANDLE_ERRORS
  return functional::CastToPyObject(PyTensor_Unpack(self)->device());
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_placement(PyObject* self, void* unused) {
  HANDLE_ERRORS
  return functional::CastToPyObject(PyTensor_Unpack(self)->parallel_desc());
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_sbp(PyObject* self, void* unused) {
  HANDLE_ERRORS
  return functional::CastToPyObject(TensorGetPyTupleOfSbp(*PyTensor_Unpack(self)));
  END_HANDLE_ERRORS
}

static PyGetSetDef PyTensorObject_properties[] = {
    {"ndim", (getter)PyTensorObject_ndim, NULL, NULL, NULL},
    {"shape", (getter)PyTensorObject_shape, NULL, NULL, NULL},
    {"dtype", (getter)PyTensorObject_dtype, NULL, NULL, NULL},
    {"is_cuda", (getter)PyTensorObject_is_cuda, NULL, NULL, NULL},
    {"grad", (getter)PyTensorObject_grad, (setter)PyTensorObject_set_grad, NULL, NULL},
    {"_is_grad_acc_inplace", (getter)PyTensorObject__is_grad_acc_inplace,
     (setter)PyTensorObject_set__is_grad_acc_inplace, NULL, NULL},
    {"data", (getter)PyTensorObject_data, (setter)PyTensorObject_set_data, NULL, NULL},
    {"grad_fn", (getter)PyTensorObject_grad_fn, NULL, NULL, NULL},
    {"is_leaf", (getter)PyTensorObject_is_leaf, NULL, NULL, NULL},
    {"requires_grad", (getter)PyTensorObject_requires_grad,
     (setter)PyTensorObject_set_requires_grad, NULL, NULL},
    {"is_lazy", (getter)PyTensorObject_is_lazy, NULL, NULL, NULL},
    {"is_eager", (getter)PyTensorObject_is_eager, NULL, NULL, NULL},
    {"is_global", (getter)PyTensorObject_is_global, NULL, NULL, NULL},
    {"is_local", (getter)PyTensorObject_is_local, NULL, NULL, NULL},
    {"_tensor_buffer_shapes_and_dtypes", (getter)PyTensorObject__tensor_buffer_shapes_and_dtypes,
     NULL, NULL, NULL},
    {"device", (getter)PyTensorObject_device, NULL, NULL, NULL},
    {"placement", (getter)PyTensorObject_placement, NULL, NULL, NULL},
    {"sbp", (getter)PyTensorObject_sbp, NULL, NULL, NULL},
    {NULL}};

static PyTypeObject* MakeTensorMetaclass() {
  constexpr auto* name = "TensorMetaCls";
  auto heap_type = (PyHeapTypeObject*)PyType_Type.tp_alloc(&PyType_Type, 0);
  heap_type->ht_name = PyUnicode_FromString(name);
  heap_type->ht_qualname = PyUnicode_FromString(name);

  auto type = &heap_type->ht_type;
  type->tp_name = name;
  Py_INCREF(&PyType_Type);
  type->tp_base = &PyType_Type;
  type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;

  if (PyType_Ready(type) < 0) { return NULL; }
  PyObject_SetAttrString((PyObject*)type, "__module__", PyUnicode_FromString("oneflow_builtin"));
  return type;
}

static PyTypeObject* MakeTensorType() {
  PyTypeObject* metaclass = MakeTensorMetaclass();
  auto heap_type = (PyHeapTypeObject*)metaclass->tp_alloc(metaclass, 0);
  if (!heap_type) { return NULL; }
  heap_type->ht_name = PyUnicode_FromString("Tensor");
  heap_type->ht_qualname = PyUnicode_FromString("Tensor");
  auto type = &heap_type->ht_type;
  type->tp_name = "Tensor";
  type->tp_basicsize = sizeof(PyTensorObject);

  type->tp_init = PyTensorObject_init;
  type->tp_new = PyTensorObject_new;
  type->tp_getset = PyTensorObject_properties;
  type->tp_methods = PyTensorObject_methods;

  type->tp_as_number = &heap_type->as_number;
  type->tp_as_sequence = &PyTensorObject_as_sequence;
  type->tp_as_mapping = &PyTensorObject_as_mapping;

  type->tp_flags |= Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE;

  if (PyType_Ready(type) < 0) { return NULL; }
  return type;
}

static PyTypeObject* MakeParameterType() {
  PyTypeObject* metaclass = MakeTensorMetaclass();
  auto heap_type = (PyHeapTypeObject*)metaclass->tp_alloc(metaclass, 0);
  if (!heap_type) { return NULL; }
  heap_type->ht_name = PyUnicode_FromString("nn.Parameter");
  heap_type->ht_qualname = PyUnicode_FromString("nn.Parameter");
  auto type = &heap_type->ht_type;
  type->tp_name = "nn.Parameter";
  type->tp_basicsize = sizeof(PyTensorObject);

  type->tp_init = PyParameterObject_init;
  type->tp_new = PyParameterObject_new;

  Py_INCREF(PyTensorObject_Type);
  type->tp_base = PyTensorObject_Type;

  type->tp_flags |= Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE;

  if (PyType_Ready(type) < 0) { return NULL; }
  return type;
}

PyTypeObject* PyTensorObject_Type = MakeTensorType();
PyTypeObject* PyParameterObject_Type = MakeParameterType();

PyObject* PyTensor_New(const std::shared_ptr<Tensor>& data) {
  auto* self = (PyTensorObject*)PyTensorObject_Type->tp_alloc(PyTensorObject_Type, 0);
  if (self) { self->data = data; }
  return (PyObject*)self;
}

PyObject* PyParameter_New(const std::shared_ptr<Parameter>& data) {
  auto* self = (PyTensorObject*)PyTensorObject_Type->tp_alloc(PyParameterObject_Type, 0);
  if (self) { self->data = data; }
  return (PyObject*)self;
}

PyObject* PyParameter_New(const std::shared_ptr<Tensor>& data, bool requires_grad) {
  auto* self = (PyTensorObject*)PyTensorObject_Type->tp_alloc(PyParameterObject_Type, 0);
  if (self) { self->data = ASSERT_PTR(Parameter::MakeTensor(data, requires_grad)); }
  return (PyObject*)self;
}

#undef ASSERT
#undef ASSERT_PTR

ONEFLOW_API_PYBIND11_MODULE("", m) {
  if (PyType_Ready(PyTensorObject_Type) < 0) { return; }
  Py_INCREF(PyTensorObject_Type);
  if (PyModule_AddObject(m.ptr(), "Tensor", (PyObject*)PyTensorObject_Type) < 0) { return; }

  auto nn = m.def_submodule("nn");
  if (PyType_Ready(PyParameterObject_Type) < 0) { return; }
  Py_INCREF(PyParameterObject_Type);
  if (PyModule_AddObject(nn.ptr(), "Parameter", (PyObject*)PyParameterObject_Type) < 0) { return; }
}

}  // namespace one
}  // namespace oneflow
