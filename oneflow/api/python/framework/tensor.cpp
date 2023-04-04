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
#include "oneflow/api/python/framework/tensortype.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/functional/python_arg.h"
#include "oneflow/api/python/functional/functional_api.yaml.pybind.h"
#include "oneflow/api/python/functional/tensor_api.yaml.pybind.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/utils/tensor_utils.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/common/stride.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/placement_utils.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/tensor_index.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace py = pybind11;

namespace oneflow {
namespace one {

#define ASSERT(x) (x).GetOrThrow()
#define ASSERT_PTR(x) (x).GetPtrOrThrow()
#define PY_XINCREF(p) (({ Py_XINCREF(p); }), (p))

#if PY_VERSION_HEX < 0x03070000
#define PYGETSET_NAME(name) const_cast<char*>(name)
#else
#define PYGETSET_NAME(name) (name)
#endif

PyTypeObject* PyTensorObject_Type = NULL;
PyTypeObject* PyParameterObject_Type = NULL;

namespace {

template<typename T>
struct AllocType {};
#define DEFINE_ALLOC_TYPE(type)  \
  template<>                     \
  struct AllocType<type> {       \
    static PyTypeObject** value; \
  };                             \
  PyTypeObject** AllocType<type>::value = &Py##type##Object_Type

DEFINE_ALLOC_TYPE(Tensor);
DEFINE_ALLOC_TYPE(Parameter);
#undef DEFINE_ALLOC_TYPE

template<typename T>
PyObject* PyTensor_wrap(const std::shared_ptr<T>& data, PyTensorObject* bind_pyobj) {
  if (!data) { Py_RETURN_NONE; }
  PyObject* py_tensor = (PyObject*)data->pyobject();
  if (bind_pyobj == nullptr && py_tensor) {
    // Has been wrapped by python before
    if (data->owns_pyobj()) {
      // PyTensor are not alive in python side, so we flip back the ownership to PyTensor
      data->set_owns_pyobj(false);
      ((PyTensorObject*)py_tensor)->data = data;
      // NOTE: Needn't incref here, because the reference count of py_tensor is already increased
      return py_tensor;
    } else {
      // PyTensor is alive, so we directly incref it and return it
      Py_XINCREF(py_tensor);
      return py_tensor;
    }
  } else {
    // Has not been wrapped by python before, so we create a new PyTensor and give it the ownership
    if (bind_pyobj == nullptr) {
      bind_pyobj = (PyTensorObject*)PyTensorObject_Type->tp_alloc(*AllocType<T>::value, 0);
    }
    bind_pyobj->data = data;
    if (py_tensor) {
      // If it has bind pyobj, reset the shared_ptr in origin PyTensorObject
      ((PyTensorObject*)py_tensor)->data.reset();
    }
    bind_pyobj->data->set_pyobject_ptr(std::unique_ptr<void, void (*)(void*)>(
        bind_pyobj, [](void* ptr) { Py_DECREF((PyObject*)ptr); }));
    bind_pyobj->data->set_owns_pyobj(false);
    return (PyObject*)bind_pyobj;
  }
}

bool PyTensor_tryResurrect(PyObject* py_tensor) {
  auto* self = (PyTensorObject*)py_tensor;
  if (self->data) {
    // PyTensor holds the ownership, now we flip it back to C++ and resurrect python object
    // temporarily
    auto tensor = self->data;
    self->data.reset();
    tensor->set_owns_pyobj(true);
    Py_XINCREF(py_tensor);
    return true;
  }
  // Otherwise, PyTensor was already not alive in python side
  return false;
}

}  // namespace

static int PyTensorObject_init(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  auto* temp = functional::_legacy_tensor_ctor(NULL, args, kwargs);
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  PyTensor_wrap<Tensor>(PyTensor_Unpack(temp), (PyTensorObject*)self);
  return 0;
  END_HANDLE_ERRORS_RET(-1)
}

static void PyTensorObject_dealloc(PyObject* self) {
  if (PyTensor_tryResurrect(self)) { return; }

  // clear __dict__
  PyObject** dict_ptr = _PyObject_GetDictPtr(self);
  if (dict_ptr) { Py_CLEAR(*dict_ptr); }
  auto* type = Py_TYPE(self);
  type->tp_free(self);
  Py_DECREF(type);
}

static int PyParameterObject_init(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* data = NULL;
  int requires_grad = 1;
  static const char* keywords[3] = {"data", "requires_grad", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p:__init__", const_cast<char**>(keywords),
                                   &data, &requires_grad)) {
    return -1;
  }
  if (self) {
    PyTensor_wrap<Parameter>(
        ASSERT_PTR(Parameter::MakeTensor(PyTensor_Unpack(data), requires_grad)),
        (PyTensorObject*)self);
  }
  return 0;
  END_HANDLE_ERRORS_RET(-1)
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

static PyObject* PyTensorObject_subscript(PyObject* self, PyObject* item) {
  HANDLE_ERRORS
  const auto& p = PyTensor_Unpack(self);
  functional::PythonArg arg(item);
  return PyTensor_New(ASSERT_PTR(functional::TensorGetItem(p, arg.As<functional::TensorIndex>())));
  END_HANDLE_ERRORS
}

static PySequenceMethods PyTensorObject_as_sequence = {
    (lenfunc)PyTensorObject_length, NULL, /*sq_concat*/
    NULL,                                 /*sq_repeat*/
    (ssizeargfunc)PyTensorObject_getitem, /*sq_item*/
};

extern int PyTensorObject_setitem(PyObject*, PyObject*, PyObject*);
static PyMappingMethods PyTensorObject_as_mapping = {
    (lenfunc)PyTensorObject_length,
    (binaryfunc)PyTensorObject_subscript,
    (objobjargproc)PyTensorObject_setitem,
};

static PyObject* PyTensorObject_storage_offset(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  return functional::CastToPyObject(PyTensor_Unpack(self)->storage_offset());
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_stride(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  const auto& stride = ASSERT_PTR(PyTensor_Unpack(self)->stride());
  PyObject* tup = PyTuple_New(stride->size());
  for (int i = 0; i < stride->size(); ++i) {
    PyTuple_SetItem(tup, i, PyLong_FromUnsignedLong(stride->at(i)));
  }
  return tup;
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

static PyObject* PyTensorObject_contiguous_(PyObject* self, PyObject* unused) {
  // NOTE: inplace version of contiguous
  HANDLE_ERRORS
  return PyTensor_New(ASSERT_PTR(functional::InplaceToContiguous(PyTensor_Unpack(self))));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_pin_memory(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  return PyTensor_New(PyTensor_Unpack(self)->pin_memory());
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_is_pinned(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  return functional::CastToPyObject(CHECK_JUST(PyTensor_Unpack(self)->is_pinned()));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_offload(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  const auto& t = PyTensor_Unpack(self);
  CHECK_JUST(t->offload());
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_load(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  const auto& t = PyTensor_Unpack(self);
  CHECK_JUST(t->load());
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_is_offloaded(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  return functional::CastToPyObject(CHECK_JUST(PyTensor_Unpack(self)->is_offloaded()));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_is_floating_point(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  if (PyTensor_Unpack(self)->dtype()->is_floating_point()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_requires_grad_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  int requires_grad = 1;
  static const char* keywords[2] = {"requires_grad", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|p:requires_grad_", const_cast<char**>(keywords),
                                   &requires_grad)) {
    return NULL;
  }
  ASSERT(PyTensor_Unpack(self)->set_requires_grad(requires_grad));
  Py_XINCREF(self);
  return self;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_retain_grad(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  const auto& t = PyTensor_Unpack(self);
  CHECK_JUST(t->set_retain_grad(true));
  Py_RETURN_NONE;
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

static PyObject* PyTensorObject_zero_(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  ASSERT(EagerLocalTensorZeros(PyTensor_Unpack(self)));
  Py_XINCREF(self);
  return self;
  END_HANDLE_ERRORS
}

std::vector<Symbol<SbpParallel>> RawSbpBToP(Symbol<NdSbp> nd_sbp) {
  std::vector<Symbol<SbpParallel>> new_nd_sbp;
  for (const auto& old_sbp : nd_sbp->sbp_parallel()) {
    SbpParallel new_sbp = old_sbp;
    if (new_sbp.has_broadcast_parallel()) { new_sbp.mutable_partial_sum_parallel(); }
    new_nd_sbp.push_back(SymbolOf(new_sbp));
  }
  return new_nd_sbp;
}

static constexpr auto* SbpBToP = DECORATE(&RawSbpBToP, ThreadLocalCached);

static PyObject* PyTensorObject_zero_grad(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  int set_to_none = 0;
  static const char* keywords[2] = {"set_to_none", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|p:_zero_grad_", const_cast<char**>(keywords),
                                   &set_to_none)) {
    return NULL;
  }
  const auto& t = PyTensor_Unpack(self);
  const auto acc_grad = ASSERT_PTR(t->acc_grad());
  if (acc_grad) {
    if (set_to_none) {
      ASSERT(t->set_acc_grad(NULL));
    } else {
      ASSERT(EagerLocalTensorZeros(acc_grad));
      if (acc_grad->is_global() && acc_grad->is_eager()) {
        const auto local_tensor = ASSERT_PTR(functional::GlobalToLocal(acc_grad, false));
        const auto p = ASSERT_PTR(functional::LocalToGlobal(
            local_tensor, ASSERT(acc_grad->parallel_desc()), SbpBToP(ASSERT(acc_grad->nd_sbp())),
            *acc_grad->shape(), acc_grad->dtype(), false, false));
        ASSERT(acc_grad->set_data(p));
      }
    }
  }
  Py_XINCREF(self);
  return self;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_register_hook(PyObject* self, PyObject* hook) {
  HANDLE_ERRORS
  const auto& _hook = py::cast<AutogradMeta::Hook>(py::reinterpret_borrow<py::object>(hook));
  ASSERT(RegisterTensorHook(PyTensor_Unpack(self), _hook));
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject__register_post_grad_accumulation_hook(PyObject* self,
                                                                      PyObject* hook) {
  HANDLE_ERRORS
  const auto& _hook = py::cast<AutogradMeta::Hook>(py::reinterpret_borrow<py::object>(hook));
  ASSERT(RegisterTensorPostGradAccumulationHook(PyTensor_Unpack(self), _hook));
  Py_RETURN_NONE;
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
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_to_numpy(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  const auto& t = PyTensor_Unpack(self);
  DataType data_type = t->dtype()->data_type();
  switch (data_type) {
#define SWITCH_EAGER_TENSOR_TO_NUMPY(cpp_type, of_type) \
  case of_type: return ASSERT(EagerLocalTensorToNumpy<cpp_type>(self));
    OF_PP_FOR_EACH_TUPLE(SWITCH_EAGER_TENSOR_TO_NUMPY, POD_DATA_TYPE_SEQ COMPLEX_DATA_TYPE_SEQ)
    case DataType::kFloat16: return ASSERT(EagerLocalTensorToNumpy<float16>(self));
    default: {
      return PyErr_Format(PyExc_RuntimeError,
                          ("Invalid datatype " + DataType_Name(data_type)).data());
    }
  }
#undef SWITCH_EAGER_TENSOR_TO_NUMPY
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_item(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  const auto& t = PyTensor_Unpack(self);
  DataType data_type = t->dtype()->data_type();
  switch (data_type) {
#define CASE_SCALAR_TENSOR_TO_SCALAR(cpp_type, of_type) \
  case of_type: return ASSERT(EagerLocalTensorItem<cpp_type>(t));
    OF_PP_FOR_EACH_TUPLE(CASE_SCALAR_TENSOR_TO_SCALAR,
                         POD_AND_HALF_DATA_TYPE_SEQ COMPLEX_DATA_TYPE_SEQ);
    default: {
      return PyErr_Format(PyExc_RuntimeError,
                          ("Invalid datatype " + DataType_Name(data_type)).data());
    }
  }
#undef CASE_SCALAR_TENSOR_TO_SCALAR
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_type(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  const auto& tensor = PyTensor_Unpack(self);
  PyObject* tensor_type = NULL;
  int non_blocking = 0;
  static const char* keywords[3] = {"dtype", "non_blocking", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Op:type", const_cast<char**>(keywords),
                                   &tensor_type, &non_blocking)) {
    return NULL;
  }
  // TODO: support non_blocking=True
  if (non_blocking == 1) {
    return PyErr_Format(PyExc_TypeError, "non_blocking=True is not supported yet");
  }
  if (tensor_type == NULL) {
    tensor_type =
        PyTensorType_FromDTypeAndDeviceType(tensor->dtype(), ASSERT(tensor->device())->enum_type());
    return PyUnicode_FromString(((PyTensorType*)tensor_type)->name);
  }
  if (PyTensorMetaClass_CheckExact(tensor_type)) {
    Optional<std::string> device = "cpu";
    return PyTensor_New(ASSERT_PTR(functional::To(tensor, device, DType::Float(), /*copy=*/false)));
  }
  if (PyUnicode_Check(tensor_type)) {
    tensor_type = PyTensorType_FromString(PyUnicode_AsUTF8(tensor_type));
  }
  if (PyTensorType_Check(tensor_type)) {
    const auto& dtype = PyTensorType_UnpackDType(tensor_type);
    DeviceType device_type = PyTensorType_UnpackDevice(tensor_type);
    if (device_type == ASSERT(tensor->device())->enum_type()) {
      return PyTensor_New(ASSERT_PTR(functional::To(tensor, dtype, /*copy=*/false)));
    }
    Optional<std::string> device = ASSERT(DeviceTag4DeviceType(device_type));
    return PyTensor_New(ASSERT_PTR(functional::To(tensor, device, dtype, /*copy=*/false)));

  } else if (functional::PyDTypeCheck(tensor_type)) {
    return PyTensor_New(
        ASSERT_PTR(functional::To(tensor, functional::PyUnpackDType(tensor_type), /*copy=*/false)));
  }
  return PyErr_Format(PyExc_TypeError, "dtype must be a type, str, or dtype object");
  END_HANDLE_ERRORS
}

namespace {
void CopyFromNumpyArray(ep::Stream* stream,
                        const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object,
                        const NumPyArrayPtr& array_ptr) {
  SyncAutoMemcpy(stream, eager_blob_object->mut_dptr(), array_ptr.data(),
                 eager_blob_object->ByteSizeOfBlobBody(), eager_blob_object->mem_case(),
                 memory::MakeHostMemCase());
}

void CopyToNumpyArray(ep::Stream* stream,
                      const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object,
                      const NumPyArrayPtr& array_ptr) {
  SyncAutoMemcpy(stream, array_ptr.data(), eager_blob_object->dptr(),
                 eager_blob_object->ByteSizeOfBlobBody(), memory::MakeHostMemCase(),
                 eager_blob_object->mem_case());
}
}  // namespace
   //
static PyObject* PyTensorObject__copy_to_numpy(PyObject* self, PyObject* array) {
  HANDLE_ERRORS
  ASSERT(CopyBetweenLocalTensorAndNumpy(PyTensor_Unpack(self), array, CopyToNumpyArray, "const",
                                        /*block_host_until_done=*/true));
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}
static PyObject* PyTensorObject__copy_from_numpy(PyObject* self, PyObject* array) {
  HANDLE_ERRORS
  auto* copied = PyArray_NewCopy((PyArrayObject*)array, NPY_CORDER);
  ASSERT(CopyBetweenLocalTensorAndNumpy(PyTensor_Unpack(self), copied, CopyFromNumpyArray, "mut",
                                        /*block_host_until_done=*/false));
  Py_DECREF(copied);
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject__register_storage_delete_hook(PyObject* self, PyObject* hook) {
  HANDLE_ERRORS
  auto _hook = py::cast<std::function<void()>>(py::reinterpret_borrow<py::object>(hook));
  ASSERT(PyTensor_Unpack(self)->RegisterStorageDeleteHook(_hook));
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

static std::vector<PyMethodDef> concat_method_def(PyMethodDef methods[],
                                                  PyMethodDef extra_methods[]) {
  int len1 = 0;
  int len2 = 0;
  PyMethodDef* p1 = methods;
  PyMethodDef* p2 = extra_methods;
  while ((p1++)->ml_name != NULL) { len1++; }
  while ((p2++)->ml_name != NULL) { len2++; }
  std::vector<PyMethodDef> total_methods(len1 + len2 + 1);
  for (int i = 0; i < len1; i++) total_methods[i] = methods[i];
  for (int i = 0; i < len2; i++) total_methods[i + len1] = extra_methods[i];
  total_methods[len1 + len2] = {NULL};
  return total_methods;
}

static PyMethodDef PyTensorObject_methods[] = {
    {"storage_offset", PyTensorObject_storage_offset, METH_NOARGS, NULL},
    {"stride", PyTensorObject_stride, METH_NOARGS, NULL},
    {"is_contiguous", PyTensorObject_is_contiguous, METH_NOARGS, NULL},
    {"contiguous", PyTensorObject_contiguous, METH_NOARGS, NULL},
    {"contiguous_", PyTensorObject_contiguous_, METH_NOARGS, NULL},
    {"pin_memory", PyTensorObject_pin_memory, METH_NOARGS, NULL},
    {"is_pinned", PyTensorObject_is_pinned, METH_NOARGS, NULL},
    {"offload", PyTensorObject_offload, METH_NOARGS, NULL},
    {"load", PyTensorObject_load, METH_NOARGS, NULL},
    {"is_offloaded", PyTensorObject_is_offloaded, METH_NOARGS, NULL},
    {"is_floating_point", PyTensorObject_is_floating_point, METH_NOARGS, NULL},
    {"requires_grad_", (PyCFunction)PyTensorObject_requires_grad_, METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"retain_grad", PyTensorObject_retain_grad, METH_NOARGS, NULL},
    {"detach", PyTensorObject_detach, METH_NOARGS, NULL},
    {"clone", PyTensorObject_clone, METH_NOARGS, NULL},
    {"zero_", PyTensorObject_zero_, METH_NOARGS, NULL},
    {"_zero_grad_", (PyCFunction)PyTensorObject_zero_grad, METH_VARARGS | METH_KEYWORDS, NULL},
    {"register_hook", PyTensorObject_register_hook, METH_O, NULL},
    {"_register_post_grad_accumulation_hook", PyTensorObject__register_post_grad_accumulation_hook,
     METH_O, NULL},
    {"global_id", PyTensorObject_global_id, METH_NOARGS, NULL},
    {"check_meta_consistency", PyTensorObject_check_meta_consistency, METH_NOARGS, NULL},
    {"to_numpy", PyTensorObject_to_numpy, METH_NOARGS, NULL},
    {"item", PyTensorObject_item, METH_NOARGS, NULL},
    {"type", (PyCFunction)PyTensorObject_type, METH_VARARGS | METH_KEYWORDS, NULL},
    {"_copy_to_numpy", PyTensorObject__copy_to_numpy, METH_O, NULL},
    {"_copy_from_numpy", PyTensorObject__copy_from_numpy, METH_O, NULL},
    {"_register_storage_delete_hook", PyTensorObject__register_storage_delete_hook, METH_O, NULL},
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
  return functional::CastToPyObject(PyTensor_Unpack(self)->is_global());
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

// NOLINTNEXTLINE
static PyGetSetDef PyTensorObject_properties[] = {
    {PYGETSET_NAME("ndim"), (getter)PyTensorObject_ndim, NULL, NULL, NULL},
    {PYGETSET_NAME("shape"), (getter)PyTensorObject_shape, NULL, NULL, NULL},
    {PYGETSET_NAME("dtype"), (getter)PyTensorObject_dtype, NULL, NULL, NULL},
    {PYGETSET_NAME("is_cuda"), (getter)PyTensorObject_is_cuda, NULL, NULL, NULL},
    {PYGETSET_NAME("grad"), (getter)PyTensorObject_grad, (setter)PyTensorObject_set_grad, NULL,
     NULL},
    {PYGETSET_NAME("data"), (getter)PyTensorObject_data, (setter)PyTensorObject_set_data, NULL,
     NULL},
    {PYGETSET_NAME("grad_fn"), (getter)PyTensorObject_grad_fn, NULL, NULL, NULL},
    {PYGETSET_NAME("is_leaf"), (getter)PyTensorObject_is_leaf, NULL, NULL, NULL},
    {PYGETSET_NAME("requires_grad"), (getter)PyTensorObject_requires_grad,
     (setter)PyTensorObject_set_requires_grad, NULL, NULL},
    {PYGETSET_NAME("is_lazy"), (getter)PyTensorObject_is_lazy, NULL, NULL, NULL},
    {PYGETSET_NAME("is_eager"), (getter)PyTensorObject_is_eager, NULL, NULL, NULL},
    {PYGETSET_NAME("is_global"), (getter)PyTensorObject_is_global, NULL, NULL, NULL},
    {PYGETSET_NAME("is_local"), (getter)PyTensorObject_is_local, NULL, NULL, NULL},
    {PYGETSET_NAME("_tensor_buffer_shapes_and_dtypes"),
     (getter)PyTensorObject__tensor_buffer_shapes_and_dtypes, NULL, NULL, NULL},
    {PYGETSET_NAME("device"), (getter)PyTensorObject_device, NULL, NULL, NULL},
    {PYGETSET_NAME("placement"), (getter)PyTensorObject_placement, NULL, NULL, NULL},
    {PYGETSET_NAME("sbp"), (getter)PyTensorObject_sbp, NULL, NULL, NULL},
    {NULL}};

// create a Tensor instance
static PyObject* TensorMetaCls_call(PyObject* type, PyObject* args, PyObject* kwargs) {
  return PyType_Type.tp_call(type, args, kwargs);
}

static void TensorMetaCls_dealloc(PyObject* type) { PyType_Type.tp_dealloc(type); }

static PyHeapTypeObject* MakeTensorMetaclass() {
  PyObject* name = PyUnicode_FromString("_TensorMeta");

  auto* heap_type = (PyHeapTypeObject*)PyType_Type.tp_alloc(&PyType_Type, 0);
  heap_type->ht_name = name;
  heap_type->ht_qualname = PY_XINCREF(name);

  auto* type = &heap_type->ht_type;
  type->tp_name = "_TensorMeta";
  type->tp_base = PY_XINCREF(&PyType_Type);
  type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;

  type->tp_call = TensorMetaCls_call;
  type->tp_dealloc = TensorMetaCls_dealloc;

  if (PyType_Ready(type) < 0) { return NULL; }
  PyObject_SetAttrString((PyObject*)type, "__module__", PyUnicode_FromString("oneflow._C"));
  return heap_type;
}

extern PyNumberMethods PyTensorObject_as_number;
extern PyObject* PyTensorObject_richcompare(PyObject*, PyObject*, int);
extern PyMethodDef PyTensorObject_extra_methods[];

static PyHeapTypeObject* TensorMetaclass_Type = MakeTensorMetaclass();

static PyTypeObject* MakeTensorType() {
  PyObject* name = PyUnicode_FromString("Tensor");

  auto* metaclass = &TensorMetaclass_Type->ht_type;
  auto* heap_type = (PyHeapTypeObject*)metaclass->tp_alloc(metaclass, 0);
  if (!heap_type) { return NULL; }
  heap_type->ht_name = name;
  heap_type->ht_qualname = PY_XINCREF(name);
  auto* type = &heap_type->ht_type;
  type->tp_name = "Tensor";
  type->tp_basicsize = sizeof(PyTensorObject);

  type->tp_init = PyTensorObject_init;
  type->tp_dealloc = PyTensorObject_dealloc;
  type->tp_getset = PyTensorObject_properties;

  static std::vector<PyMethodDef> total_methods =
      concat_method_def(PyTensorObject_methods, PyTensorObject_extra_methods);
  type->tp_methods = total_methods.data();

  type->tp_as_number = &PyTensorObject_as_number;
  type->tp_as_sequence = &PyTensorObject_as_sequence;
  type->tp_as_mapping = &PyTensorObject_as_mapping;
  type->tp_richcompare = PyTensorObject_richcompare;
  type->tp_hash = (hashfunc)_Py_HashPointer;

  type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;

  if (PyType_Ready(type) < 0) { return NULL; }
  PyObject_SetAttrString((PyObject*)type, "__module__", PyUnicode_FromString("oneflow"));
  return type;
}

static PyTypeObject* MakeParameterType() {
  PyObject* name = PyUnicode_FromString("Parameter");

  auto* metaclass = &TensorMetaclass_Type->ht_type;
  auto* heap_type = (PyHeapTypeObject*)metaclass->tp_alloc(metaclass, 0);
  if (!heap_type) { return NULL; }
  heap_type->ht_name = name;
  heap_type->ht_qualname = PY_XINCREF(name);
  auto* type = &heap_type->ht_type;
  type->tp_name = "Parameter";
  type->tp_basicsize = sizeof(PyTensorObject);

  type->tp_init = PyParameterObject_init;

  type->tp_base = PY_XINCREF(PyTensorObject_Type);

  type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;

  if (PyType_Ready(type) < 0) { return NULL; }
  PyObject_SetAttrString((PyObject*)type, "__module__", PyUnicode_FromString("oneflow.nn"));
  return type;
}

PyObject* PyTensor_New(const std::shared_ptr<Tensor>& data) {
  return PyTensor_wrap<Tensor>(data, /*bind_pyobj=*/nullptr);
}

PyObject* PyParameter_New(const std::shared_ptr<Parameter>& data) {
  return PyTensor_wrap<Parameter>(data, /*bind_pyobj=*/nullptr);
}

PyObject* PyParameter_New(const std::shared_ptr<Tensor>& data, bool requires_grad) {
  if (!data) { Py_RETURN_NONE; }
  return PyTensor_wrap<Parameter>(ASSERT_PTR(Parameter::MakeTensor(data, requires_grad)),
                                  /*bind_pyobj=*/nullptr);
}

}  // namespace one
}  // namespace oneflow

#undef ASSERT
#undef ASSERT_PTR

using namespace oneflow::one;

ONEFLOW_API_PYBIND11_MODULE("", m) {
  PyTensorObject_Type = MakeTensorType();
  PyParameterObject_Type = MakeParameterType();
  if (PyTensorObject_Type
      && PyModule_AddObject(m.ptr(), "Tensor", (PyObject*)PyTensorObject_Type) < 0) {
    return;
  }
  auto nn = m.def_submodule("nn");
  if (PyParameterObject_Type
      && PyModule_AddObject(nn.ptr(), "Parameter", (PyObject*)PyParameterObject_Type) < 0) {
    return;
  }
}
