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
#include <object.h>
#include <tupleobject.h>
#include "oneflow/api/python/exception/exception.h"
#include "oneflow/api/python/framework/size.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/functional/functional_api.yaml.pybind.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {
namespace one {

#define ASSERT(x) (x).GetOrThrow()
#define ASSERT_PTR(x) (x).GetPtrOrThrow()

using functional::PyObjectPtr;

#define NB_UNARY_FUNC(func_name, bind_func)                  \
  static PyObject* func_name(PyObject* self) {               \
    HANDLE_ERRORS                                            \
    PyObjectPtr tuple(PyTuple_Pack(1, self));                \
    auto* result = bind_func(NULL, tuple.get(), NULL);       \
    if (PyErr_Occurred()) { throw py::error_already_set(); } \
    return result;                                           \
    END_HANDLE_ERRORS                                        \
  }

#define NB_BINARY_FUNC(func_name, bind_func)                 \
  static PyObject* func_name(PyObject* a, PyObject* b) {     \
    HANDLE_ERRORS                                            \
    PyObjectPtr tuple(PyTuple_Pack(2, a, b));                \
    auto* result = bind_func(NULL, tuple.get(), NULL);       \
    if (PyErr_Occurred()) { throw py::error_already_set(); } \
    return result;                                           \
    END_HANDLE_ERRORS                                        \
  }

NB_UNARY_FUNC(PyTensorObject_nb_absolute, functional::abs);
NB_UNARY_FUNC(PyTensorObject_nb_negative, functional::negative);
// TODO: not implemented yet
// NB_UNARY_FUNC(PyTensorObject_positive, functional::positive);

NB_BINARY_FUNC(PyTensorObject_nb_add, functional::add);
NB_BINARY_FUNC(PyTensorObject_nb_sub, functional::sub);
NB_BINARY_FUNC(PyTensorObject_nb_mul, functional::mul);
NB_BINARY_FUNC(PyTensorObject_nb_fmod, functional::fmod);
NB_BINARY_FUNC(PyTensorObject_nb_div, functional::div);
NB_BINARY_FUNC(PyTensorObject_nb_and, functional::logical_and);
NB_BINARY_FUNC(PyTensorObject_nb_xor, functional::logical_xor);
NB_BINARY_FUNC(PyTensorObject_nb_or, functional::logical_or);
NB_BINARY_FUNC(PyTensorObject_nb_floor_div, functional::floor_divide);
NB_BINARY_FUNC(PyTensorObject_nb_true_div, functional::div);
NB_BINARY_FUNC(PyTensorObject_nb_matrix_multiply, functional::matmul);

PyObject* PyTensorObject_nb_pow(PyObject* a, PyObject* b, PyObject* unsed) {
  HANDLE_ERRORS
  PyObjectPtr tuple(PyTuple_Pack(2, a, b));
  auto* result = functional::pow(NULL, tuple.get(), NULL);
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_nb_invert(PyObject* self) {
  HANDLE_ERRORS
  CHECK_OR_THROW(PyTensor_Unpack(self)->dtype()->data_type() == DataType::kBool)
      << "~ (operator.invert) is only implemented on integer and Boolean-type tensors";
  PyObjectPtr tuple(PyTuple_Pack(1, self));
  auto* result = functional::logical_not(NULL, tuple.get(), NULL);
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

#define NB_INPLACE_BINARY_FUNC(func_name, bind_func)                           \
  static PyObject* func_name(PyObject* a, PyObject* b) {                       \
    HANDLE_ERRORS                                                              \
    PyObjectPtr tuple(PyTuple_Pack(2, a, b));                                  \
    PyObjectPtr dict(PyDict_New());                                            \
    CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "inplace", Py_True) > -1); \
    auto* result = bind_func(NULL, tuple.get(), dict.get());                   \
    if (PyErr_Occurred()) { throw py::error_already_set(); }                   \
    return result;                                                             \
    END_HANDLE_ERRORS                                                          \
  }

// inplace operators
NB_INPLACE_BINARY_FUNC(PyTensorObject_nb_inplace_add, functional::add);
NB_INPLACE_BINARY_FUNC(PyTensorObject_nb_inplace_sub, functional::sub);
// The interface of inplace mul not mul(*, inplace=True) but mul_
NB_BINARY_FUNC(PyTensorObject_nb_inplace_mul, functional::mul_);
NB_BINARY_FUNC(PyTensorObject_nb_inplace_true_div, functional::div_);

PyObject* PyTensorObject_inplace_pow(PyObject* a, PyObject* b, PyObject* unsed) {
  HANDLE_ERRORS
  PyObjectPtr tuple(PyTuple_Pack(2, a, b));
  PyObjectPtr dict(PyDict_New());
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "inplace", Py_True) > -1);
  auto* result = functional::pow(NULL, tuple.get(), NULL);
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

PyNumberMethods PyTensorObject_as_number = {
    PyTensorObject_nb_add,       // nb_add
    PyTensorObject_nb_sub,       // nb_subtract
    PyTensorObject_nb_mul,       // nb_multiply
    PyTensorObject_nb_fmod,      // nb_remainder
    NULL,                        // nb_divmod
    PyTensorObject_nb_pow,       // nb_power
    PyTensorObject_nb_negative,  // nb_negative
    NULL,                        // nb_positive
    PyTensorObject_nb_absolute,  // nb_absolute
    NULL,                        // nb_bool
    PyTensorObject_nb_invert,    // nb_invert
    NULL,                        // nb_lshift
    NULL,                        // nb_rshift
    PyTensorObject_nb_and,       // nb_and
    PyTensorObject_nb_xor,       // nb_xor
    PyTensorObject_nb_or,        // nb_or
    NULL,                        // nb_int
    NULL,                        // nb_reserved
    NULL,                        // nb_float

    PyTensorObject_nb_inplace_add,  // nb_inplace_add
    PyTensorObject_nb_inplace_sub,  // nb_inplace_sub
    PyTensorObject_nb_inplace_mul,  // nb_inplace_mul
    NULL,                           // nb_inplace_remainder
    NULL,                           // nb_inplace_pow
    NULL,                           // nb_inplace_lshift
    NULL,                           // nb_inplace_rshift
    NULL,                           // nb_inplace_and
    NULL,                           // nb_inplace_xor
    NULL,                           // nb_inplace_or

    PyTensorObject_nb_floor_div,         // nb_floor_div
    PyTensorObject_nb_true_div,          // nb_true_div
    NULL,                                // nb_inplace_floor_div
    PyTensorObject_nb_inplace_true_div,  // nb_inplace_true_div

    NULL,                               // nb_index
    PyTensorObject_nb_matrix_multiply,  // nb_matrix_multiply
    NULL,                               // not implemented yet nb_inplace_matrix_multiply

};

// extra methods
#define UNARY_METHOD(func_name, bind_func)                             \
  static PyObject* func_name(PyObject* self, PyObject* unused) {       \
    HANDLE_ERRORS                                                      \
    std::cout << "cpython" << std::endl;                               \
    return PyTensor_New(ASSERT_PTR(bind_func(PyTensor_Unpack(self)))); \
    END_HANDLE_ERRORS                                                  \
  }

UNARY_METHOD(PyTensorObject_abs, functional::Abs);
UNARY_METHOD(PyTensorObject_exp, functional::Exp);
UNARY_METHOD(PyTensorObject_floor, functional::Floor);
UNARY_METHOD(PyTensorObject_floor_, functional::Floor_);
UNARY_METHOD(PyTensorObject_sign, functional::Sign);
UNARY_METHOD(PyTensorObject_gelu, functional::Gelu);
UNARY_METHOD(PyTensorObject_mish, functional::Mish);
UNARY_METHOD(PyTensorObject_negative, functional::Negative);
UNARY_METHOD(PyTensorObject_sigmoid, functional::Sigmoid);
UNARY_METHOD(PyTensorObject_silu, functional::Silu);
UNARY_METHOD(PyTensorObject_selu, functional::Selu);
UNARY_METHOD(PyTensorObject_softsign, functional::SoftSign);
UNARY_METHOD(PyTensorObject_log1p, functional::Log1p);
UNARY_METHOD(PyTensorObject_log2, functional::Log2);
UNARY_METHOD(PyTensorObject_reciprocal, functional::Reciprocal);
UNARY_METHOD(PyTensorObject_ceil, functional::Ceil);
UNARY_METHOD(PyTensorObject_erf, functional::Erf);
UNARY_METHOD(PyTensorObject_erfc, functional::Erfc);
UNARY_METHOD(PyTensorObject_erfinv, functional::Erfinv);
UNARY_METHOD(PyTensorObject_erfinv_, functional::ErfinvInplace);
UNARY_METHOD(PyTensorObject_expm1, functional::Expm1);
UNARY_METHOD(PyTensorObject_log, functional::Log);
UNARY_METHOD(PyTensorObject_rsqrt, functional::Rsqrt);
UNARY_METHOD(PyTensorObject_sqrt, functional::Sqrt);
UNARY_METHOD(PyTensorObject_square, functional::Square);
UNARY_METHOD(PyTensorObject_round, functional::Round);
UNARY_METHOD(PyTensorObject_t, functional::TransposeAllDimFunction);
UNARY_METHOD(PyTensorObject_isnan, functional::IsNan);
UNARY_METHOD(PyTensorObject_isinf, functional::IsInf);
UNARY_METHOD(PyTensorObject_sin, functional::Sin);
UNARY_METHOD(PyTensorObject_sin_, functional::Sin_);
UNARY_METHOD(PyTensorObject_asin, functional::Asin);
UNARY_METHOD(PyTensorObject_cos, functional::Cos);
UNARY_METHOD(PyTensorObject_acos, functional::Acos);
UNARY_METHOD(PyTensorObject_tan, functional::Tan);
UNARY_METHOD(PyTensorObject_atan, functional::Atan);
UNARY_METHOD(PyTensorObject_sinh, functional::Sinh);
UNARY_METHOD(PyTensorObject_asinh, functional::Asinh);
UNARY_METHOD(PyTensorObject_cosh, functional::Cosh);
UNARY_METHOD(PyTensorObject_acosh, functional::Acosh);
UNARY_METHOD(PyTensorObject_tanh, functional::Tanh);
UNARY_METHOD(PyTensorObject_atanh, functional::Atanh);
UNARY_METHOD(PyTensorObject_logical_not, functional::LogicalNot);

#define BINARY_METHOD(func_name, bind_func, name)                                           \
  static PyObject* func_name(PyObject* self, PyObject* args, PyObject* kwargs) {            \
    HANDLE_ERRORS                                                                           \
    std::cout << "cpython" << std::endl;                                                    \
    PyObject* other = NULL;                                                                 \
    static const char* keywords[2] = {"other", NULL};                                       \
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:" name, const_cast<char**>(keywords), \
                                     &other)) {                                             \
      return NULL;                                                                          \
    }                                                                                       \
    PyObjectPtr tuple(PyTuple_Pack(2, self, other));                                        \
    auto* result = bind_func(NULL, tuple.get(), NULL);                                      \
    if (PyErr_Occurred()) { throw py::error_already_set(); }                                \
    return result;                                                                          \
    END_HANDLE_ERRORS                                                                       \
  }

BINARY_METHOD(PyTensorObject_floor_divide, functional::floor_divide, "floor_divide");
BINARY_METHOD(PyTensorObject_atan2, functional::atan2, "atan2");
BINARY_METHOD(PyTensorObject_gt, functional::greater, "gt");
BINARY_METHOD(PyTensorObject_ge, functional::greater_equal, "ge");
BINARY_METHOD(PyTensorObject_div, functional::div, "div");
BINARY_METHOD(PyTensorObject_div_, functional::div_, "div_");
BINARY_METHOD(PyTensorObject_mul, functional::mul, "mul");
BINARY_METHOD(PyTensorObject_mul_, functional::mul_, "mul_");
BINARY_METHOD(PyTensorObject_sub, functional::sub, "sub");
// TODO: not implemented yet
// BINARY_METHOD(PyTensorObject_sub, functional::sub, "sub_");
BINARY_METHOD(PyTensorObject_fmod, functional::fmod, "fmod");
BINARY_METHOD(PyTensorObject_matmul, functional::matmul, "matmul");
BINARY_METHOD(PyTensorObject_logical_and, functional::logical_and, "logical_and");
BINARY_METHOD(PyTensorObject_logical_or, functional::logical_or, "logical_or");
BINARY_METHOD(PyTensorObject_logical_xor, functional::logical_xor, "logical_xor");
BINARY_METHOD(PyTensorObject_bmm, functional::batch_matmul, "bmm");
BINARY_METHOD(PyTensorObject_ne, functional::not_equal, "ne");
BINARY_METHOD(PyTensorObject_lt, functional::less, "lt");
BINARY_METHOD(PyTensorObject_le, functional::less_equal, "le");

static PyObject* PyTensorObject_byte(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  return PyTensor_New(ASSERT_PTR(functional::To(PyTensor_Unpack(self), DType::UInt8(), false)));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_dim(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  return functional::CastToPyObject(PyTensor_Unpack(self)->ndim());
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_nelement(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  return functional::CastToPyObject(PyTensor_Unpack(self)->nelement());
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_element_size(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  return functional::CastToPyObject(PyTensor_Unpack(self)->dtype()->bytes());
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_get_device(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  DeviceType device_type = ASSERT(PyTensor_Unpack(self)->device())->enum_type();
  CHECK_OR_THROW(device_type == DeviceType::kCUDA)
      << "get_device is only available for GPU tensor.";
  return functional::CastToPyObject(ASSERT(PyTensor_Unpack(self)->device())->device_id());
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_size(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython size" << std::endl;
  PyObject* idx = Py_None;
  static const char* keywords[2] = {"idx", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:size", const_cast<char**>(keywords), &idx)) {
    return NULL;
  }
  auto shape = PyTensor_Unpack(self)->shape();
  PyObject* shape_object = TensorSize_NewFromShape(*shape);
  if (idx == NULL || idx == Py_None) return shape_object;
  return shape_object->ob_type->tp_as_mapping->mp_subscript(shape_object, idx);
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_argmax(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython argmax" << std::endl;
  PyObject* dim = Py_None;
  PyObject* keepdim = Py_None;
  static const char* keywords[3] = {"dim", "keepdim", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO:argmax", const_cast<char**>(keywords), &dim,
                                   &keepdim)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(1, self));
  PyObjectPtr dict(PyDict_New());
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "dim", dim) > -1);
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "keepdim", keepdim) > -1);
  PyObject* result = functional::argmax(NULL, tuple.get(), dict.get());
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_argmin(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython argmax" << std::endl;
  PyObject* dim = Py_None;
  PyObject* keepdim = Py_None;
  static const char* keywords[3] = {"dim", "keepdim", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO:argmin", const_cast<char**>(keywords), &dim,
                                   &keepdim)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(1, self));
  PyObjectPtr dict(PyDict_New());
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "dim", dim) > -1);
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "keepdim", keepdim) > -1);
  PyObject* result = functional::argmin(NULL, tuple.get(), dict.get());
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_amin(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython amin" << std::endl;
  PyObject* dim = Py_None;
  PyObject* keepdim = Py_None;
  static const char* keywords[3] = {"dim", "keepdim", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO:amin", const_cast<char**>(keywords), &dim,
                                   &keepdim)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(1, self));
  PyObjectPtr dict(PyDict_New());
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "dim", dim) > -1);
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "keepdim", keepdim) > -1);
  PyObject* result = functional::amin(NULL, tuple.get(), dict.get());
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_cast(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython" << std::endl;
  PyObject* dtype = NULL;
  static const char* keywords[2] = {"dtype", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:cast", const_cast<char**>(keywords), &dtype)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(2, self, dtype));
  PyObject* result = functional::cast(NULL, tuple.get(), NULL);
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_diag(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython" << std::endl;
  int32_t diagonal = 0;
  static const char* keywords[2] = {"diagonal", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i:diag", const_cast<char**>(keywords),
                                   &diagonal)) {
    return NULL;
  }
  return PyTensor_New(ASSERT_PTR(functional::Diag(PyTensor_Unpack(self), diagonal)));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_diagonal(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython" << std::endl;
  int32_t offset = 0;
  int32_t dim1 = 0;
  int32_t dim2 = 1;
  static const char* keywords[4] = {"offset", "dim1", "dim2", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iii:diagonal", const_cast<char**>(keywords),
                                   &offset, &dim1, &dim2)) {
    return NULL;
  }
  return PyTensor_New(ASSERT_PTR(functional::Diagonal(PyTensor_Unpack(self), offset, dim1, dim2)));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_addcmul(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython" << std::endl;
  PyObject* tensor1 = NULL;
  PyObject* tensor2 = NULL;
  PyObject* value = PyFloat_FromDouble(1.0);
  static const char* keywords[4] = {"tensor1", "tensor2", "value", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|$O:addcmul", const_cast<char**>(keywords),
                                   &tensor1, &tensor2, &value)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(3, self, tensor1, tensor2));
  PyObjectPtr dict(PyDict_New());
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "value", value) > -1);
  PyObject* result = functional::addcmul(NULL, tuple.get(), dict.get());
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_addcmul_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython" << std::endl;
  PyObject* tensor1 = NULL;
  PyObject* tensor2 = NULL;
  PyObject* value = PyFloat_FromDouble(1.0);
  static const char* keywords[4] = {"tensor1", "tensor2", "value", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|$O:addcmul_", const_cast<char**>(keywords),
                                   &tensor1, &tensor2, &value)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(3, self, tensor1, tensor2));
  PyObjectPtr dict(PyDict_New());
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "value", value) > -1);
  PyObject* result = functional::addcmul_(NULL, tuple.get(), dict.get());
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_sub_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython" << std::endl;
  PyObject* other = NULL;
  static const char* keywords[2] = {"other", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:sub_", const_cast<char**>(keywords), &other)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(2, self, other));
  PyObjectPtr dict(PyDict_New());
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "inplace", Py_True) > -1);
  PyObject* result = functional::sub(NULL, tuple.get(), dict.get());
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_clamp(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython" << std::endl;
  PyObject* min = Py_None;
  PyObject* max = Py_None;
  static const char* keywords[3] = {"min", "max", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO:clamp", const_cast<char**>(keywords), &min,
                                   &max)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(1, self));
  PyObjectPtr dict(PyDict_New());
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "min", min) > -1);
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "max", max) > -1);
  PyObject* result = functional::clamp(NULL, tuple.get(), dict.get());
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_clamp_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython" << std::endl;
  PyObject* min = Py_None;
  PyObject* max = Py_None;
  static const char* keywords[3] = {"min", "max", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO:clamp_", const_cast<char**>(keywords), &min,
                                   &max)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(1, self));
  PyObjectPtr dict(PyDict_New());
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "min", min) > -1);
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "max", max) > -1);
  PyObject* result = functional::clamp_(NULL, tuple.get(), dict.get());
  if (PyErr_Occurred()) { throw py::error_already_set(); }

  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_clip(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython" << std::endl;
  PyObject* min = Py_None;
  PyObject* max = Py_None;
  static const char* keywords[3] = {"min", "max", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO:clip", const_cast<char**>(keywords), &min,
                                   &max)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(1, self));
  PyObjectPtr dict(PyDict_New());
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "min", min) > -1);
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "max", max) > -1);
  return functional::clip(NULL, tuple.get(), dict.get());
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_clip_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython" << std::endl;
  PyObject* min = Py_None;
  PyObject* max = Py_None;
  static const char* keywords[3] = {"min", "max", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO:clip_", const_cast<char**>(keywords), &min,
                                   &max)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(1, self));
  PyObjectPtr dict(PyDict_New());
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "min", min) > -1);
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "max", max) > -1);
  PyObject* result = functional::clip_(NULL, tuple.get(), dict.get());
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_cpu(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  std::cout << "cpython" << std::endl;
  Optional<std::string> device = "cpu";
  return PyTensor_New(ASSERT_PTR(functional::To(PyTensor_Unpack(self), device, NullOpt, false)));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_cuda(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython" << std::endl;
  PyObject* device_obj = Py_None;
  static const char* keywords[2] = {"device", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:cuda", const_cast<char**>(keywords),
                                   &device_obj)) {
    return NULL;
  }
  PyObjectPtr dict(PyDict_New());
  if (device_obj == Py_None) {
    CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "device", PyUnicode_FromString("cuda")) > -1);
  } else if (PyLong_Check(device_obj)) {
    std::string device = "cuda:" + std::to_string(PyLong_AsLong(device_obj));
    CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "device", PyUnicode_FromString(device.data()))
                   > -1);
  } else {
    CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "device", device_obj) > -1);
  }
  PyObjectPtr tuple(PyTuple_Pack(1, self));
  PyObject* result = functional::to(NULL, tuple.get(), dict.get());
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_in_top_k(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* predictions = NULL;
  PyObject* k = NULL;
  static const char* keywords[3] = {"predictions", "k", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO:in_top_k", const_cast<char**>(keywords),
                                   &predictions, &k)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(3, self, predictions, k));
  PyObject* result = functional::in_top_k(NULL, tuple.get(), NULL);
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_index_select(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* dim = NULL;
  PyObject* index = NULL;
  static const char* keywords[3] = {"dim", "index", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO:index_select", const_cast<char**>(keywords),
                                   &dim, &index)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(3, self, dim, index));
  PyObject* result = functional::index_select(NULL, tuple.get(), NULL);
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_minimum(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* y = NULL;
  static const char* keywords[2] = {"y", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:minimum", const_cast<char**>(keywords), &y)) {
    return NULL;
  }
  CHECK_OR_THROW(PyTensor_Check(y))
      << Error::TypeError() << "minimum(): argument 'other' must be tensor, not "
      << functional::PyStringAsString(PyObject_Str((PyObject*)Py_TYPE(y)));
  return PyTensor_New(ASSERT_PTR(functional::Minimum(PyTensor_Unpack(self), PyTensor_Unpack(y))));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_maximum(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* y = NULL;
  static const char* keywords[2] = {"y", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:maximum", const_cast<char**>(keywords), &y)) {
    return NULL;
  }
  CHECK_OR_THROW(PyTensor_Check(y))
      << Error::TypeError() << "maximum(): argument 'other' must be tensor, not "
      << functional::PyStringAsString(PyObject_Str((PyObject*)Py_TYPE(y)));
  return PyTensor_New(ASSERT_PTR(functional::Maximum(PyTensor_Unpack(self), PyTensor_Unpack(y))));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_pow(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* b = NULL;
  static const char* keywords[2] = {"b", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:pow", const_cast<char**>(keywords), &b)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(2, self, b));
  PyObject* result = functional::pow(NULL, tuple.get(), NULL);
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_var(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* dim = Py_None;
  PyObject* unbiased = Py_True;
  PyObject* keepdim = Py_False;
  static const char* keywords[4] = {"dim", "unbiased", "keepdim", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOO:var", const_cast<char**>(keywords), &dim,
                                   &unbiased, &keepdim)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(1, self));
  PyObjectPtr dict(PyDict_New());
  PyDict_SetItemString(dict.get(), "dim", dim);
  PyDict_SetItemString(dict.get(), "unbiased", unbiased);
  PyDict_SetItemString(dict.get(), "keepdim", keepdim);
  PyObject* result = functional::var(NULL, tuple.get(), dict.get());
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_std(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* dim = Py_None;
  PyObject* unbiased = Py_True;
  PyObject* keepdim = Py_False;
  static const char* keywords[4] = {"dim", "unbiased", "keepdim", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOO:std", const_cast<char**>(keywords), &dim,
                                   &unbiased, &keepdim)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(1, self));
  PyObjectPtr dict(PyDict_New());
  PyDict_SetItemString(dict.get(), "dim", dim);
  PyDict_SetItemString(dict.get(), "unbiased", unbiased);
  PyDict_SetItemString(dict.get(), "keepdim", keepdim);
  PyObject* result = functional::std(NULL, tuple.get(), dict.get());
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_tril(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython" << std::endl;
  PyObject* diagonal = PyLong_FromLong(0);
  static const char* keywords[4] = {"diagonal", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:tril", const_cast<char**>(keywords),
                                   &diagonal)) {
    return NULL;
  }
  CHECK_OR_THROW(PyLong_Check(diagonal))
      << Error::TypeError() << "tril(): argument 'diagonal' must be int64, not "
      << functional::PyStringAsString(PyObject_Str((PyObject*)Py_TYPE(diagonal)));
  return PyTensor_New(ASSERT_PTR(functional::Tril(PyTensor_Unpack(self), PyLong_AsLong(diagonal))));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_triu(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython" << std::endl;
  PyObject* diagonal = PyLong_FromLong(0);
  static const char* keywords[4] = {"diagonal", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:triu", const_cast<char**>(keywords),
                                   &diagonal)) {
    return NULL;
  }
  CHECK_OR_THROW(PyLong_Check(diagonal))
      << Error::TypeError() << "triu(): argument 'diagonal' must be int64, not "
      << functional::PyStringAsString(PyObject_Str((PyObject*)Py_TYPE(diagonal)));
  return PyTensor_New(ASSERT_PTR(functional::Triu(PyTensor_Unpack(self), PyLong_AsLong(diagonal))));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_norm(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* p = Py_None;
  PyObject* dim = Py_None;
  PyObject* keepdim = Py_False;
  PyObject* dtype = Py_None;
  static const char* keywords[5] = {"p", "dim", "keepdim", "dtype", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOOO:norm", const_cast<char**>(keywords), &p,
                                   &dim, &keepdim, &dtype)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(4, self, p, dim, keepdim));
  PyObjectPtr dict(PyDict_New());
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "dtype", dtype) > -1);
  PyObject* result = functional::norm(NULL, tuple.get(), dict.get());
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_reshape(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* shape = args;
  if (PyTuple_Size(args) == 1) {
    PyObject* item = PyTuple_GetItem(args, 0);
    if (!PyLong_Check(item)) { shape = item; }
  }

  PyObjectPtr _args = PyObjectPtr(PyTuple_Pack(2, self, shape));
  PyObject* result = functional::reshape(NULL, _args.get(), kwargs);
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_reshape_as(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  auto tensor = PyTensor_Unpack(self);
  PyObject* other = NULL;
  static const char* keywords[2] = {"other", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|:reshape_as", const_cast<char**>(keywords),
                                   &other)) {
    return NULL;
  }
  return PyTensor_New(ASSERT_PTR(functional::Reshape(tensor, *PyTensor_Unpack(other)->shape())));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_relu(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  auto tensor = PyTensor_Unpack(self);
  return PyTensor_New(ASSERT_PTR(functional::Relu(tensor, false)));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_relu_(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  auto tensor = PyTensor_Unpack(self);
  return PyTensor_New(ASSERT_PTR(functional::Relu(tensor, true)));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_softmax(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  auto tensor = PyTensor_Unpack(self);
  PyObject* dim = Py_None;
  static const char* keywords[2] = {"dim", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:softmax", const_cast<char**>(keywords),
                                   &dim)) {
    return NULL;
  }
  if (dim == Py_None) return PyTensor_New(ASSERT_PTR(functional::Softmax(tensor, NullOpt)));
  CHECK_OR_THROW(PyLong_Check(dim))
      << Error::TypeError() << "softmax(): argument 'dim' must be int64, not "
      << functional::PyStringAsString(PyObject_Str((PyObject*)Py_TYPE(dim)));
  return PyTensor_New(ASSERT_PTR(functional::Softmax(tensor, PyLong_AsLong(dim))));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_log_softmax(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  auto tensor = PyTensor_Unpack(self);
  PyObject* dim = Py_None;
  static const char* keywords[2] = {"dim", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:log_softmax", const_cast<char**>(keywords),
                                   &dim)) {
    return NULL;
  }
  if (dim == Py_None) return PyTensor_New(ASSERT_PTR(functional::LogSoftmax(tensor, NullOpt)));
  CHECK_OR_THROW(PyLong_Check(dim))
      << Error::TypeError() << "log_softmax(): argument 'dim' must be int64, not "
      << functional::PyStringAsString(PyObject_Str((PyObject*)Py_TYPE(dim)));
  return PyTensor_New(ASSERT_PTR(functional::LogSoftmax(tensor, PyLong_AsLong(dim))));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_roll(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* shifts = NULL;
  PyObject* dims = Py_None;
  static const char* keywords[3] = {"shifts", "dims", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O:roll", const_cast<char**>(keywords), &shifts,
                                   &dims)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(3, self, shifts, dims));
  PyObject* result = functional::roll(NULL, tuple.get(), NULL);
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_chunk(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* chunks = Py_None;
  PyObject* dim = Py_None;
  static const char* keywords[3] = {"chunks", "dim", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO:chunk", const_cast<char**>(keywords), &chunks,
                                   &dim)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(3, self, chunks, dim));
  PyObject* result = functional::chunk(NULL, tuple.get(), NULL);
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

#define DATATYPE_FUNC(func_name, dtype)                                    \
  static PyObject* func_name(PyObject* self, PyObject* unused) {           \
    HANDLE_ERRORS                                                          \
    auto tensor = PyTensor_Unpack(self);                                   \
    return PyTensor_New(ASSERT_PTR(functional::To(tensor, dtype, false))); \
    END_HANDLE_ERRORS                                                      \
  }

DATATYPE_FUNC(PyTensorObject_int, DType::Int32());
DATATYPE_FUNC(PyTensorObject_long, DType::Int64());
DATATYPE_FUNC(PyTensorObject_float, DType::Float());
DATATYPE_FUNC(PyTensorObject_double, DType::Double());

static PyObject* PyTensorObject_view(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* shape = args;
  if (PyTuple_Size(args) == 1) {
    PyObject* item = PyTuple_GetItem(args, 0);
    if (!PyLong_Check(item)) { shape = item; }
  }

  PyObjectPtr _args = PyObjectPtr(PyTuple_Pack(2, self, shape));
  PyObject* result = functional::view(NULL, _args.get(), kwargs);
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_view_as(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  auto tensor = PyTensor_Unpack(self);
  PyObject* other = NULL;
  static const char* keywords[2] = {"other", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|:view_as", const_cast<char**>(keywords),
                                   &other)) {
    return NULL;
  }
  return PyTensor_New(ASSERT_PTR(functional::View(tensor, *PyTensor_Unpack(other)->shape())));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_permute(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* dims = args;
  if (PyTuple_Size(args) == 1) {
    PyObject* item = PyTuple_GetItem(args, 0);
    if (!PyLong_Check(item)) { dims = item; }
  }

  PyObjectPtr _args = PyObjectPtr(PyTuple_Pack(2, self, dims));
  PyObject* result = functional::permute(NULL, _args.get(), kwargs);
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;

  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_transpose(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  auto tensor = PyTensor_Unpack(self);
  int dim0 = 0;
  int dim1 = 0;
  static const char* keywords[3] = {"dim0", "dim1", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii:transpose", const_cast<char**>(keywords),
                                   &dim0, &dim1)) {
    return NULL;
  }
  return PyTensor_New(ASSERT_PTR(functional::Transpose2dim(tensor, dim0, dim1)));
  END_HANDLE_ERRORS
}

PyMethodDef PyTensorObject_extra_methods[] = {
    {"byte", PyTensorObject_byte, METH_NOARGS, NULL},
    {"size", (PyCFunction)PyTensorObject_size, METH_VARARGS | METH_KEYWORDS, NULL},
    {"argmax", (PyCFunction)PyTensorObject_argmax, METH_VARARGS | METH_KEYWORDS, NULL},
    {"argmin", (PyCFunction)PyTensorObject_argmin, METH_VARARGS | METH_KEYWORDS, NULL},
    {"amin", (PyCFunction)PyTensorObject_amin, METH_VARARGS | METH_KEYWORDS, NULL},
    {"dim", PyTensorObject_dim, METH_NOARGS, NULL},
    {"ndimension", PyTensorObject_dim, METH_NOARGS, NULL},
    {"nelement", PyTensorObject_nelement, METH_NOARGS, NULL},
    {"numel", PyTensorObject_nelement, METH_NOARGS, NULL},
    {"element_size", PyTensorObject_element_size, METH_NOARGS, NULL},
    {"get_device", PyTensorObject_get_device, METH_NOARGS, NULL},
    {"cast", (PyCFunction)PyTensorObject_cast, METH_VARARGS | METH_KEYWORDS, NULL},
    {"diag", (PyCFunction)PyTensorObject_diag, METH_VARARGS | METH_KEYWORDS, NULL},
    {"diagonal", (PyCFunction)PyTensorObject_diagonal, METH_VARARGS | METH_KEYWORDS, NULL},
    {"addcmul", (PyCFunction)PyTensorObject_addcmul, METH_VARARGS | METH_KEYWORDS, NULL},
    {"addcmul_", (PyCFunction)PyTensorObject_addcmul, METH_VARARGS | METH_KEYWORDS, NULL},
    {"sub_", (PyCFunction)PyTensorObject_sub_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"clamp", (PyCFunction)PyTensorObject_clamp, METH_VARARGS | METH_KEYWORDS, NULL},
    {"clamp_", (PyCFunction)PyTensorObject_clamp_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"clip", (PyCFunction)PyTensorObject_clip, METH_VARARGS | METH_KEYWORDS, NULL},
    {"clip_", (PyCFunction)PyTensorObject_clip_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"cpu", PyTensorObject_cpu, METH_NOARGS, NULL},
    {"cuda", (PyCFunction)PyTensorObject_cuda, METH_VARARGS | METH_KEYWORDS, NULL},
    {"in_top_k", (PyCFunction)PyTensorObject_in_top_k, METH_VARARGS | METH_KEYWORDS, NULL},
    {"index_select", (PyCFunction)PyTensorObject_index_select, METH_VARARGS | METH_KEYWORDS, NULL},
    {"minimum", (PyCFunction)PyTensorObject_minimum, METH_VARARGS | METH_KEYWORDS, NULL},
    {"maximum", (PyCFunction)PyTensorObject_maximum, METH_VARARGS | METH_KEYWORDS, NULL},
    {"pow", (PyCFunction)PyTensorObject_pow, METH_VARARGS | METH_KEYWORDS, NULL},
    {"var", (PyCFunction)PyTensorObject_var, METH_VARARGS | METH_KEYWORDS, NULL},
    {"std", (PyCFunction)PyTensorObject_std, METH_VARARGS | METH_KEYWORDS, NULL},
    {"tril", (PyCFunction)PyTensorObject_tril, METH_VARARGS | METH_KEYWORDS, NULL},
    {"triu", (PyCFunction)PyTensorObject_triu, METH_VARARGS | METH_KEYWORDS, NULL},
    {"norm", (PyCFunction)PyTensorObject_norm, METH_VARARGS | METH_KEYWORDS, NULL},
    {"relu", PyTensorObject_relu, METH_NOARGS, NULL},
    {"relu_", PyTensorObject_relu_, METH_NOARGS, NULL},
    {"softmax", (PyCFunction)PyTensorObject_softmax, METH_VARARGS | METH_KEYWORDS, NULL},
    {"log_softmax", (PyCFunction)PyTensorObject_log_softmax, METH_VARARGS | METH_KEYWORDS, NULL},
    {"roll", (PyCFunction)PyTensorObject_roll, METH_VARARGS | METH_KEYWORDS, NULL},
    {"chunk", (PyCFunction)PyTensorObject_chunk, METH_VARARGS | METH_KEYWORDS, NULL},
    {"int", PyTensorObject_int, METH_NOARGS, NULL},
    {"long", PyTensorObject_long, METH_NOARGS, NULL},
    {"float", PyTensorObject_float, METH_NOARGS, NULL},
    {"double", PyTensorObject_double, METH_NOARGS, NULL},

    // macro BINARY_METHOD
    {"floor_divide", (PyCFunction)PyTensorObject_floor_divide, METH_VARARGS | METH_KEYWORDS, NULL},
    {"atan2", (PyCFunction)PyTensorObject_atan2, METH_VARARGS | METH_KEYWORDS, NULL},
    {"gt", (PyCFunction)PyTensorObject_gt, METH_VARARGS | METH_KEYWORDS, NULL},
    {"ge", (PyCFunction)PyTensorObject_ge, METH_VARARGS | METH_KEYWORDS, NULL},
    {"div", (PyCFunction)PyTensorObject_div, METH_VARARGS | METH_KEYWORDS, NULL},
    // {"floor_divide", (PyCFunction)PyTensorObject_div, METH_VARARGS | METH_KEYWORDS, NULL},
    {"div_", (PyCFunction)PyTensorObject_div_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"mul", (PyCFunction)PyTensorObject_mul, METH_VARARGS | METH_KEYWORDS, NULL},
    {"mul_", (PyCFunction)PyTensorObject_mul_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"sub", (PyCFunction)PyTensorObject_sub, METH_VARARGS | METH_KEYWORDS, NULL},
    {"fmod", (PyCFunction)PyTensorObject_fmod, METH_VARARGS | METH_KEYWORDS, NULL},
    {"matmul", (PyCFunction)PyTensorObject_matmul, METH_VARARGS | METH_KEYWORDS, NULL},
    {"logical_and", (PyCFunction)PyTensorObject_logical_and, METH_VARARGS | METH_KEYWORDS, NULL},
    {"logical_or", (PyCFunction)PyTensorObject_logical_or, METH_VARARGS | METH_KEYWORDS, NULL},
    {"logical_xor", (PyCFunction)PyTensorObject_logical_xor, METH_VARARGS | METH_KEYWORDS, NULL},
    {"bmm", (PyCFunction)PyTensorObject_bmm, METH_VARARGS | METH_KEYWORDS, NULL},
    {"ne", (PyCFunction)PyTensorObject_ne, METH_VARARGS | METH_KEYWORDS, NULL},
    {"lt", (PyCFunction)PyTensorObject_lt, METH_VARARGS | METH_KEYWORDS, NULL},
    {"le", (PyCFunction)PyTensorObject_le, METH_VARARGS | METH_KEYWORDS, NULL},

    // macro UNARY_METHOD
    {"abs", PyTensorObject_abs, METH_NOARGS, NULL},
    {"exp", PyTensorObject_exp, METH_NOARGS, NULL},
    {"floor", PyTensorObject_floor, METH_NOARGS, NULL},
    {"floor_", PyTensorObject_floor_, METH_NOARGS, NULL},
    {"acos", PyTensorObject_acos, METH_NOARGS, NULL},
    {"arccos", PyTensorObject_acos, METH_NOARGS, NULL},
    {"acosh", PyTensorObject_acosh, METH_NOARGS, NULL},
    {"arccosh", PyTensorObject_acosh, METH_NOARGS, NULL},
    {"atanh", PyTensorObject_atanh, METH_NOARGS, NULL},
    {"arctanh", PyTensorObject_atanh, METH_NOARGS, NULL},
    {"sign", PyTensorObject_sign, METH_NOARGS, NULL},
    {"sinh", PyTensorObject_sinh, METH_NOARGS, NULL},
    {"tan", PyTensorObject_tan, METH_NOARGS, NULL},
    {"gelu", PyTensorObject_gelu, METH_NOARGS, NULL},
    {"mish", PyTensorObject_mish, METH_NOARGS, NULL},
    {"negative", PyTensorObject_negative, METH_NOARGS, NULL},
    {"neg", PyTensorObject_negative, METH_NOARGS, NULL},
    {"sigmoid", PyTensorObject_sigmoid, METH_NOARGS, NULL},
    {"tanh", PyTensorObject_tanh, METH_NOARGS, NULL},
    {"silu", PyTensorObject_silu, METH_NOARGS, NULL},
    {"selu", PyTensorObject_selu, METH_NOARGS, NULL},
    {"softsign", PyTensorObject_softsign, METH_NOARGS, NULL},
    {"log1p", PyTensorObject_log1p, METH_NOARGS, NULL},
    {"log2", PyTensorObject_log2, METH_NOARGS, NULL},
    {"reciprocal", PyTensorObject_reciprocal, METH_NOARGS, NULL},
    {"asin", PyTensorObject_asin, METH_NOARGS, NULL},
    {"arcsin", PyTensorObject_asin, METH_NOARGS, NULL},
    {"asinh", PyTensorObject_asinh, METH_NOARGS, NULL},
    {"arcsinh", PyTensorObject_asinh, METH_NOARGS, NULL},
    {"atan", PyTensorObject_atan, METH_NOARGS, NULL},
    {"arctan", PyTensorObject_atan, METH_NOARGS, NULL},
    {"ceil", PyTensorObject_ceil, METH_NOARGS, NULL},
    {"cos", PyTensorObject_cos, METH_NOARGS, NULL},
    {"cosh", PyTensorObject_cosh, METH_NOARGS, NULL},
    {"erf", PyTensorObject_erf, METH_NOARGS, NULL},
    {"erfc", PyTensorObject_erfc, METH_NOARGS, NULL},
    {"erfinv", PyTensorObject_erfinv, METH_NOARGS, NULL},
    {"erfinv_", PyTensorObject_erfinv_, METH_NOARGS, NULL},
    {"expm1", PyTensorObject_expm1, METH_NOARGS, NULL},
    {"log", PyTensorObject_log, METH_NOARGS, NULL},
    {"rsqrt", PyTensorObject_rsqrt, METH_NOARGS, NULL},
    {"sqrt", PyTensorObject_sqrt, METH_NOARGS, NULL},
    {"square", PyTensorObject_square, METH_NOARGS, NULL},
    {"round", PyTensorObject_round, METH_NOARGS, NULL},
    {"t", PyTensorObject_t, METH_NOARGS, NULL},
    {"sin", PyTensorObject_sin, METH_NOARGS, NULL},
    {"sin_", PyTensorObject_sin_, METH_NOARGS, NULL},
    {"isnan", PyTensorObject_isnan, METH_NOARGS, NULL},
    {"isinf", PyTensorObject_isinf, METH_NOARGS, NULL},
    {"logical_not", PyTensorObject_logical_not, METH_NOARGS, NULL},
    // {"floor_divide", PyTensorObject_div, METH_O, NULL},
    {"floor", PyTensorObject_floor, METH_NOARGS, NULL},
    {"floor_", PyTensorObject_floor_, METH_NOARGS, NULL},
    {"reshape", (PyCFunction)PyTensorObject_reshape, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reshape_as", (PyCFunction)PyTensorObject_reshape_as, METH_VARARGS | METH_KEYWORDS, NULL},
    {"view", (PyCFunction)PyTensorObject_view, METH_VARARGS | METH_KEYWORDS, NULL},
    {"view_as", (PyCFunction)PyTensorObject_view_as, METH_VARARGS | METH_KEYWORDS, NULL},
    {"permute", (PyCFunction)PyTensorObject_permute, METH_VARARGS | METH_KEYWORDS, NULL},
    {"transpose", (PyCFunction)PyTensorObject_transpose, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL},
};

// tp_richcompare
PyObject* PyTensorObject_richcompare(PyObject* self, PyObject* other, int op) {
  PyObjectPtr tuple(PyTuple_Pack(2, self, other));

  switch (op) {
    case Py_LT: return functional::less(NULL, tuple.get(), NULL);
    case Py_LE: return functional::less_equal(NULL, tuple.get(), NULL);
    case Py_EQ: {
      if (self == Py_None || other == Py_None) return Py_False;
      return functional::equal(NULL, tuple.get(), NULL);
    }
    case Py_NE: return functional::not_equal(NULL, tuple.get(), NULL);
    case Py_GT: return functional::greater(NULL, tuple.get(), NULL);
    case Py_GE: return functional::greater_equal(NULL, tuple.get(), NULL);
  }
  return NULL;
}

}  // namespace one
}  // namespace oneflow

#undef ASSERT
#undef ASSERT_PTR
