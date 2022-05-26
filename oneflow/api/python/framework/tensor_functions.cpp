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
#include "oneflow/api/python/exception/exception.h"
#include "oneflow/api/python/framework/size.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/functional/functional_api.yaml.pybind.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

using functional::PyObjectPtr;

#define ASSERT(x) (x).GetOrThrow()
#define ASSERT_PTR(x) (x).GetPtrOrThrow()

#define NB_UNARY_FUNC(func_name, bind_func)                  \
  static PyObject* func_name(PyObject* self) {               \
    HANDLE_ERRORS                                            \
    PyObject* tuple = PyTuple_Pack(1, self);                 \
    auto* result = bind_func(NULL, tuple, NULL);             \
    if (PyErr_Occurred()) { throw py::error_already_set(); } \
    return result;                                           \
    END_HANDLE_ERRORS                                        \
  }

#define NB_BINARY_FUNC(func_name, bind_func)                 \
  static PyObject* func_name(PyObject* a, PyObject* b) {     \
    HANDLE_ERRORS                                            \
    PyObject* tuple = PyTuple_Pack(2, a, b);                 \
    auto* result = bind_func(NULL, tuple, NULL);             \
    if (PyErr_Occurred()) { throw py::error_already_set(); } \
    return result;                                           \
    END_HANDLE_ERRORS                                        \
  }

NB_UNARY_FUNC(PyTensorObject_nb_absolute, functional::abs);
NB_UNARY_FUNC(PyTensorObject_nb_negative, functional::negative);

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
// TODO: not implemented yet
// NB_UNARY_FUNC(PyTensorObject_positive, functional::positive);
PyObject* PyTensorObject_nb_pow(PyObject* a, PyObject* b, PyObject* unsed) {
  HANDLE_ERRORS
  PyObject* tuple = PyTuple_Pack(2, a, b);
  auto* result = functional::pow(NULL, tuple, NULL);
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_nb_invert(PyObject* self) {
  HANDLE_ERRORS
  CHECK_OR_THROW(PyTensor_Unpack(self)->dtype()->data_type() == DataType::kBool)
      << "~ (operator.invert) is only implemented on integer and Boolean-type tensors";
  PyObject* tuple = PyTuple_Pack(1, self);
  auto* result = functional::logical_not(NULL, tuple, NULL);
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

#define NB_INPLACE_BINARY_FUNC(func_name, bind_func)                     \
  static PyObject* func_name(PyObject* a, PyObject* b) {                 \
    HANDLE_ERRORS                                                        \
    PyObject* tuple = PyTuple_Pack(2, a, b);                             \
    PyObject* dict = PyDict_New();                                       \
    CHECK_OR_THROW(PyDict_SetItemString(dict, "inplace", Py_True) > -1); \
    const auto& result = bind_func(NULL, tuple, dict);                   \
    if (PyErr_Occurred()) { throw py::error_already_set(); }             \
    return result;                                                       \
    END_HANDLE_ERRORS                                                    \
  }

// TODO: still have bug here
// NB_INPLACE_BINARY_FUNC(PyTensorObject_inplace_add, functional::add, "add");
NB_INPLACE_BINARY_FUNC(PyTensorObject_nb_inplace_sub, functional::sub);
NB_INPLACE_BINARY_FUNC(PyTensorObject_nb_inplace_mul, functional::mul);
NB_INPLACE_BINARY_FUNC(PyTensorObject_nb_inplace_fmod, functional::fmod);
NB_INPLACE_BINARY_FUNC(PyTensorObject_nb_inplace_and, functional::logical_and);
NB_INPLACE_BINARY_FUNC(PyTensorObject_nb_inplace_xor, functional::logical_xor);
NB_INPLACE_BINARY_FUNC(PyTensorObject_nb_inplace_or, functional::logical_or);
NB_INPLACE_BINARY_FUNC(PyTensorObject_nb_inplace_floor_div, functional::floor_divide);
NB_INPLACE_BINARY_FUNC(PyTensorObject_nb_inplace_true_div, functional::div);
// TODO: inplace matmul not supported yet
// INPLACE_BINARY_FUNC(PyTensorObject_inplace_matrix_multiply, functional::matmul, "matmul");

PyObject* PyTensorObject_nb_inplace_pow(PyObject* a, PyObject* b, PyObject* unsed) {
  HANDLE_ERRORS
  PyObject* tuple = PyTuple_Pack(2, a, b);
  PyObject* dict = PyDict_New();
  CHECK_OR_THROW(PyDict_SetItemString(dict, "inplace", Py_True) > -1);
  auto* result = functional::pow(NULL, tuple, NULL);
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

    NULL,                            // nb_inplace_add
    PyTensorObject_nb_inplace_sub,   // nb_inplace_sub
    PyTensorObject_nb_inplace_mul,   // nb_inplace_mul
    PyTensorObject_nb_inplace_fmod,  // nb_inplace_remainder
    PyTensorObject_nb_inplace_pow,   // nb_inplace_pow
    NULL,                            // nb_inplace_lshift
    NULL,                            // nb_inplace_rshift
    PyTensorObject_nb_inplace_and,   // nb_inplace_and
    PyTensorObject_nb_inplace_xor,   // nb_inplace_xor
    PyTensorObject_nb_inplace_or,    // nb_inplace_or

    PyTensorObject_nb_floor_div,          // nb_floor_div
    PyTensorObject_nb_true_div,           // nb_true_div
    PyTensorObject_nb_inplace_floor_div,  // nb_inplace_floor_div
    PyTensorObject_nb_inplace_true_div,   // nb_inplace_true_div

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
    return bind_func(NULL, tuple.get(), NULL);                                              \
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
  auto device_type = ASSERT(PyTensor_Unpack(self)->device())->enum_type();
  CHECK_OR_THROW(device_type == DeviceType::kCUDA)
      << "get_device is only available for GPU tensor.";
  return functional::CastToPyObject(ASSERT(PyTensor_Unpack(self)->device())->device_id());
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_size(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython size" << std::endl;
  PyObject* idx = NULL;
  static const char* keywords[2] = {"idx", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:size", const_cast<char**>(keywords), &idx)) {
    return NULL;
  }
  Shape shape = *PyTensor_Unpack(self)->shape();
  PyObject* shape_object = TensorSize_NewFromShape(shape);
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
  return functional::argmax(NULL, tuple.get(), dict.get());
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
  return functional::argmin(NULL, tuple.get(), dict.get());
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
  return functional::amin(NULL, tuple.get(), dict.get());
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
  return functional::cast(NULL, tuple.get(), NULL);
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_diag(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython" << std::endl;
  PyObject* diagonal = PyLong_FromLong(0);
  static const char* keywords[2] = {"diagonal", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:diag", const_cast<char**>(keywords),
                                   &diagonal)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(2, self, diagonal));
  return functional::diag(NULL, tuple.get(), NULL);
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_diagonal(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython" << std::endl;
  PyObject* offset = PyLong_FromLong(0);
  PyObject* dim1 = PyLong_FromLong(0);
  PyObject* dim2 = PyLong_FromLong(1);
  static const char* keywords[4] = {"offset", "dim1", "dim2", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOO:diag", const_cast<char**>(keywords), &offset,
                                   &dim1, &dim2)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(4, self, offset, dim1, dim2));
  return functional::diagonal(NULL, tuple.get(), NULL);
  END_HANDLE_ERRORS
}

// static PyObject* PyTensorObject_add(PyObject* self, PyObject* args, PyObject* kwargs) {
//   HANDLE_ERRORS
//   std::cout << "cpython" << std::endl;
//   PyObject* other = NULL;
//   PyObject* alpha = PyFloat_FromDouble(1.0);
//   static const char* keywords[3] = {"other", "alpha", NULL};
//   if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|$O:add", const_cast<char**>(keywords),
//   &other, &alpha)) {
//     return NULL;
//   }
//   std::cout << "run1" << std::endl;
//   if(other == NULL)
//     std::cout << "other null" << std::endl;
//   std::cout << "run2" << std::endl;
//   PyObjectPtr tuple(PyTuple_Pack(3, self, other, alpha));//, other, alpha));
//   // PyObjectPtr dict(PyDict_New());
//   PyObject* dict = PyDict_New();
//   CHECK_OR_THROW(PyDict_SetItemString(dict, "alpha", alpha) > -1);
//   CHECK_OR_THROW(PyDict_SetItemString(dict, "inplace", Py_False) > -1);
//   std::cout << "run3" << std::endl;
//   return functional::add(NULL, tuple.get(), dict);
//   END_HANDLE_ERRORS
// }

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
  return functional::addcmul(NULL, tuple.get(), dict.get());
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
  return functional::addcmul_(NULL, tuple.get(), dict.get());
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
  return functional::sub(NULL, tuple.get(), dict.get());
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
  return functional::clamp(NULL, tuple.get(), dict.get());
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
  return functional::clamp_(NULL, tuple.get(), dict.get());
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
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO:clip", const_cast<char**>(keywords), &min,
                                   &max)) {
    return NULL;
  }
  PyObjectPtr tuple(PyTuple_Pack(1, self));
  PyObjectPtr dict(PyDict_New());
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "min", min) > -1);
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "max", max) > -1);
  return functional::clip_(NULL, tuple.get(), dict.get());
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_cpu(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  std::cout << "cpython" << std::endl;
  PyObjectPtr dict(PyDict_New());
  PyObjectPtr device(PyUnicode_FromString("cpu"));
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "device", device.get()) > -1);
  PyObjectPtr tuple(PyTuple_Pack(1, self));
  return functional::to(NULL, tuple.get(), dict.get());
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_cuda(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  std::cout << "cpython" << std::endl;
  PyObject* device = Py_None;
  static const char* keywords[2] = {"device", NULL};
  if(!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:cuda", const_cast<char**>(keywords), &device))
  {
    return NULL;
  }
  if(device == Py_None)
    device = PyUnicode_FromString("cuda");
  else if(PyLong_Check(device)) {
    device = PyUnicode_Concat(PyUnicode_FromString("cuda:"), PyUnicode_FromFormat("%d", PyLong_AsLong(device)));
  }
  PyObjectPtr dict(PyDict_New());
  CHECK_OR_THROW(PyDict_SetItemString(dict.get(), "device", device) > -1);
  PyObjectPtr tuple(PyTuple_Pack(1, self));
  return functional::to(NULL, tuple.get(), dict.get());
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
    // {"add", (PyCFunction)PyTensorObject_add, METH_VARARGS | METH_KEYWORDS, NULL},
    {"addcmul", (PyCFunction)PyTensorObject_addcmul, METH_VARARGS | METH_KEYWORDS, NULL},
    {"addcmul_", (PyCFunction)PyTensorObject_addcmul, METH_VARARGS | METH_KEYWORDS, NULL},
    {"sub_", (PyCFunction)PyTensorObject_sub_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"clamp", (PyCFunction)PyTensorObject_clamp, METH_VARARGS | METH_KEYWORDS, NULL},
    {"clamp_", (PyCFunction)PyTensorObject_clamp_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"clip", (PyCFunction)PyTensorObject_clip, METH_VARARGS | METH_KEYWORDS, NULL},
    {"clip_", (PyCFunction)PyTensorObject_clip_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"cpu", PyTensorObject_cpu, METH_NOARGS, NULL},
    {"cuda", (PyCFunction)PyTensorObject_cuda, METH_VARARGS | METH_KEYWORDS, NULL},

    // macro BINARY_METHOD
    {"floor_divide", (PyCFunction)PyTensorObject_floor_divide, METH_VARARGS | METH_KEYWORDS, NULL},
    {"atan2", (PyCFunction)PyTensorObject_atan2, METH_VARARGS | METH_KEYWORDS, NULL},
    {"gt", (PyCFunction)PyTensorObject_gt, METH_VARARGS | METH_KEYWORDS, NULL},
    {"ge", (PyCFunction)PyTensorObject_ge, METH_VARARGS | METH_KEYWORDS, NULL},
    {"div", (PyCFunction)PyTensorObject_div, METH_VARARGS | METH_KEYWORDS, NULL},
    {"floor_divide", (PyCFunction)PyTensorObject_div, METH_VARARGS | METH_KEYWORDS, NULL},
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
    {"floor", PyTensorObject_floor, METH_NOARGS, NULL},
    {"floor_", PyTensorObject_floor_, METH_NOARGS, NULL},
    {"logical_not", PyTensorObject_logical_not, METH_NOARGS, NULL},
    {NULL},
};

// tp_richcompare
PyObject* PyTensorObject_richcompare(PyObject* self, PyObject* other, int op) {
  PyObject* tuple = PyTuple_Pack(2, self, other);

  switch (op) {
    case Py_LT: return functional::less(NULL, tuple, NULL);
    case Py_LE: return functional::less_equal(NULL, tuple, NULL);
    case Py_EQ: {
      if (self == Py_None || other == Py_None) return Py_False;
      return functional::equal(NULL, tuple, NULL);
    }
    case Py_NE: return functional::not_equal(NULL, tuple, NULL);
    case Py_GT: return functional::greater(NULL, tuple, NULL);
    case Py_GE: return functional::greater_equal(NULL, tuple, NULL);
  }
  return NULL;
}

}  // namespace one
}  // namespace oneflow

#undef ASSERT
#undef ASSERT_PTR