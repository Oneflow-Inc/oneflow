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
#include <Python.h>
#include "oneflow/api/python/exception/exception.h"
#include "oneflow/api/python/framework/size.h"
#include "oneflow/api/python/framework/tensor.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/functional/python_arg.h"
#include "oneflow/api/python/functional/functional_api.yaml.pybind.h"
#include "oneflow/api/python/functional/tensor_api.yaml.pybind.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/ofblob/ofblob.e.h"
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

namespace oneflow {
namespace one {

#define ASSERT(x) (x).GetOrThrow()
#define ASSERT_PTR(x) (x).GetPtrOrThrow()

#define NB_UNARY_FUNC(func_name, bind_func, name)               \
  static PyObject* func_name(PyObject* self) {               \
    HANDLE_ERRORS                                            \
    PyObject* tuple = PyTuple_Pack(1, self);                 \
    std::cout << "cpython " << name << std::endl; \
    auto* result = bind_func(NULL, tuple, NULL);             \
    if (PyErr_Occurred()) { throw py::error_already_set(); } \
    return result;                                           \
    END_HANDLE_ERRORS                                        \
  }

#define NB_BINARY_FUNC(func_name, bind_func, name)              \
  static PyObject* func_name(PyObject* a, PyObject* b) {     \
    HANDLE_ERRORS                                            \
    PyObject* tuple = PyTuple_Pack(2, a, b);                 \
    std::cout << "cpython " << name << std::endl; \
    auto* result = bind_func(NULL, tuple, NULL);             \
    if (PyErr_Occurred()) { throw py::error_already_set(); } \
    return result;                                           \
    END_HANDLE_ERRORS                                        \
  }

NB_BINARY_FUNC(PyTensorObject_add, functional::add, "add");
NB_BINARY_FUNC(PyTensorObject_sub, functional::sub, "sub");
NB_BINARY_FUNC(PyTensorObject_mul, functional::mul, "mul");
NB_BINARY_FUNC(PyTensorObject_fmod, functional::fmod, "fmod");
NB_BINARY_FUNC(PyTensorObject_div, functional::div, "div");
PyObject* PyTensorObject_pow(PyObject* a, PyObject* b, PyObject* unsed) {
  HANDLE_ERRORS
  PyObject* tuple = PyTuple_Pack(2, a, b);
  auto* result = functional::pow(NULL, tuple, NULL);
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  // std::cout << "using cpython pow" << std::endl;
  return result;
  END_HANDLE_ERRORS
}
NB_UNARY_FUNC(PyTensorObject_negative, functional::negative, "negative");
// NB_UNARY_FUNC(PyTensorObject_positive, functional::positive);
NB_UNARY_FUNC(PyTensorObject_absolute, functional::abs, "abs");

static PyObject* PyTensorObject_invert(PyObject* self) {
  HANDLE_ERRORS
  CHECK_OR_THROW(PyTensor_Unpack(self)->dtype()->data_type() == DataType::kBool)
      << "~ (operator.invert) is only implemented on integer and Boolean-type tensors";
  PyObject* tuple = PyTuple_Pack(1, self);
  auto* result = functional::logical_not(NULL, tuple, NULL);
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  std::cout << "using cpython invert" << std::endl;
  return result;
  END_HANDLE_ERRORS
}

NB_BINARY_FUNC(PyTensorObject_and, functional::logical_and, "logical_and");
NB_BINARY_FUNC(PyTensorObject_xor, functional::logical_xor, "logical_xor");
NB_BINARY_FUNC(PyTensorObject_or, functional::logical_or, "logical or");

#define INPLACE_BINARY_FUNC(func_name, bind_func, name)                  \
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

INPLACE_BINARY_FUNC(PyTensorObject_inplace_add, functional::add, "add");
INPLACE_BINARY_FUNC(PyTensorObject_inplace_sub, functional::sub, "sub");
INPLACE_BINARY_FUNC(PyTensorObject_inplace_mul, functional::mul, "mul");
INPLACE_BINARY_FUNC(PyTensorObject_inplace_fmod, functional::fmod, "fmod");

PyObject* PyTensorObject_inplace_pow(PyObject* a, PyObject* b, PyObject* unsed) {
  HANDLE_ERRORS
  PyObject* tuple = PyTuple_Pack(2, a, b);
  PyObject* dict = PyDict_New();
  CHECK_OR_THROW(PyDict_SetItemString(dict, "inplace", Py_True) > -1);
  auto* result = functional::pow(NULL, tuple, NULL);
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  // std::cout << "using cpython bind_func" << std::endl;
  return result;
  END_HANDLE_ERRORS
}

INPLACE_BINARY_FUNC(PyTensorObject_inplace_and, functional::logical_and, "logical_and");
INPLACE_BINARY_FUNC(PyTensorObject_inplace_xor, functional::logical_xor, "logical_xor");
INPLACE_BINARY_FUNC(PyTensorObject_inplace_or, functional::logical_or, "logical_or");
NB_BINARY_FUNC(PyTensorObject_floor_div, functional::floor_divide, "floor divide");
NB_BINARY_FUNC(PyTensorObject_true_div, functional::div, "true_div");
INPLACE_BINARY_FUNC(PyTensorObject_inplace_floor_div, functional::floor_divide, "floor_divide");
INPLACE_BINARY_FUNC(PyTensorObject_inplace_true_div, functional::div, "true_divide");
NB_BINARY_FUNC(PyTensorObject_matrix_multiply, functional::matmul, "matmul");
// INPLACE_BINARY_FUNC(PyTensorObject_inplace_matrix_multiply, functional::matmul, "matmul");

PyNumberMethods PyTensorObject_as_number = {
    PyTensorObject_add,       // nb_add, __add__
    PyTensorObject_sub,       // nb_subtract, __sub__
    PyTensorObject_mul,       // nb_multiply, __mul__
    PyTensorObject_fmod,      // nb_remainder, __mod__, __rmod__
    NULL,                     // nb_divmod
    PyTensorObject_pow,       // nb_power
    PyTensorObject_negative,  // nb_negative
    NULL,                     // nb_positive
    PyTensorObject_absolute,  // nb_absolute
    NULL,                     // nb_bool torch doesn't implement
    PyTensorObject_invert,    // nb_invert
    NULL,                     // nb_lshift
    NULL,                     // nb_rshift
    PyTensorObject_and,       // nb_and
    PyTensorObject_xor,       // nb_xor
    PyTensorObject_or,        // nb_or
    NULL,                     // nb_int
    NULL,                     // nb_reserved
    NULL,                     // nb_float

    NULL,                         // bug PyTensorObject_inplace_add, //nb_inplace_add
    PyTensorObject_inplace_sub,   // nb_inplace_sub
    PyTensorObject_inplace_mul,   // nb_inplace_mul
    PyTensorObject_inplace_fmod,  // nb_inplace_remainder
    PyTensorObject_inplace_pow,   // nb_inplace_pow
    NULL,                         // nb_inplace_lshift
    NULL,                         // nb_inplace_rshift
    PyTensorObject_inplace_and,   // nb_inplace_and
    PyTensorObject_inplace_xor,   // nb_inplace_xor
    PyTensorObject_inplace_or,    // nb_inplace_or

    PyTensorObject_floor_div,          // nb_floor_div
    PyTensorObject_true_div,           // nb_true_div
    PyTensorObject_inplace_floor_div,  // nb_inplace_floor_div
    PyTensorObject_inplace_true_div,   // nb_inplace_true_div

    NULL,                            // nb_index
    PyTensorObject_matrix_multiply,  // nb_matrix_multiply
    NULL,                            // not implemented yet nb_inplace_matrix_multiply

};


// extra methods

static PyObject* PyTensorObject_byte(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  // std::cout << "cpython byte" << std::endl;
  return PyTensor_New(ASSERT_PTR(functional::To(PyTensor_Unpack(self), DType::Int8(), false)));
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_dim(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  // std::cout << "cpython dim" << std::endl;
  return functional::CastToPyObject(PyTensor_Unpack(self)->ndim());
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_nelement(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  // std::cout << "cpython nelement" << std::endl;
  return functional::CastToPyObject(PyTensor_Unpack(self)->nelement());
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_element_size(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  // std::cout << "cpython element_size" << std::endl;
  return functional::CastToPyObject(PyTensor_Unpack(self)->dtype()->bytes());
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_get_device(PyObject* self, PyObject* unused) {
  HANDLE_ERRORS
  // std::cout << "cpython get_device" << std::endl;
  auto device_type = ASSERT(PyTensor_Unpack(self)->device())->enum_type();
  CHECK_OR_THROW(device_type == DeviceType::kCUDA)
      << "get_device is only available for GPU tensor.";
  return functional::CastToPyObject(ASSERT(PyTensor_Unpack(self)->device())->device_id());
  END_HANDLE_ERRORS
}

#define UNARY_METHOD(func_name, bind_func, name)                       \
  static PyObject* func_name(PyObject* self, PyObject* unused) {       \
    HANDLE_ERRORS                                                      \
    return PyTensor_New(ASSERT_PTR(bind_func(PyTensor_Unpack(self)))); \
    END_HANDLE_ERRORS                                                  \
  }

UNARY_METHOD(PyTensorObject_abs, functional::Abs, "abs");
UNARY_METHOD(PyTensorObject_exp, functional::Exp, "exp");
UNARY_METHOD(PyTensorObject_floor, functional::Floor, "floor");
UNARY_METHOD(PyTensorObject_floor_, functional::Floor_, "floor_");
UNARY_METHOD(PyTensorObject_sign, functional::Sign, "sign");
UNARY_METHOD(PyTensorObject_gelu, functional::Gelu, "gelu");
UNARY_METHOD(PyTensorObject_mish, functional::Mish, "mish");
UNARY_METHOD(PyTensorObject_negative, functional::Negative, "negatinve");
UNARY_METHOD(PyTensorObject_sigmoid, functional::Sigmoid, "sigmoid");
UNARY_METHOD(PyTensorObject_silu, functional::Silu, "silu");
UNARY_METHOD(PyTensorObject_selu, functional::Selu, "selu");
UNARY_METHOD(PyTensorObject_softsign, functional::SoftSign, "softsign");
UNARY_METHOD(PyTensorObject_log1p, functional::Log1p, "log1p");
UNARY_METHOD(PyTensorObject_log2, functional::Log2, "log2");
UNARY_METHOD(PyTensorObject_reciprocal, functional::Reciprocal, "reciprocal");
UNARY_METHOD(PyTensorObject_ceil, functional::Ceil, "ceil");
UNARY_METHOD(PyTensorObject_erf, functional::Erf, "erf");
UNARY_METHOD(PyTensorObject_erfc, functional::Erfc, "erfc");
UNARY_METHOD(PyTensorObject_erfinv, functional::Erfinv, "erfinv");
UNARY_METHOD(PyTensorObject_erfinv_, functional::ErfinvInplace, "erfinv_inplace");
UNARY_METHOD(PyTensorObject_expm1, functional::Expm1, "expm1");
UNARY_METHOD(PyTensorObject_log, functional::Log, "log");
UNARY_METHOD(PyTensorObject_rsqrt, functional::Rsqrt, "rsqrt");
UNARY_METHOD(PyTensorObject_sqrt, functional::Sqrt, "sqrt");
UNARY_METHOD(PyTensorObject_square, functional::Square, "square");
UNARY_METHOD(PyTensorObject_round, functional::Round, "round");
UNARY_METHOD(PyTensorObject_t, functional::TransposeAllDimFunction, "t");
UNARY_METHOD(PyTensorObject_isnan, functional::IsNan, "isnan");
UNARY_METHOD(PyTensorObject_isinf, functional::IsInf, "isinf");
UNARY_METHOD(PyTensorObject_sin, functional::Sin, "sin");
UNARY_METHOD(PyTensorObject_sin_, functional::Sin_, "sin_");
UNARY_METHOD(PyTensorObject_asin, functional::Asin, "asin");
UNARY_METHOD(PyTensorObject_cos, functional::Cos, "cos");
UNARY_METHOD(PyTensorObject_acos, functional::Acos, "acos");
UNARY_METHOD(PyTensorObject_tan, functional::Tan, "Tan");
UNARY_METHOD(PyTensorObject_atan, functional::Atan, "atan");
UNARY_METHOD(PyTensorObject_sinh, functional::Sinh, "sinh");
UNARY_METHOD(PyTensorObject_asinh, functional::Asinh, "asinh");
UNARY_METHOD(PyTensorObject_cosh, functional::Cosh, "cosh");
UNARY_METHOD(PyTensorObject_acosh, functional::Acosh, "acosh");
UNARY_METHOD(PyTensorObject_tanh, functional::Tanh, "tanh");
UNARY_METHOD(PyTensorObject_atanh, functional::Atanh, "atanh");

PyMethodDef PyTensorObject_extra_methods[] = {
    {"byte", PyTensorObject_byte, METH_NOARGS, NULL},
    {"dim", PyTensorObject_dim, METH_NOARGS, NULL},
    {"ndimension", PyTensorObject_dim, METH_NOARGS, NULL},
    {"nelement", PyTensorObject_nelement, METH_NOARGS, NULL},
    {"numel", PyTensorObject_nelement, METH_NOARGS, NULL},
    {"element_size", PyTensorObject_element_size, METH_NOARGS, NULL},
    {"get_device", PyTensorObject_get_device, METH_NOARGS, NULL},
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
};


}  // namespace one
}  // namespace oneflow

#undef ASSERT
#undef ASSERT_PTR