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
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/functional/functional_api.yaml.pybind.h"
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

NB_UNARY_FUNC(PyTensorObject_absolute, functional::abs);
NB_UNARY_FUNC(PyTensorObject_negative, functional::negative);
// TODO: not implemented yet
// NB_UNARY_FUNC(PyTensorObject_positive, functional::positive);

NB_BINARY_FUNC(PyTensorObject_add, functional::add);
NB_BINARY_FUNC(PyTensorObject_sub, functional::sub);
NB_BINARY_FUNC(PyTensorObject_mul, functional::mul);
NB_BINARY_FUNC(PyTensorObject_fmod, functional::fmod);
NB_BINARY_FUNC(PyTensorObject_div, functional::div);
NB_BINARY_FUNC(PyTensorObject_and, functional::logical_and);
NB_BINARY_FUNC(PyTensorObject_xor, functional::logical_xor);
NB_BINARY_FUNC(PyTensorObject_or, functional::logical_or);
NB_BINARY_FUNC(PyTensorObject_floor_div, functional::floor_divide);
NB_BINARY_FUNC(PyTensorObject_true_div, functional::div);
NB_BINARY_FUNC(PyTensorObject_matrix_multiply, functional::matmul);

PyObject* PyTensorObject_pow(PyObject* a, PyObject* b, PyObject* unsed) {
  HANDLE_ERRORS
  PyObjectPtr tuple(PyTuple_Pack(2, a, b));
  auto* result = functional::pow(NULL, tuple.get(), NULL);
  if (PyErr_Occurred()) { throw py::error_already_set(); }
  return result;
  END_HANDLE_ERRORS
}

static PyObject* PyTensorObject_invert(PyObject* self) {
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
    const auto& result = bind_func(NULL, tuple.get(), dict.get());             \
    if (PyErr_Occurred()) { throw py::error_already_set(); }                   \
    return result;                                                             \
    END_HANDLE_ERRORS                                                          \
  }

// inplace operators
NB_INPLACE_BINARY_FUNC(PyTensorObject_inplace_add, functional::add);
NB_INPLACE_BINARY_FUNC(PyTensorObject_inplace_sub, functional::sub);
// The interface of inplace mul not mul(*, inplace=True) but mul_
NB_BINARY_FUNC(PyTensorObject_inplace_mul, functional::mul_);
NB_BINARY_FUNC(PyTensorObject_inplace_true_div, functional::div_);

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
    PyTensorObject_add,       // nb_add
    PyTensorObject_sub,       // nb_subtract
    PyTensorObject_mul,       // nb_multiply
    PyTensorObject_fmod,      // nb_remainder
    NULL,                     // nb_divmod
    PyTensorObject_pow,       // nb_power
    PyTensorObject_negative,  // nb_negative
    NULL,                     // nb_positive
    PyTensorObject_absolute,  // nb_absolute
    NULL,                     // nb_bool
    PyTensorObject_invert,    // nb_invert
    NULL,                     // nb_lshift
    NULL,                     // nb_rshift
    PyTensorObject_and,       // nb_and
    PyTensorObject_xor,       // nb_xor
    PyTensorObject_or,        // nb_or
    NULL,                     // nb_int
    NULL,                     // nb_reserved
    NULL,                     // nb_float

    PyTensorObject_inplace_add,  // nb_inplace_add
    PyTensorObject_inplace_sub,  // nb_inplace_sub
    PyTensorObject_inplace_mul,  // nb_inplace_mul
    NULL,                        // nb_inplace_remainder
    NULL,                        // nb_inplace_pow
    NULL,                        // nb_inplace_lshift
    NULL,                        // nb_inplace_rshift
    NULL,                        // nb_inplace_and
    NULL,                        // nb_inplace_xor
    NULL,                        // nb_inplace_or

    PyTensorObject_floor_div,         // nb_floor_div
    PyTensorObject_true_div,          // nb_true_div
    NULL,                             // nb_inplace_floor_div
    PyTensorObject_inplace_true_div,  // nb_inplace_true_div

    NULL,                            // nb_index
    PyTensorObject_matrix_multiply,  // nb_matrix_multiply
    NULL,                            // not implemented yet nb_inplace_matrix_multiply

};

// extra methods
#define UNARY_METHOD(func_name, bind_func)                             \
  static PyObject* func_name(PyObject* self, PyObject* unused) {       \
    HANDLE_ERRORS                                                      \
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

static PyObject* PyTensorObject_reshape(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  PyObject* shape = args;
  if (kwargs != NULL) {
    CHECK_OR_THROW(PyTuple_Size(args) == 0)
        << Error::TypeError() << "reshape() got multiple values for argument 'shape'";
    // keyword parameter
    static const char* keywords[2] = {"shape", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|:reshape", const_cast<char**>(keywords),
                                     &shape)) {
      return NULL;
    }
  } else {
    // positional parameter
    PyObject* item = PyTuple_GetItem(args, 0);
    if (!PyLong_Check(item)) { shape = item; }
  }

  CHECK_OR_THROW(functional::PyLongSequenceCheck(shape))
      << Error::TypeError() << "reshape(): argument 'shape' must be tuple of ints, but found "
      << functional::PyStringAsString(PyObject_Str((PyObject*)Py_TYPE(shape)));
  const auto& dims = functional::PyUnpackLongSequence<int64_t>(shape);
  DimVector dim(dims.begin(), dims.end());
  return PyTensor_New(ASSERT_PTR(functional::Reshape(PyTensor_Unpack(self), Shape(dim))));
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

static PyObject* PyTensorObject_view(PyObject* self, PyObject* args) {
  HANDLE_ERRORS
  PyObject* shape = args;
  if (PyTuple_Size(args) == 1) {
    PyObject* item = PyTuple_GetItem(args, 0);
    if (!PyLong_Check(item)) { shape = item; }
  }
  CHECK_OR_THROW(functional::PyLongSequenceCheck(shape))
      << Error::TypeError() << "view() received an invalid combination of arguments - got ("
      << Py_TYPE(shape)->tp_name << "), but expected tuple of ints size.";

  const auto& dims = functional::PyUnpackLongSequence<int64_t>(shape);
  DimVector dim(dims.begin(), dims.end());
  return PyTensor_New(ASSERT_PTR(functional::View(PyTensor_Unpack(self), Shape(dim))));
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
  if (PyTuple_Size(args) == 0) {
    // keyword parameter
    static const char* keywords[2] = {"dims", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|:permute", const_cast<char**>(keywords),
                                     &dims)) {
      return NULL;
    }
  } else if (PyTuple_Size(args) == 1) {
    // positional parameter
    PyObject* item = PyTuple_GetItem(args, 0);
    if (!PyLong_Check(item)) { dims = item; }
  }
  CHECK_OR_THROW(functional::PyLongSequenceCheck(dims))
      << Error::TypeError() << "permute(): argument 'dims' must be tuple of ints, but found "
      << Py_TYPE(dims)->tp_name;
  const auto& dims_vec = functional::PyUnpackLongSequence<int32_t>(dims);
  return PyTensor_New(ASSERT_PTR(functional::Permute(PyTensor_Unpack(self), dims_vec)));
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
    {"floor_divide", PyTensorObject_div, METH_O, NULL},
    {"floor", PyTensorObject_floor, METH_NOARGS, NULL},
    {"floor_", PyTensorObject_floor_, METH_NOARGS, NULL},
    {"reshape", (PyCFunction)PyTensorObject_reshape, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reshape_as", (PyCFunction)PyTensorObject_reshape_as, METH_VARARGS | METH_KEYWORDS, NULL},
    {"view", PyTensorObject_view, METH_VARARGS, NULL},
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
