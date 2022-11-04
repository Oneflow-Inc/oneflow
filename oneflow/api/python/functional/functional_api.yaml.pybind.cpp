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

// Generated from oneflow/core/functional/functional_api.yaml. DO NOT EDIT!

#include <Python.h>

#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/api/python/exception/exception.h"
#include "oneflow/api/python/functional/common.h"
#include "oneflow/api/python/functional/function_def.h"
#include "oneflow/api/python/functional/python_arg.h"
#include "oneflow/api/python/functional/python_arg_parser.h"
#include "oneflow/api/python/functional/python_frame.h"
#include "oneflow/api/python/functional/python_return_types.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {
namespace functional {

struct AddSchema_TTTScB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other, const Scalar& alpha, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Add;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other, *, Scalar alpha=1, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t AddSchema_TTTScB::max_args;
constexpr size_t AddSchema_TTTScB::max_pos_args;
constexpr char const* AddSchema_TTTScB::signature;
FunctionDef AddSchema_TTTScB::function_def = {
/*name*/"add",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"alpha", /*default_value*/Scalar(1), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct ScalarAddSchema_TTScScB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& other, const Scalar& alpha, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarAdd;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar other, *, Scalar alpha=1, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t ScalarAddSchema_TTScScB::max_args;
constexpr size_t ScalarAddSchema_TTScScB::max_pos_args;
constexpr char const* ScalarAddSchema_TTScScB::signature;
FunctionDef ScalarAddSchema_TTScScB::function_def = {
/*name*/"add",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"alpha", /*default_value*/Scalar(1), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct ScalarAddSchema_TScTSc {
  using FType = Maybe<one::Tensor> (const Scalar& input, const std::shared_ptr<one::Tensor>& other, const Scalar& alpha);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarAdd;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Scalar input, Tensor other, *, Scalar alpha=1)";
  static FunctionDef function_def;
};

constexpr size_t ScalarAddSchema_TScTSc::max_args;
constexpr size_t ScalarAddSchema_TScTSc::max_pos_args;
constexpr char const* ScalarAddSchema_TScTSc::signature;
FunctionDef ScalarAddSchema_TScTSc::function_def = {
/*name*/"add",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"alpha", /*default_value*/Scalar(1), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct AddSchema_TTtB {
  using FType = Maybe<one::Tensor> (const TensorTuple& inputs, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Add;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (TensorTuple inputs, *, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t AddSchema_TTtB::max_args;
constexpr size_t AddSchema_TTtB::max_pos_args;
constexpr char const* AddSchema_TTtB::signature;
FunctionDef AddSchema_TTtB::function_def = {
/*name*/"add",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"inputs", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* add(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("add");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AddSchema_TTTScB, functional::ScalarAddSchema_TTScScB, functional::ScalarAddSchema_TScTSc, functional::AddSchema_TTtB> parser("add");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Add(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Scalar>(), r[3].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ScalarAdd(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>(), r[2].As<Scalar>(), r[3].As<bool>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::ScalarAdd(r[0].As<Scalar>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Scalar>()));
  }
  if (idx == 3) {
    return CastToPyObject(functional::Add(r[0].As<TensorTuple>(), r[1].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AminSchema_TTI32lB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Optional<std::vector<int32_t>>& dim, bool keepdim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Amin;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Int32List dim=None, Bool keepdim=False)";
  static FunctionDef function_def;
};

constexpr size_t AminSchema_TTI32lB::max_args;
constexpr size_t AminSchema_TTI32lB::max_pos_args;
constexpr char const* AminSchema_TTI32lB::signature;
FunctionDef AminSchema_TTI32lB::function_def = {
/*name*/"amin",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/Optional<std::vector<int32_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* amin(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("amin");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AminSchema_TTI32lB> parser("amin");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Amin(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<std::vector<int32_t>>>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SubSchema_TTTScB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other, const Scalar& alpha, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Sub;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other, *, Scalar alpha=1, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t SubSchema_TTTScB::max_args;
constexpr size_t SubSchema_TTTScB::max_pos_args;
constexpr char const* SubSchema_TTTScB::signature;
FunctionDef SubSchema_TTTScB::function_def = {
/*name*/"sub",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"alpha", /*default_value*/Scalar(1), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct ScalarSubSchema_TTScScB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& other, const Scalar& alpha, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarSub;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar other, *, Scalar alpha=1, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t ScalarSubSchema_TTScScB::max_args;
constexpr size_t ScalarSubSchema_TTScScB::max_pos_args;
constexpr char const* ScalarSubSchema_TTScScB::signature;
FunctionDef ScalarSubSchema_TTScScB::function_def = {
/*name*/"sub",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"alpha", /*default_value*/Scalar(1), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct ScalarSubSchema_TScTSc {
  using FType = Maybe<one::Tensor> (const Scalar& input, const std::shared_ptr<one::Tensor>& other, const Scalar& alpha);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarSub;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Scalar input, Tensor other, *, Scalar alpha=1)";
  static FunctionDef function_def;
};

constexpr size_t ScalarSubSchema_TScTSc::max_args;
constexpr size_t ScalarSubSchema_TScTSc::max_pos_args;
constexpr char const* ScalarSubSchema_TScTSc::signature;
FunctionDef ScalarSubSchema_TScTSc::function_def = {
/*name*/"sub",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"alpha", /*default_value*/Scalar(1), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* sub(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("sub");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SubSchema_TTTScB, functional::ScalarSubSchema_TTScScB, functional::ScalarSubSchema_TScTSc> parser("sub");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Sub(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Scalar>(), r[3].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ScalarSub(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>(), r[2].As<Scalar>(), r[3].As<bool>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::ScalarSub(r[0].As<Scalar>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Scalar>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MulSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Mul;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t MulSchema_TTT::max_args;
constexpr size_t MulSchema_TTT::max_pos_args;
constexpr char const* MulSchema_TTT::signature;
FunctionDef MulSchema_TTT::function_def = {
/*name*/"mul",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarMulSchema_TTScB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& other, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarMul;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar other, *, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t ScalarMulSchema_TTScB::max_args;
constexpr size_t ScalarMulSchema_TTScB::max_pos_args;
constexpr char const* ScalarMulSchema_TTScB::signature;
FunctionDef ScalarMulSchema_TTScB::function_def = {
/*name*/"mul",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct ScalarMulSchema_TScT {
  using FType = Maybe<one::Tensor> (const Scalar& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarMul;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Scalar input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarMulSchema_TScT::max_args;
constexpr size_t ScalarMulSchema_TScT::max_pos_args;
constexpr char const* ScalarMulSchema_TScT::signature;
FunctionDef ScalarMulSchema_TScT::function_def = {
/*name*/"mul",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* mul(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("mul");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MulSchema_TTT, functional::ScalarMulSchema_TTScB, functional::ScalarMulSchema_TScT> parser("mul");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Mul(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ScalarMul(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>(), r[2].As<bool>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::ScalarMul(r[0].As<Scalar>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct InplaceMulSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::InplaceMul;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t InplaceMulSchema_TTT::max_args;
constexpr size_t InplaceMulSchema_TTT::max_pos_args;
constexpr char const* InplaceMulSchema_TTT::signature;
FunctionDef InplaceMulSchema_TTT::function_def = {
/*name*/"mul_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct InplaceScalarMulSchema_TTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::InplaceScalarMul;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar other)";
  static FunctionDef function_def;
};

constexpr size_t InplaceScalarMulSchema_TTSc::max_args;
constexpr size_t InplaceScalarMulSchema_TTSc::max_pos_args;
constexpr char const* InplaceScalarMulSchema_TTSc::signature;
FunctionDef InplaceScalarMulSchema_TTSc::function_def = {
/*name*/"mul_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* mul_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("mul_");
  PythonFrameGuard pf;
  static PythonArgParser<functional::InplaceMulSchema_TTT, functional::InplaceScalarMulSchema_TTSc> parser("mul_");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::InplaceMul(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::InplaceScalarMul(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AddcmulSchema_TTTTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& tensor1, const std::shared_ptr<one::Tensor>& tensor2, const Scalar& value);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Addcmul;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor tensor1, Tensor tensor2, *, Scalar value=1)";
  static FunctionDef function_def;
};

constexpr size_t AddcmulSchema_TTTTSc::max_args;
constexpr size_t AddcmulSchema_TTTTSc::max_pos_args;
constexpr char const* AddcmulSchema_TTTTSc::signature;
FunctionDef AddcmulSchema_TTTTSc::function_def = {
/*name*/"addcmul",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"tensor1", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"tensor2", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"value", /*default_value*/Scalar(1), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* addcmul(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("addcmul");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AddcmulSchema_TTTTSc> parser("addcmul");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Addcmul(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<Scalar>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct InplaceAddcmulSchema_TTTTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& tensor1, const std::shared_ptr<one::Tensor>& tensor2, const Scalar& value);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::InplaceAddcmul;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor tensor1, Tensor tensor2, *, Scalar value=1)";
  static FunctionDef function_def;
};

constexpr size_t InplaceAddcmulSchema_TTTTSc::max_args;
constexpr size_t InplaceAddcmulSchema_TTTTSc::max_pos_args;
constexpr char const* InplaceAddcmulSchema_TTTTSc::signature;
FunctionDef InplaceAddcmulSchema_TTTTSc::function_def = {
/*name*/"addcmul_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"tensor1", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"tensor2", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"value", /*default_value*/Scalar(1), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* addcmul_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("addcmul_");
  PythonFrameGuard pf;
  static PythonArgParser<functional::InplaceAddcmulSchema_TTTTSc> parser("addcmul_");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::InplaceAddcmul(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<Scalar>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AddCDivSchema_TTTTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& tensor1, const std::shared_ptr<one::Tensor>& tensor2, const Scalar& value);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::AddCDiv;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor tensor1, Tensor tensor2, *, Scalar value=1)";
  static FunctionDef function_def;
};

constexpr size_t AddCDivSchema_TTTTSc::max_args;
constexpr size_t AddCDivSchema_TTTTSc::max_pos_args;
constexpr char const* AddCDivSchema_TTTTSc::signature;
FunctionDef AddCDivSchema_TTTTSc::function_def = {
/*name*/"addcdiv",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"tensor1", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"tensor2", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"value", /*default_value*/Scalar(1), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* addcdiv(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("addcdiv");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AddCDivSchema_TTTTSc> parser("addcdiv");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::AddCDiv(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<Scalar>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct InplaceAddCDivSchema_TTTTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& tensor1, const std::shared_ptr<one::Tensor>& tensor2, const Scalar& value);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::InplaceAddCDiv;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor tensor1, Tensor tensor2, *, Scalar value=1)";
  static FunctionDef function_def;
};

constexpr size_t InplaceAddCDivSchema_TTTTSc::max_args;
constexpr size_t InplaceAddCDivSchema_TTTTSc::max_pos_args;
constexpr char const* InplaceAddCDivSchema_TTTTSc::signature;
FunctionDef InplaceAddCDivSchema_TTTTSc::function_def = {
/*name*/"addcdiv_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"tensor1", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"tensor2", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"value", /*default_value*/Scalar(1), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* addcdiv_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("addcdiv_");
  PythonFrameGuard pf;
  static PythonArgParser<functional::InplaceAddCDivSchema_TTTTSc> parser("addcdiv_");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::InplaceAddCDiv(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<Scalar>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct DivSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Div;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t DivSchema_TTT::max_args;
constexpr size_t DivSchema_TTT::max_pos_args;
constexpr char const* DivSchema_TTT::signature;
FunctionDef DivSchema_TTT::function_def = {
/*name*/"div",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarDivSchema_TTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarDiv;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarDivSchema_TTSc::max_args;
constexpr size_t ScalarDivSchema_TTSc::max_pos_args;
constexpr char const* ScalarDivSchema_TTSc::signature;
FunctionDef ScalarDivSchema_TTSc::function_def = {
/*name*/"div",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarDivSchema_TScT {
  using FType = Maybe<one::Tensor> (const Scalar& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarDiv;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Scalar input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarDivSchema_TScT::max_args;
constexpr size_t ScalarDivSchema_TScT::max_pos_args;
constexpr char const* ScalarDivSchema_TScT::signature;
FunctionDef ScalarDivSchema_TScT::function_def = {
/*name*/"div",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* div(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("div");
  PythonFrameGuard pf;
  static PythonArgParser<functional::DivSchema_TTT, functional::ScalarDivSchema_TTSc, functional::ScalarDivSchema_TScT> parser("div");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Div(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ScalarDiv(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::ScalarDiv(r[0].As<Scalar>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct InplaceDivSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::InplaceDiv;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t InplaceDivSchema_TTT::max_args;
constexpr size_t InplaceDivSchema_TTT::max_pos_args;
constexpr char const* InplaceDivSchema_TTT::signature;
FunctionDef InplaceDivSchema_TTT::function_def = {
/*name*/"div_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct InplaceScalarDivSchema_TTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::InplaceScalarDiv;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar other)";
  static FunctionDef function_def;
};

constexpr size_t InplaceScalarDivSchema_TTSc::max_args;
constexpr size_t InplaceScalarDivSchema_TTSc::max_pos_args;
constexpr char const* InplaceScalarDivSchema_TTSc::signature;
FunctionDef InplaceScalarDivSchema_TTSc::function_def = {
/*name*/"div_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* div_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("div_");
  PythonFrameGuard pf;
  static PythonArgParser<functional::InplaceDivSchema_TTT, functional::InplaceScalarDivSchema_TTSc> parser("div_");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::InplaceDiv(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::InplaceScalarDiv(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BroadcastEqualSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BroadcastEqual;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t BroadcastEqualSchema_TTT::max_args;
constexpr size_t BroadcastEqualSchema_TTT::max_pos_args;
constexpr char const* BroadcastEqualSchema_TTT::signature;
FunctionDef BroadcastEqualSchema_TTT::function_def = {
/*name*/"equal",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarLogicalEqualSchema_TTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarLogicalEqual;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarLogicalEqualSchema_TTSc::max_args;
constexpr size_t ScalarLogicalEqualSchema_TTSc::max_pos_args;
constexpr char const* ScalarLogicalEqualSchema_TTSc::signature;
FunctionDef ScalarLogicalEqualSchema_TTSc::function_def = {
/*name*/"equal",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarLogicalEqualSchema_TScT {
  using FType = Maybe<one::Tensor> (const Scalar& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarLogicalEqual;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Scalar input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarLogicalEqualSchema_TScT::max_args;
constexpr size_t ScalarLogicalEqualSchema_TScT::max_pos_args;
constexpr char const* ScalarLogicalEqualSchema_TScT::signature;
FunctionDef ScalarLogicalEqualSchema_TScT::function_def = {
/*name*/"equal",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* equal(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("equal");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BroadcastEqualSchema_TTT, functional::ScalarLogicalEqualSchema_TTSc, functional::ScalarLogicalEqualSchema_TScT> parser("equal");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BroadcastEqual(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ScalarLogicalEqual(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::ScalarLogicalEqual(r[0].As<Scalar>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BroadcastNotEqualSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BroadcastNotEqual;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t BroadcastNotEqualSchema_TTT::max_args;
constexpr size_t BroadcastNotEqualSchema_TTT::max_pos_args;
constexpr char const* BroadcastNotEqualSchema_TTT::signature;
FunctionDef BroadcastNotEqualSchema_TTT::function_def = {
/*name*/"not_equal",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarLogicalNotEqualSchema_TTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarLogicalNotEqual;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarLogicalNotEqualSchema_TTSc::max_args;
constexpr size_t ScalarLogicalNotEqualSchema_TTSc::max_pos_args;
constexpr char const* ScalarLogicalNotEqualSchema_TTSc::signature;
FunctionDef ScalarLogicalNotEqualSchema_TTSc::function_def = {
/*name*/"not_equal",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarLogicalNotEqualSchema_TScT {
  using FType = Maybe<one::Tensor> (const Scalar& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarLogicalNotEqual;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Scalar input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarLogicalNotEqualSchema_TScT::max_args;
constexpr size_t ScalarLogicalNotEqualSchema_TScT::max_pos_args;
constexpr char const* ScalarLogicalNotEqualSchema_TScT::signature;
FunctionDef ScalarLogicalNotEqualSchema_TScT::function_def = {
/*name*/"not_equal",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* not_equal(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("not_equal");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BroadcastNotEqualSchema_TTT, functional::ScalarLogicalNotEqualSchema_TTSc, functional::ScalarLogicalNotEqualSchema_TScT> parser("not_equal");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BroadcastNotEqual(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ScalarLogicalNotEqual(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::ScalarLogicalNotEqual(r[0].As<Scalar>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BroadcastGreaterSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BroadcastGreater;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t BroadcastGreaterSchema_TTT::max_args;
constexpr size_t BroadcastGreaterSchema_TTT::max_pos_args;
constexpr char const* BroadcastGreaterSchema_TTT::signature;
FunctionDef BroadcastGreaterSchema_TTT::function_def = {
/*name*/"greater",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarLogicalGreaterSchema_TTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarLogicalGreater;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarLogicalGreaterSchema_TTSc::max_args;
constexpr size_t ScalarLogicalGreaterSchema_TTSc::max_pos_args;
constexpr char const* ScalarLogicalGreaterSchema_TTSc::signature;
FunctionDef ScalarLogicalGreaterSchema_TTSc::function_def = {
/*name*/"greater",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarLogicalGreaterSchema_TScT {
  using FType = Maybe<one::Tensor> (const Scalar& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarLogicalGreater;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Scalar input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarLogicalGreaterSchema_TScT::max_args;
constexpr size_t ScalarLogicalGreaterSchema_TScT::max_pos_args;
constexpr char const* ScalarLogicalGreaterSchema_TScT::signature;
FunctionDef ScalarLogicalGreaterSchema_TScT::function_def = {
/*name*/"greater",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* greater(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("greater");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BroadcastGreaterSchema_TTT, functional::ScalarLogicalGreaterSchema_TTSc, functional::ScalarLogicalGreaterSchema_TScT> parser("greater");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BroadcastGreater(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ScalarLogicalGreater(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::ScalarLogicalGreater(r[0].As<Scalar>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BroadcastGreaterEqualSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BroadcastGreaterEqual;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t BroadcastGreaterEqualSchema_TTT::max_args;
constexpr size_t BroadcastGreaterEqualSchema_TTT::max_pos_args;
constexpr char const* BroadcastGreaterEqualSchema_TTT::signature;
FunctionDef BroadcastGreaterEqualSchema_TTT::function_def = {
/*name*/"greater_equal",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarLogicalGreaterEqualSchema_TTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarLogicalGreaterEqual;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarLogicalGreaterEqualSchema_TTSc::max_args;
constexpr size_t ScalarLogicalGreaterEqualSchema_TTSc::max_pos_args;
constexpr char const* ScalarLogicalGreaterEqualSchema_TTSc::signature;
FunctionDef ScalarLogicalGreaterEqualSchema_TTSc::function_def = {
/*name*/"greater_equal",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarLogicalGreaterEqualSchema_TScT {
  using FType = Maybe<one::Tensor> (const Scalar& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarLogicalGreaterEqual;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Scalar input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarLogicalGreaterEqualSchema_TScT::max_args;
constexpr size_t ScalarLogicalGreaterEqualSchema_TScT::max_pos_args;
constexpr char const* ScalarLogicalGreaterEqualSchema_TScT::signature;
FunctionDef ScalarLogicalGreaterEqualSchema_TScT::function_def = {
/*name*/"greater_equal",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* greater_equal(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("greater_equal");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BroadcastGreaterEqualSchema_TTT, functional::ScalarLogicalGreaterEqualSchema_TTSc, functional::ScalarLogicalGreaterEqualSchema_TScT> parser("greater_equal");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BroadcastGreaterEqual(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ScalarLogicalGreaterEqual(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::ScalarLogicalGreaterEqual(r[0].As<Scalar>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BroadcastLogicalAndSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BroadcastLogicalAnd;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t BroadcastLogicalAndSchema_TTT::max_args;
constexpr size_t BroadcastLogicalAndSchema_TTT::max_pos_args;
constexpr char const* BroadcastLogicalAndSchema_TTT::signature;
FunctionDef BroadcastLogicalAndSchema_TTT::function_def = {
/*name*/"logical_and",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarLogicalAndSchema_TTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarLogicalAnd;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarLogicalAndSchema_TTSc::max_args;
constexpr size_t ScalarLogicalAndSchema_TTSc::max_pos_args;
constexpr char const* ScalarLogicalAndSchema_TTSc::signature;
FunctionDef ScalarLogicalAndSchema_TTSc::function_def = {
/*name*/"logical_and",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarLogicalAndSchema_TScT {
  using FType = Maybe<one::Tensor> (const Scalar& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarLogicalAnd;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Scalar input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarLogicalAndSchema_TScT::max_args;
constexpr size_t ScalarLogicalAndSchema_TScT::max_pos_args;
constexpr char const* ScalarLogicalAndSchema_TScT::signature;
FunctionDef ScalarLogicalAndSchema_TScT::function_def = {
/*name*/"logical_and",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* logical_and(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("logical_and");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BroadcastLogicalAndSchema_TTT, functional::ScalarLogicalAndSchema_TTSc, functional::ScalarLogicalAndSchema_TScT> parser("logical_and");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BroadcastLogicalAnd(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ScalarLogicalAnd(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::ScalarLogicalAnd(r[0].As<Scalar>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BroadcastLogicalOrSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BroadcastLogicalOr;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t BroadcastLogicalOrSchema_TTT::max_args;
constexpr size_t BroadcastLogicalOrSchema_TTT::max_pos_args;
constexpr char const* BroadcastLogicalOrSchema_TTT::signature;
FunctionDef BroadcastLogicalOrSchema_TTT::function_def = {
/*name*/"logical_or",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarLogicalOrSchema_TTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarLogicalOr;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarLogicalOrSchema_TTSc::max_args;
constexpr size_t ScalarLogicalOrSchema_TTSc::max_pos_args;
constexpr char const* ScalarLogicalOrSchema_TTSc::signature;
FunctionDef ScalarLogicalOrSchema_TTSc::function_def = {
/*name*/"logical_or",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarLogicalOrSchema_TScT {
  using FType = Maybe<one::Tensor> (const Scalar& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarLogicalOr;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Scalar input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarLogicalOrSchema_TScT::max_args;
constexpr size_t ScalarLogicalOrSchema_TScT::max_pos_args;
constexpr char const* ScalarLogicalOrSchema_TScT::signature;
FunctionDef ScalarLogicalOrSchema_TScT::function_def = {
/*name*/"logical_or",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* logical_or(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("logical_or");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BroadcastLogicalOrSchema_TTT, functional::ScalarLogicalOrSchema_TTSc, functional::ScalarLogicalOrSchema_TScT> parser("logical_or");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BroadcastLogicalOr(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ScalarLogicalOr(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::ScalarLogicalOr(r[0].As<Scalar>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct LogicalNotSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::LogicalNot;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input)";
  static FunctionDef function_def;
};

constexpr size_t LogicalNotSchema_TT::max_args;
constexpr size_t LogicalNotSchema_TT::max_pos_args;
constexpr char const* LogicalNotSchema_TT::signature;
FunctionDef LogicalNotSchema_TT::function_def = {
/*name*/"logical_not",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* logical_not(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("logical_not");
  PythonFrameGuard pf;
  static PythonArgParser<functional::LogicalNotSchema_TT> parser("logical_not");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::LogicalNot(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BroadcastLogicalXorSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BroadcastLogicalXor;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t BroadcastLogicalXorSchema_TTT::max_args;
constexpr size_t BroadcastLogicalXorSchema_TTT::max_pos_args;
constexpr char const* BroadcastLogicalXorSchema_TTT::signature;
FunctionDef BroadcastLogicalXorSchema_TTT::function_def = {
/*name*/"logical_xor",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarLogicalXorSchema_TTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarLogicalXor;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarLogicalXorSchema_TTSc::max_args;
constexpr size_t ScalarLogicalXorSchema_TTSc::max_pos_args;
constexpr char const* ScalarLogicalXorSchema_TTSc::signature;
FunctionDef ScalarLogicalXorSchema_TTSc::function_def = {
/*name*/"logical_xor",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarLogicalXorSchema_TScT {
  using FType = Maybe<one::Tensor> (const Scalar& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarLogicalXor;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Scalar input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarLogicalXorSchema_TScT::max_args;
constexpr size_t ScalarLogicalXorSchema_TScT::max_pos_args;
constexpr char const* ScalarLogicalXorSchema_TScT::signature;
FunctionDef ScalarLogicalXorSchema_TScT::function_def = {
/*name*/"logical_xor",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* logical_xor(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("logical_xor");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BroadcastLogicalXorSchema_TTT, functional::ScalarLogicalXorSchema_TTSc, functional::ScalarLogicalXorSchema_TScT> parser("logical_xor");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BroadcastLogicalXor(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ScalarLogicalXor(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::ScalarLogicalXor(r[0].As<Scalar>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BroadcastLessSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BroadcastLess;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t BroadcastLessSchema_TTT::max_args;
constexpr size_t BroadcastLessSchema_TTT::max_pos_args;
constexpr char const* BroadcastLessSchema_TTT::signature;
FunctionDef BroadcastLessSchema_TTT::function_def = {
/*name*/"less",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarLogicalLessSchema_TTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarLogicalLess;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarLogicalLessSchema_TTSc::max_args;
constexpr size_t ScalarLogicalLessSchema_TTSc::max_pos_args;
constexpr char const* ScalarLogicalLessSchema_TTSc::signature;
FunctionDef ScalarLogicalLessSchema_TTSc::function_def = {
/*name*/"less",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarLogicalLessSchema_TScT {
  using FType = Maybe<one::Tensor> (const Scalar& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarLogicalLess;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Scalar input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarLogicalLessSchema_TScT::max_args;
constexpr size_t ScalarLogicalLessSchema_TScT::max_pos_args;
constexpr char const* ScalarLogicalLessSchema_TScT::signature;
FunctionDef ScalarLogicalLessSchema_TScT::function_def = {
/*name*/"less",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* less(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("less");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BroadcastLessSchema_TTT, functional::ScalarLogicalLessSchema_TTSc, functional::ScalarLogicalLessSchema_TScT> parser("less");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BroadcastLess(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ScalarLogicalLess(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::ScalarLogicalLess(r[0].As<Scalar>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BroadcastLessEqualSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BroadcastLessEqual;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t BroadcastLessEqualSchema_TTT::max_args;
constexpr size_t BroadcastLessEqualSchema_TTT::max_pos_args;
constexpr char const* BroadcastLessEqualSchema_TTT::signature;
FunctionDef BroadcastLessEqualSchema_TTT::function_def = {
/*name*/"less_equal",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarLogicalLessEqualSchema_TTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarLogicalLessEqual;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarLogicalLessEqualSchema_TTSc::max_args;
constexpr size_t ScalarLogicalLessEqualSchema_TTSc::max_pos_args;
constexpr char const* ScalarLogicalLessEqualSchema_TTSc::signature;
FunctionDef ScalarLogicalLessEqualSchema_TTSc::function_def = {
/*name*/"less_equal",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarLogicalLessEqualSchema_TScT {
  using FType = Maybe<one::Tensor> (const Scalar& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarLogicalLessEqual;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Scalar input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarLogicalLessEqualSchema_TScT::max_args;
constexpr size_t ScalarLogicalLessEqualSchema_TScT::max_pos_args;
constexpr char const* ScalarLogicalLessEqualSchema_TScT::signature;
FunctionDef ScalarLogicalLessEqualSchema_TScT::function_def = {
/*name*/"less_equal",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* less_equal(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("less_equal");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BroadcastLessEqualSchema_TTT, functional::ScalarLogicalLessEqualSchema_TTSc, functional::ScalarLogicalLessEqualSchema_TScT> parser("less_equal");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BroadcastLessEqual(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ScalarLogicalLessEqual(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::ScalarLogicalLessEqual(r[0].As<Scalar>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct PowSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& exponent);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Pow;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor exponent)";
  static FunctionDef function_def;
};

constexpr size_t PowSchema_TTT::max_args;
constexpr size_t PowSchema_TTT::max_pos_args;
constexpr char const* PowSchema_TTT::signature;
FunctionDef PowSchema_TTT::function_def = {
/*name*/"pow",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"exponent", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarPowSchema_TTScB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& exponent, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarPow;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar exponent, *, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t ScalarPowSchema_TTScB::max_args;
constexpr size_t ScalarPowSchema_TTScB::max_pos_args;
constexpr char const* ScalarPowSchema_TTScB::signature;
FunctionDef ScalarPowSchema_TTScB::function_def = {
/*name*/"pow",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"exponent", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct ScalarPowSchema_TTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& exponent);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarPow;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar exponent)";
  static FunctionDef function_def;
};

constexpr size_t ScalarPowSchema_TTSc::max_args;
constexpr size_t ScalarPowSchema_TTSc::max_pos_args;
constexpr char const* ScalarPowSchema_TTSc::signature;
FunctionDef ScalarPowSchema_TTSc::function_def = {
/*name*/"pow",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"exponent", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarReversePowSchema_TScT {
  using FType = Maybe<one::Tensor> (const Scalar& exponent, const std::shared_ptr<one::Tensor>& input);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarReversePow;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Scalar exponent, Tensor input)";
  static FunctionDef function_def;
};

constexpr size_t ScalarReversePowSchema_TScT::max_args;
constexpr size_t ScalarReversePowSchema_TScT::max_pos_args;
constexpr char const* ScalarReversePowSchema_TScT::signature;
FunctionDef ScalarReversePowSchema_TScT::function_def = {
/*name*/"pow",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"exponent", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* pow(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("pow");
  PythonFrameGuard pf;
  static PythonArgParser<functional::PowSchema_TTT, functional::ScalarPowSchema_TTScB, functional::ScalarPowSchema_TTSc, functional::ScalarReversePowSchema_TScT> parser("pow");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Pow(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ScalarPow(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>(), r[2].As<bool>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::ScalarPow(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>()));
  }
  if (idx == 3) {
    return CastToPyObject(functional::ScalarReversePow(r[0].As<Scalar>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SearchSortedSchema_TTTBB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& sorted_sequence, const std::shared_ptr<one::Tensor>& values, bool out_int32, bool right);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::SearchSorted;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor sorted_sequence, Tensor values, Bool out_int32=False, Bool right=False)";
  static FunctionDef function_def;
};

constexpr size_t SearchSortedSchema_TTTBB::max_args;
constexpr size_t SearchSortedSchema_TTTBB::max_pos_args;
constexpr char const* SearchSortedSchema_TTTBB::signature;
FunctionDef SearchSortedSchema_TTTBB::function_def = {
/*name*/"searchsorted",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"sorted_sequence", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"values", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"out_int32", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"right", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct SearchSortedScalarSchema_TTScBB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& sorted_sequence, const Scalar& values, bool out_int32, bool right);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::SearchSortedScalar;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor sorted_sequence, Scalar values, Bool out_int32=False, Bool right=False)";
  static FunctionDef function_def;
};

constexpr size_t SearchSortedScalarSchema_TTScBB::max_args;
constexpr size_t SearchSortedScalarSchema_TTScBB::max_pos_args;
constexpr char const* SearchSortedScalarSchema_TTScBB::signature;
FunctionDef SearchSortedScalarSchema_TTScBB::function_def = {
/*name*/"searchsorted",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"sorted_sequence", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"values", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"out_int32", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"right", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* searchsorted(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("searchsorted");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SearchSortedSchema_TTTBB, functional::SearchSortedScalarSchema_TTScBB> parser("searchsorted");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::SearchSorted(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<bool>(), r[3].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::SearchSortedScalar(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>(), r[2].As<bool>(), r[3].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct FloorDivSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::FloorDiv;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t FloorDivSchema_TTT::max_args;
constexpr size_t FloorDivSchema_TTT::max_pos_args;
constexpr char const* FloorDivSchema_TTT::signature;
FunctionDef FloorDivSchema_TTT::function_def = {
/*name*/"floor_divide",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarFloorDivSchema_TTScB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& other, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarFloorDiv;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar other, *, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t ScalarFloorDivSchema_TTScB::max_args;
constexpr size_t ScalarFloorDivSchema_TTScB::max_pos_args;
constexpr char const* ScalarFloorDivSchema_TTScB::signature;
FunctionDef ScalarFloorDivSchema_TTScB::function_def = {
/*name*/"floor_divide",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct ScalarFloorDivSchema_TTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarFloorDiv;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarFloorDivSchema_TTSc::max_args;
constexpr size_t ScalarFloorDivSchema_TTSc::max_pos_args;
constexpr char const* ScalarFloorDivSchema_TTSc::signature;
FunctionDef ScalarFloorDivSchema_TTSc::function_def = {
/*name*/"floor_divide",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* floor_divide(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("floor_divide");
  PythonFrameGuard pf;
  static PythonArgParser<functional::FloorDivSchema_TTT, functional::ScalarFloorDivSchema_TTScB, functional::ScalarFloorDivSchema_TTSc> parser("floor_divide");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::FloorDiv(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ScalarFloorDiv(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>(), r[2].As<bool>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::ScalarFloorDiv(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TruncDivSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::TruncDiv;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t TruncDivSchema_TTT::max_args;
constexpr size_t TruncDivSchema_TTT::max_pos_args;
constexpr char const* TruncDivSchema_TTT::signature;
FunctionDef TruncDivSchema_TTT::function_def = {
/*name*/"trunc_divide",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarTruncDivSchema_TTScB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& other, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarTruncDiv;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar other, *, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t ScalarTruncDivSchema_TTScB::max_args;
constexpr size_t ScalarTruncDivSchema_TTScB::max_pos_args;
constexpr char const* ScalarTruncDivSchema_TTScB::signature;
FunctionDef ScalarTruncDivSchema_TTScB::function_def = {
/*name*/"trunc_divide",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* trunc_divide(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("trunc_divide");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TruncDivSchema_TTT, functional::ScalarTruncDivSchema_TTScB> parser("trunc_divide");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::TruncDiv(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ScalarTruncDiv(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MaxSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Max;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input)";
  static FunctionDef function_def;
};

constexpr size_t MaxSchema_TT::max_args;
constexpr size_t MaxSchema_TT::max_pos_args;
constexpr char const* MaxSchema_TT::signature;
FunctionDef MaxSchema_TT::function_def = {
/*name*/"max",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct MaxSchema_TtTI32B {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, int32_t dim, bool keepdim);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::Max;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "TensorTuple (Tensor input, Int32 dim, Bool keepdim=False)";
  static FunctionDef function_def;
};

constexpr size_t MaxSchema_TtTI32B::max_args;
constexpr size_t MaxSchema_TtTI32B::max_pos_args;
constexpr char const* MaxSchema_TtTI32B::signature;
FunctionDef MaxSchema_TtTI32B::function_def = {
/*name*/"max",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct MaxSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Max;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t MaxSchema_TTT::max_args;
constexpr size_t MaxSchema_TTT::max_pos_args;
constexpr char const* MaxSchema_TTT::signature;
FunctionDef MaxSchema_TTT::function_def = {
/*name*/"max",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* max(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("max");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MaxSchema_TT, functional::MaxSchema_TtTI32B, functional::MaxSchema_TTT> parser("max");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Max(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {

  const auto& tensortuple = functional::Max(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>(), r[2].As<bool>()).GetOrThrow();



  static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indice", ""},  {nullptr} }; 
  static PyTypeObject MaxNamedTuple1; 
  static bool is_initialized = false; 
  static PyStructSequence_Desc desc = { "oneflow.return_types.max", nullptr, NamedTuple_fields, 2 }; 
  if (!is_initialized) { 
      PyStructSequence_InitType(&MaxNamedTuple1, &desc); 
      MaxNamedTuple1.tp_repr = (reprfunc)returned_structseq_repr; 
      is_initialized = true; 
  }

  // PyObjectPtr r (PyStructSequence_New(tensortuple.size()));
  PyObjectPtr r (PyStructSequence_New(&MaxNamedTuple1));
  if (!r) {
    // throw python_error();
  }
  for (int i = 0; i < tensortuple.size(); i++) {
    PyTuple_SET_ITEM(r.get(), i, CastToPyObject(tensortuple.at(i)));
  }
  // return (PyObject*)&MaxNamedTuple1; 
  return r.release(); 
  }
  if (idx == 2) {
    return CastToPyObject(functional::Max(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MinSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Min;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input)";
  static FunctionDef function_def;
};

constexpr size_t MinSchema_TT::max_args;
constexpr size_t MinSchema_TT::max_pos_args;
constexpr char const* MinSchema_TT::signature;
FunctionDef MinSchema_TT::function_def = {
/*name*/"min",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct MinSchema_TtTI32B {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, int32_t dim, bool keepdim);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::Min;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "TensorTuple (Tensor input, Int32 dim, Bool keepdim=False)";
  static FunctionDef function_def;
};

constexpr size_t MinSchema_TtTI32B::max_args;
constexpr size_t MinSchema_TtTI32B::max_pos_args;
constexpr char const* MinSchema_TtTI32B::signature;
FunctionDef MinSchema_TtTI32B::function_def = {
/*name*/"min",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct MinSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Min;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t MinSchema_TTT::max_args;
constexpr size_t MinSchema_TTT::max_pos_args;
constexpr char const* MinSchema_TTT::signature;
FunctionDef MinSchema_TTT::function_def = {
/*name*/"min",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* min(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("min");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MinSchema_TT, functional::MinSchema_TtTI32B, functional::MinSchema_TTT> parser("min");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Min(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::Min(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>(), r[2].As<bool>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::Min(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MedianSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Median;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input)";
  static FunctionDef function_def;
};

constexpr size_t MedianSchema_TT::max_args;
constexpr size_t MedianSchema_TT::max_pos_args;
constexpr char const* MedianSchema_TT::signature;
FunctionDef MedianSchema_TT::function_def = {
/*name*/"median",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct MedianWithIndicesSchema_TtTI32B {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, int32_t dim, bool keepdim);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::MedianWithIndices;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "TensorTuple (Tensor input, Int32 dim=-1, Bool keepdim=False)";
  static FunctionDef function_def;
};

constexpr size_t MedianWithIndicesSchema_TtTI32B::max_args;
constexpr size_t MedianWithIndicesSchema_TtTI32B::max_pos_args;
constexpr char const* MedianWithIndicesSchema_TtTI32B::signature;
FunctionDef MedianWithIndicesSchema_TtTI32B::function_def = {
/*name*/"median",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/int32_t(-1), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* median(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("median");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MedianSchema_TT, functional::MedianWithIndicesSchema_TtTI32B> parser("median");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Median(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::MedianWithIndices(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ReduceMaxSchema_TTI32lB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& axis, bool keepdim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ReduceMax;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Int32List axis, Bool keepdim=False)";
  static FunctionDef function_def;
};

constexpr size_t ReduceMaxSchema_TTI32lB::max_args;
constexpr size_t ReduceMaxSchema_TTI32lB::max_pos_args;
constexpr char const* ReduceMaxSchema_TTI32lB::signature;
FunctionDef ReduceMaxSchema_TTI32lB::function_def = {
/*name*/"reduce_max",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"axis", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* reduce_max(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("reduce_max");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ReduceMaxSchema_TTI32lB> parser("reduce_max");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ReduceMax(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ReduceMinSchema_TTI32lB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& axis, bool keepdim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ReduceMin;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Int32List axis, Bool keepdim=False)";
  static FunctionDef function_def;
};

constexpr size_t ReduceMinSchema_TTI32lB::max_args;
constexpr size_t ReduceMinSchema_TTI32lB::max_pos_args;
constexpr char const* ReduceMinSchema_TTI32lB::signature;
FunctionDef ReduceMinSchema_TTI32lB::function_def = {
/*name*/"reduce_min",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"axis", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* reduce_min(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("reduce_min");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ReduceMinSchema_TTI32lB> parser("reduce_min");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ReduceMin(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ReduceSumSchema_TTI32lB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& dim, bool keepdim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ReduceSum;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Int32List dim, Bool keepdim=False)";
  static FunctionDef function_def;
};

constexpr size_t ReduceSumSchema_TTI32lB::max_args;
constexpr size_t ReduceSumSchema_TTI32lB::max_pos_args;
constexpr char const* ReduceSumSchema_TTI32lB::signature;
FunctionDef ReduceSumSchema_TTI32lB::function_def = {
/*name*/"reduce_sum",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ReduceSumWholeSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ReduceSumWhole;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t ReduceSumWholeSchema_TT::max_args;
constexpr size_t ReduceSumWholeSchema_TT::max_pos_args;
constexpr char const* ReduceSumWholeSchema_TT::signature;
FunctionDef ReduceSumWholeSchema_TT::function_def = {
/*name*/"reduce_sum",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* reduce_sum(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("reduce_sum");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ReduceSumSchema_TTI32lB, functional::ReduceSumWholeSchema_TT> parser("reduce_sum");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ReduceSum(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ReduceSumWhole(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ReduceNanSumSchema_TTI32lBDt {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ReduceNanSum;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Int32List dim, Bool keepdim=False, *, DataType dtype=None)";
  static FunctionDef function_def;
};

constexpr size_t ReduceNanSumSchema_TTI32lBDt::max_args;
constexpr size_t ReduceNanSumSchema_TTI32lBDt::max_pos_args;
constexpr char const* ReduceNanSumSchema_TTI32lBDt::signature;
FunctionDef ReduceNanSumSchema_TTI32lBDt::function_def = {
/*name*/"reduce_nansum",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

struct ReduceNanSumWholeSchema_TTDt {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Optional<Symbol<DType>>& dtype);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ReduceNanSumWhole;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input, *, DataType dtype=None)";
  static FunctionDef function_def;
};

constexpr size_t ReduceNanSumWholeSchema_TTDt::max_args;
constexpr size_t ReduceNanSumWholeSchema_TTDt::max_pos_args;
constexpr char const* ReduceNanSumWholeSchema_TTDt::signature;
FunctionDef ReduceNanSumWholeSchema_TTDt::function_def = {
/*name*/"reduce_nansum",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

PyObject* reduce_nansum(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("reduce_nansum");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ReduceNanSumSchema_TTI32lBDt, functional::ReduceNanSumWholeSchema_TTDt> parser("reduce_nansum");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ReduceNanSum(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<bool>(), r[3].As<Optional<Symbol<DType>>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ReduceNanSumWhole(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<Symbol<DType>>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ReduceMeanSchema_TTI32lB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& dim, bool keepdim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ReduceMean;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Int32List dim, Bool keepdim=False)";
  static FunctionDef function_def;
};

constexpr size_t ReduceMeanSchema_TTI32lB::max_args;
constexpr size_t ReduceMeanSchema_TTI32lB::max_pos_args;
constexpr char const* ReduceMeanSchema_TTI32lB::signature;
FunctionDef ReduceMeanSchema_TTI32lB::function_def = {
/*name*/"reduce_mean",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ReduceMeanWholeSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ReduceMeanWhole;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t ReduceMeanWholeSchema_TT::max_args;
constexpr size_t ReduceMeanWholeSchema_TT::max_pos_args;
constexpr char const* ReduceMeanWholeSchema_TT::signature;
FunctionDef ReduceMeanWholeSchema_TT::function_def = {
/*name*/"reduce_mean",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* reduce_mean(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("reduce_mean");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ReduceMeanSchema_TTI32lB, functional::ReduceMeanWholeSchema_TT> parser("reduce_mean");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ReduceMean(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ReduceMeanWhole(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ReduceAllSchema_TTI32lB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& dim, bool keepdim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ReduceAll;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Int32List dim, Bool keepdim=False)";
  static FunctionDef function_def;
};

constexpr size_t ReduceAllSchema_TTI32lB::max_args;
constexpr size_t ReduceAllSchema_TTI32lB::max_pos_args;
constexpr char const* ReduceAllSchema_TTI32lB::signature;
FunctionDef ReduceAllSchema_TTI32lB::function_def = {
/*name*/"reduce_all",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ReduceAllWholeSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ReduceAllWhole;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t ReduceAllWholeSchema_TT::max_args;
constexpr size_t ReduceAllWholeSchema_TT::max_pos_args;
constexpr char const* ReduceAllWholeSchema_TT::signature;
FunctionDef ReduceAllWholeSchema_TT::function_def = {
/*name*/"reduce_all",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* reduce_all(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("reduce_all");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ReduceAllSchema_TTI32lB, functional::ReduceAllWholeSchema_TT> parser("reduce_all");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ReduceAll(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ReduceAllWhole(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ReduceAnySchema_TTI32lB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& dim, bool keepdim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ReduceAny;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Int32List dim, Bool keepdim=False)";
  static FunctionDef function_def;
};

constexpr size_t ReduceAnySchema_TTI32lB::max_args;
constexpr size_t ReduceAnySchema_TTI32lB::max_pos_args;
constexpr char const* ReduceAnySchema_TTI32lB::signature;
FunctionDef ReduceAnySchema_TTI32lB::function_def = {
/*name*/"reduce_any",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ReduceAnyWholeSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ReduceAnyWhole;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t ReduceAnyWholeSchema_TT::max_args;
constexpr size_t ReduceAnyWholeSchema_TT::max_pos_args;
constexpr char const* ReduceAnyWholeSchema_TT::signature;
FunctionDef ReduceAnyWholeSchema_TT::function_def = {
/*name*/"reduce_any",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* reduce_any(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("reduce_any");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ReduceAnySchema_TTI32lB, functional::ReduceAnyWholeSchema_TT> parser("reduce_any");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ReduceAny(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ReduceAnyWhole(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ReduceProdSchema_TTI32lBDt {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ReduceProd;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Int32List dim, Bool keepdim=False, *, DataType dtype=None)";
  static FunctionDef function_def;
};

constexpr size_t ReduceProdSchema_TTI32lBDt::max_args;
constexpr size_t ReduceProdSchema_TTI32lBDt::max_pos_args;
constexpr char const* ReduceProdSchema_TTI32lBDt::signature;
FunctionDef ReduceProdSchema_TTI32lBDt::function_def = {
/*name*/"reduce_prod",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

struct ReduceProdWholeSchema_TTDt {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const Optional<Symbol<DType>>& dtype);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ReduceProdWhole;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x, *, DataType dtype=None)";
  static FunctionDef function_def;
};

constexpr size_t ReduceProdWholeSchema_TTDt::max_args;
constexpr size_t ReduceProdWholeSchema_TTDt::max_pos_args;
constexpr char const* ReduceProdWholeSchema_TTDt::signature;
FunctionDef ReduceProdWholeSchema_TTDt::function_def = {
/*name*/"reduce_prod",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

PyObject* reduce_prod(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("reduce_prod");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ReduceProdSchema_TTI32lBDt, functional::ReduceProdWholeSchema_TTDt> parser("reduce_prod");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ReduceProd(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<bool>(), r[3].As<Optional<Symbol<DType>>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ReduceProdWhole(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<Symbol<DType>>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ReduceMinDeviceStageSchema_TtTI32l {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& in, const std::vector<int32_t>& axis);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::ReduceMinDeviceStage;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "TensorTuple (Tensor in, Int32List axis)";
  static FunctionDef function_def;
};

constexpr size_t ReduceMinDeviceStageSchema_TtTI32l::max_args;
constexpr size_t ReduceMinDeviceStageSchema_TtTI32l::max_pos_args;
constexpr char const* ReduceMinDeviceStageSchema_TtTI32l::signature;
FunctionDef ReduceMinDeviceStageSchema_TtTI32l::function_def = {
/*name*/"reduce_min_device_stage",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"in", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"axis", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* reduce_min_device_stage(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("reduce_min_device_stage");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ReduceMinDeviceStageSchema_TtTI32l> parser("reduce_min_device_stage");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ReduceMinDeviceStage(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ReduceMaxDeviceStageSchema_TtTI32l {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& in, const std::vector<int32_t>& axis);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::ReduceMaxDeviceStage;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "TensorTuple (Tensor in, Int32List axis)";
  static FunctionDef function_def;
};

constexpr size_t ReduceMaxDeviceStageSchema_TtTI32l::max_args;
constexpr size_t ReduceMaxDeviceStageSchema_TtTI32l::max_pos_args;
constexpr char const* ReduceMaxDeviceStageSchema_TtTI32l::signature;
FunctionDef ReduceMaxDeviceStageSchema_TtTI32l::function_def = {
/*name*/"reduce_max_device_stage",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"in", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"axis", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* reduce_max_device_stage(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("reduce_max_device_stage");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ReduceMaxDeviceStageSchema_TtTI32l> parser("reduce_max_device_stage");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ReduceMaxDeviceStage(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ReduceMinGlobalStageSchema_TtTTI32lB {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& device_count, const std::vector<int32_t>& axis, bool keepdims);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::ReduceMinGlobalStage;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "TensorTuple (Tensor in, Tensor device_count, Int32List axis, Bool keepdims=False)";
  static FunctionDef function_def;
};

constexpr size_t ReduceMinGlobalStageSchema_TtTTI32lB::max_args;
constexpr size_t ReduceMinGlobalStageSchema_TtTTI32lB::max_pos_args;
constexpr char const* ReduceMinGlobalStageSchema_TtTTI32lB::signature;
FunctionDef ReduceMinGlobalStageSchema_TtTTI32lB::function_def = {
/*name*/"reduce_min_global_stage",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"in", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"device_count", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"axis", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"keepdims", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* reduce_min_global_stage(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("reduce_min_global_stage");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ReduceMinGlobalStageSchema_TtTTI32lB> parser("reduce_min_global_stage");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ReduceMinGlobalStage(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::vector<int32_t>>(), r[3].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ReduceMaxGlobalStageSchema_TtTTI32lB {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& device_count, const std::vector<int32_t>& axis, bool keepdims);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::ReduceMaxGlobalStage;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "TensorTuple (Tensor in, Tensor device_count, Int32List axis, Bool keepdims=False)";
  static FunctionDef function_def;
};

constexpr size_t ReduceMaxGlobalStageSchema_TtTTI32lB::max_args;
constexpr size_t ReduceMaxGlobalStageSchema_TtTTI32lB::max_pos_args;
constexpr char const* ReduceMaxGlobalStageSchema_TtTTI32lB::signature;
FunctionDef ReduceMaxGlobalStageSchema_TtTTI32lB::function_def = {
/*name*/"reduce_max_global_stage",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"in", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"device_count", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"axis", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"keepdims", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* reduce_max_global_stage(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("reduce_max_global_stage");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ReduceMaxGlobalStageSchema_TtTTI32lB> parser("reduce_max_global_stage");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ReduceMaxGlobalStage(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::vector<int32_t>>(), r[3].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TransposeSchema_TTI32l {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& perm);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Transpose;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Int32List perm)";
  static FunctionDef function_def;
};

constexpr size_t TransposeSchema_TTI32l::max_args;
constexpr size_t TransposeSchema_TTI32l::max_pos_args;
constexpr char const* TransposeSchema_TTI32l::signature;
FunctionDef TransposeSchema_TTI32l::function_def = {
/*name*/"transpose",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"perm", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct Transpose2dimSchema_TTI32I32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int32_t dim0, int32_t dim1);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Transpose2dim;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Int32 dim0, Int32 dim1)";
  static FunctionDef function_def;
};

constexpr size_t Transpose2dimSchema_TTI32I32::max_args;
constexpr size_t Transpose2dimSchema_TTI32I32::max_pos_args;
constexpr char const* Transpose2dimSchema_TTI32I32::signature;
FunctionDef Transpose2dimSchema_TTI32I32::function_def = {
/*name*/"transpose",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim0", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim1", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* transpose(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("transpose");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TransposeSchema_TTI32l, functional::Transpose2dimSchema_TTI32I32> parser("transpose");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Transpose(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::Transpose2dim(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>(), r[2].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AsStridedSchema_TTI32lI32lI32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& size, const std::vector<int32_t>& stride, int32_t storage_offset);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::AsStrided;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor input, Int32List size, Int32List stride, Int32 storage_offset=0)";
  static FunctionDef function_def;
};

constexpr size_t AsStridedSchema_TTI32lI32lI32::max_args;
constexpr size_t AsStridedSchema_TTI32lI32lI32::max_pos_args;
constexpr char const* AsStridedSchema_TTI32lI32lI32::signature;
FunctionDef AsStridedSchema_TTI32lI32lI32::function_def = {
/*name*/"as_strided",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"size", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"stride", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"storage_offset", /*default_value*/int32_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* as_strided(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("as_strided");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AsStridedSchema_TTI32lI32lI32> parser("as_strided");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::AsStrided(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<std::vector<int32_t>>(), r[3].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SelectSchema_TTI32I32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int32_t dim, int32_t index);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Select;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Int32 dim, Int32 index)";
  static FunctionDef function_def;
};

constexpr size_t SelectSchema_TTI32I32::max_args;
constexpr size_t SelectSchema_TTI32I32::max_pos_args;
constexpr char const* SelectSchema_TTI32I32::signature;
FunctionDef SelectSchema_TTI32I32::function_def = {
/*name*/"select",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"index", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* select(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("select");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SelectSchema_TTI32I32> parser("select");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Select(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>(), r[2].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SwapaxesSchema_TTI32I32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int32_t dim0, int32_t dim1);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Swapaxes;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Int32 dim0, Int32 dim1)";
  static FunctionDef function_def;
};

constexpr size_t SwapaxesSchema_TTI32I32::max_args;
constexpr size_t SwapaxesSchema_TTI32I32::max_pos_args;
constexpr char const* SwapaxesSchema_TTI32I32::signature;
FunctionDef SwapaxesSchema_TTI32I32::function_def = {
/*name*/"swapaxes",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim0", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim1", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* swapaxes(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("swapaxes");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SwapaxesSchema_TTI32I32> parser("swapaxes");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Swapaxes(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>(), r[2].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SwapdimsSchema_TTI32I32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int32_t dim0, int32_t dim1);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Swapdims;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Int32 dim0, Int32 dim1)";
  static FunctionDef function_def;
};

constexpr size_t SwapdimsSchema_TTI32I32::max_args;
constexpr size_t SwapdimsSchema_TTI32I32::max_pos_args;
constexpr char const* SwapdimsSchema_TTI32I32::signature;
FunctionDef SwapdimsSchema_TTI32I32::function_def = {
/*name*/"swapdims",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim0", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim1", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* swapdims(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("swapdims");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SwapdimsSchema_TTI32I32> parser("swapdims");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Swapdims(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>(), r[2].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AmaxSchema_TTI32lB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Optional<std::vector<int32_t>>& dim, bool keepdim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Amax;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Int32List dim=None, Bool keepdim=False)";
  static FunctionDef function_def;
};

constexpr size_t AmaxSchema_TTI32lB::max_args;
constexpr size_t AmaxSchema_TTI32lB::max_pos_args;
constexpr char const* AmaxSchema_TTI32lB::signature;
FunctionDef AmaxSchema_TTI32lB::function_def = {
/*name*/"amax",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/Optional<std::vector<int32_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* amax(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("amax");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AmaxSchema_TTI32lB> parser("amax");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Amax(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<std::vector<int32_t>>>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct PermuteSchema_TTI32l {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& dims);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Permute;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Int32List dims)";
  static FunctionDef function_def;
};

constexpr size_t PermuteSchema_TTI32l::max_args;
constexpr size_t PermuteSchema_TTI32l::max_pos_args;
constexpr char const* PermuteSchema_TTI32l::signature;
FunctionDef PermuteSchema_TTI32l::function_def = {
/*name*/"permute",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dims", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* permute(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("permute");
  PythonFrameGuard pf;
  static PythonArgParser<functional::PermuteSchema_TTI32l> parser("permute");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Permute(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TransposeAllDimPropertySchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::TransposeAllDimProperty;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input)";
  static FunctionDef function_def;
};

constexpr size_t TransposeAllDimPropertySchema_TT::max_args;
constexpr size_t TransposeAllDimPropertySchema_TT::max_pos_args;
constexpr char const* TransposeAllDimPropertySchema_TT::signature;
FunctionDef TransposeAllDimPropertySchema_TT::function_def = {
/*name*/"T",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* T(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("T");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TransposeAllDimPropertySchema_TT> parser("T");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::TransposeAllDimProperty(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TransposeAllDimFunctionSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::TransposeAllDimFunction;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input)";
  static FunctionDef function_def;
};

constexpr size_t TransposeAllDimFunctionSchema_TT::max_args;
constexpr size_t TransposeAllDimFunctionSchema_TT::max_pos_args;
constexpr char const* TransposeAllDimFunctionSchema_TT::signature;
FunctionDef TransposeAllDimFunctionSchema_TT::function_def = {
/*name*/"t",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* t(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("t");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TransposeAllDimFunctionSchema_TT> parser("t");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::TransposeAllDimFunction(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ReciprocalSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Reciprocal;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t ReciprocalSchema_TT::max_args;
constexpr size_t ReciprocalSchema_TT::max_pos_args;
constexpr char const* ReciprocalSchema_TT::signature;
FunctionDef ReciprocalSchema_TT::function_def = {
/*name*/"reciprocal",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* reciprocal(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("reciprocal");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ReciprocalSchema_TT> parser("reciprocal");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Reciprocal(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ReciprocalNoNanSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ReciprocalNoNan;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t ReciprocalNoNanSchema_TT::max_args;
constexpr size_t ReciprocalNoNanSchema_TT::max_pos_args;
constexpr char const* ReciprocalNoNanSchema_TT::signature;
FunctionDef ReciprocalNoNanSchema_TT::function_def = {
/*name*/"reciprocal_no_nan",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* reciprocal_no_nan(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("reciprocal_no_nan");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ReciprocalNoNanSchema_TT> parser("reciprocal_no_nan");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ReciprocalNoNan(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ImageFlipSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& flip_code);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ImageFlip;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor flip_code)";
  static FunctionDef function_def;
};

constexpr size_t ImageFlipSchema_TTT::max_args;
constexpr size_t ImageFlipSchema_TTT::max_pos_args;
constexpr char const* ImageFlipSchema_TTT::signature;
FunctionDef ImageFlipSchema_TTT::function_def = {
/*name*/"image_flip",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"flip_code", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* image_flip(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("image_flip");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ImageFlipSchema_TTT> parser("image_flip");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ImageFlip(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SinSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Sin;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t SinSchema_TT::max_args;
constexpr size_t SinSchema_TT::max_pos_args;
constexpr char const* SinSchema_TT::signature;
FunctionDef SinSchema_TT::function_def = {
/*name*/"sin",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* sin(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("sin");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SinSchema_TT> parser("sin");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Sin(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct Sin_Schema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Sin_;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t Sin_Schema_TT::max_args;
constexpr size_t Sin_Schema_TT::max_pos_args;
constexpr char const* Sin_Schema_TT::signature;
FunctionDef Sin_Schema_TT::function_def = {
/*name*/"sin_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* sin_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("sin_");
  PythonFrameGuard pf;
  static PythonArgParser<functional::Sin_Schema_TT> parser("sin_");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Sin_(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct CosSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Cos;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t CosSchema_TT::max_args;
constexpr size_t CosSchema_TT::max_pos_args;
constexpr char const* CosSchema_TT::signature;
FunctionDef CosSchema_TT::function_def = {
/*name*/"cos",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* cos(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("cos");
  PythonFrameGuard pf;
  static PythonArgParser<functional::CosSchema_TT> parser("cos");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Cos(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct CoshSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Cosh;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t CoshSchema_TT::max_args;
constexpr size_t CoshSchema_TT::max_pos_args;
constexpr char const* CoshSchema_TT::signature;
FunctionDef CoshSchema_TT::function_def = {
/*name*/"cosh",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* cosh(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("cosh");
  PythonFrameGuard pf;
  static PythonArgParser<functional::CoshSchema_TT> parser("cosh");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Cosh(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct CoshGradSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::CoshGrad;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor dy)";
  static FunctionDef function_def;
};

constexpr size_t CoshGradSchema_TTT::max_args;
constexpr size_t CoshGradSchema_TTT::max_pos_args;
constexpr char const* CoshGradSchema_TTT::signature;
FunctionDef CoshGradSchema_TTT::function_def = {
/*name*/"cosh_grad",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dy", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* cosh_grad(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("cosh_grad");
  PythonFrameGuard pf;
  static PythonArgParser<functional::CoshGradSchema_TTT> parser("cosh_grad");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::CoshGrad(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BroadcastFModSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BroadcastFMod;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t BroadcastFModSchema_TTT::max_args;
constexpr size_t BroadcastFModSchema_TTT::max_pos_args;
constexpr char const* BroadcastFModSchema_TTT::signature;
FunctionDef BroadcastFModSchema_TTT::function_def = {
/*name*/"fmod",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ScalarFModSchema_TTScB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& other, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarFMod;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar other, *, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t ScalarFModSchema_TTScB::max_args;
constexpr size_t ScalarFModSchema_TTScB::max_pos_args;
constexpr char const* ScalarFModSchema_TTScB::signature;
FunctionDef ScalarFModSchema_TTScB::function_def = {
/*name*/"fmod",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct ScalarFModSchema_TTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarFMod;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar other)";
  static FunctionDef function_def;
};

constexpr size_t ScalarFModSchema_TTSc::max_args;
constexpr size_t ScalarFModSchema_TTSc::max_pos_args;
constexpr char const* ScalarFModSchema_TTSc::signature;
FunctionDef ScalarFModSchema_TTSc::function_def = {
/*name*/"fmod",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* fmod(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("fmod");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BroadcastFModSchema_TTT, functional::ScalarFModSchema_TTScB, functional::ScalarFModSchema_TTSc> parser("fmod");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BroadcastFMod(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::ScalarFMod(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>(), r[2].As<bool>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::ScalarFMod(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct LogSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Log;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t LogSchema_TT::max_args;
constexpr size_t LogSchema_TT::max_pos_args;
constexpr char const* LogSchema_TT::signature;
FunctionDef LogSchema_TT::function_def = {
/*name*/"log",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* log(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("log");
  PythonFrameGuard pf;
  static PythonArgParser<functional::LogSchema_TT> parser("log");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Log(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct Log2Schema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Log2;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t Log2Schema_TT::max_args;
constexpr size_t Log2Schema_TT::max_pos_args;
constexpr char const* Log2Schema_TT::signature;
FunctionDef Log2Schema_TT::function_def = {
/*name*/"log2",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* log2(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("log2");
  PythonFrameGuard pf;
  static PythonArgParser<functional::Log2Schema_TT> parser("log2");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Log2(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct Log10Schema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Log10;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t Log10Schema_TT::max_args;
constexpr size_t Log10Schema_TT::max_pos_args;
constexpr char const* Log10Schema_TT::signature;
FunctionDef Log10Schema_TT::function_def = {
/*name*/"log10",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* log10(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("log10");
  PythonFrameGuard pf;
  static PythonArgParser<functional::Log10Schema_TT> parser("log10");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Log10(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SqrtSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Sqrt;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t SqrtSchema_TT::max_args;
constexpr size_t SqrtSchema_TT::max_pos_args;
constexpr char const* SqrtSchema_TT::signature;
FunctionDef SqrtSchema_TT::function_def = {
/*name*/"sqrt",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* sqrt(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("sqrt");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SqrtSchema_TT> parser("sqrt");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Sqrt(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RsqrtSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Rsqrt;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t RsqrtSchema_TT::max_args;
constexpr size_t RsqrtSchema_TT::max_pos_args;
constexpr char const* RsqrtSchema_TT::signature;
FunctionDef RsqrtSchema_TT::function_def = {
/*name*/"rsqrt",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* rsqrt(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("rsqrt");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RsqrtSchema_TT> parser("rsqrt");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Rsqrt(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SquareSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Square;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t SquareSchema_TT::max_args;
constexpr size_t SquareSchema_TT::max_pos_args;
constexpr char const* SquareSchema_TT::signature;
FunctionDef SquareSchema_TT::function_def = {
/*name*/"square",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* square(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("square");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SquareSchema_TT> parser("square");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Square(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SqrtSquareSumSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::SqrtSquareSum;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t SqrtSquareSumSchema_TT::max_args;
constexpr size_t SqrtSquareSumSchema_TT::max_pos_args;
constexpr char const* SqrtSquareSumSchema_TT::signature;
FunctionDef SqrtSquareSumSchema_TT::function_def = {
/*name*/"sqrt_square_sum",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* sqrt_square_sum(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("sqrt_square_sum");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SqrtSquareSumSchema_TT> parser("sqrt_square_sum");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::SqrtSquareSum(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct StandardDeviationSchema_TTI32lBB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const Optional<std::vector<int32_t>>& dim, const Optional<bool>& unbiased, const Optional<bool>& keepdim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::StandardDeviation;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor x, Int32List dim=None, Bool unbiased=None, Bool keepdim=None)";
  static FunctionDef function_def;
};

constexpr size_t StandardDeviationSchema_TTI32lBB::max_args;
constexpr size_t StandardDeviationSchema_TTI32lBB::max_pos_args;
constexpr char const* StandardDeviationSchema_TTI32lBB::signature;
FunctionDef StandardDeviationSchema_TTI32lBB::function_def = {
/*name*/"std",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/Optional<std::vector<int32_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"unbiased", /*default_value*/Optional<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"keepdim", /*default_value*/Optional<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* std(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("std");
  PythonFrameGuard pf;
  static PythonArgParser<functional::StandardDeviationSchema_TTI32lBB> parser("std");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::StandardDeviation(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<std::vector<int32_t>>>(), r[2].As<Optional<bool>>(), r[3].As<Optional<bool>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct VarianceSchema_TTI32lBB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const Optional<std::vector<int32_t>>& dim, const Optional<bool>& unbiased, const Optional<bool>& keepdim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Variance;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor x, Int32List dim=None, Bool unbiased=None, Bool keepdim=None)";
  static FunctionDef function_def;
};

constexpr size_t VarianceSchema_TTI32lBB::max_args;
constexpr size_t VarianceSchema_TTI32lBB::max_pos_args;
constexpr char const* VarianceSchema_TTI32lBB::signature;
FunctionDef VarianceSchema_TTI32lBB::function_def = {
/*name*/"var",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/Optional<std::vector<int32_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"unbiased", /*default_value*/Optional<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"keepdim", /*default_value*/Optional<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* var(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("var");
  PythonFrameGuard pf;
  static PythonArgParser<functional::VarianceSchema_TTI32lBB> parser("var");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Variance(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<std::vector<int32_t>>>(), r[2].As<Optional<bool>>(), r[3].As<Optional<bool>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RMSLayerNormalizationSchema_TTTF {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& hidden_states, const std::shared_ptr<one::Tensor>& weight, float variance_epsilon);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::RMSLayerNormalization;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor hidden_states, Tensor weight, Float variance_epsilon)";
  static FunctionDef function_def;
};

constexpr size_t RMSLayerNormalizationSchema_TTTF::max_args;
constexpr size_t RMSLayerNormalizationSchema_TTTF::max_pos_args;
constexpr char const* RMSLayerNormalizationSchema_TTTF::signature;
FunctionDef RMSLayerNormalizationSchema_TTTF::function_def = {
/*name*/"rms_layer_norm",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"hidden_states", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"variance_epsilon", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* rms_layer_norm(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("rms_layer_norm");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RMSLayerNormalizationSchema_TTTF> parser("rms_layer_norm");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::RMSLayerNormalization(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<float>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ReluSchema_TTB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Relu;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t ReluSchema_TTB::max_args;
constexpr size_t ReluSchema_TTB::max_pos_args;
constexpr char const* ReluSchema_TTB::signature;
FunctionDef ReluSchema_TTB::function_def = {
/*name*/"relu",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* relu(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("relu");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ReluSchema_TTB> parser("relu");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Relu(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct HannWindowSchema_TI64BDeDtB {
  using FType = Maybe<one::Tensor> (int64_t window_length, bool periodic, const Optional<Symbol<Device>>& device, const Optional<Symbol<DType>>& dtype, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::HannWindow;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Int64 window_length, Bool periodic=True, *, Device device=None, DataType dtype=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t HannWindowSchema_TI64BDeDtB::max_args;
constexpr size_t HannWindowSchema_TI64BDeDtB::max_pos_args;
constexpr char const* HannWindowSchema_TI64BDeDtB::signature;
FunctionDef HannWindowSchema_TI64BDeDtB::function_def = {
/*name*/"hann_window",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"window_length", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"periodic", /*default_value*/bool(true), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"device", /*default_value*/Optional<Symbol<Device>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct GlobalHannWindowSchema_TI64BPSbplDtB {
  using FType = Maybe<one::Tensor> (int64_t window_length, bool periodic, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GlobalHannWindow;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Int64 window_length, Bool periodic=True, *, Placement placement, SbpList sbp, DataType dtype=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t GlobalHannWindowSchema_TI64BPSbplDtB::max_args;
constexpr size_t GlobalHannWindowSchema_TI64BPSbplDtB::max_pos_args;
constexpr char const* GlobalHannWindowSchema_TI64BPSbplDtB::signature;
FunctionDef GlobalHannWindowSchema_TI64BPSbplDtB::function_def = {
/*name*/"hann_window",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"window_length", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"periodic", /*default_value*/bool(true), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"placement", /*value_type*/ValueTypeOf<Symbol<ParallelDesc>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"sbp", /*value_type*/ValueTypeOf<std::vector<Symbol<SbpParallel>>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* hann_window(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("hann_window");
  PythonFrameGuard pf;
  static PythonArgParser<functional::HannWindowSchema_TI64BDeDtB, functional::GlobalHannWindowSchema_TI64BPSbplDtB> parser("hann_window");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::HannWindow(r[0].As<int64_t>(), r[1].As<bool>(), r[2].As<Optional<Symbol<Device>>>(), r[3].As<Optional<Symbol<DType>>>(), r[4].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::GlobalHannWindow(r[0].As<int64_t>(), r[1].As<bool>(), r[2].As<Symbol<ParallelDesc>>(), r[3].As<std::vector<Symbol<SbpParallel>>>(), r[4].As<Optional<Symbol<DType>>>(), r[5].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct HardTanhSchema_TTDD {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, double min_val, double max_val);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::HardTanh;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Double min_val, Double max_val)";
  static FunctionDef function_def;
};

constexpr size_t HardTanhSchema_TTDD::max_args;
constexpr size_t HardTanhSchema_TTDD::max_pos_args;
constexpr char const* HardTanhSchema_TTDD::signature;
FunctionDef HardTanhSchema_TTDD::function_def = {
/*name*/"hardtanh",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"min_val", /*value_type*/ValueTypeOf<double>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"max_val", /*value_type*/ValueTypeOf<double>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* hardtanh(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("hardtanh");
  PythonFrameGuard pf;
  static PythonArgParser<functional::HardTanhSchema_TTDD> parser("hardtanh");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::HardTanh(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<double>(), r[2].As<double>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TanSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Tan;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t TanSchema_TT::max_args;
constexpr size_t TanSchema_TT::max_pos_args;
constexpr char const* TanSchema_TT::signature;
FunctionDef TanSchema_TT::function_def = {
/*name*/"tan",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* tan(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("tan");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TanSchema_TT> parser("tan");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Tan(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TanGradSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::TanGrad;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor dy)";
  static FunctionDef function_def;
};

constexpr size_t TanGradSchema_TTT::max_args;
constexpr size_t TanGradSchema_TTT::max_pos_args;
constexpr char const* TanGradSchema_TTT::signature;
FunctionDef TanGradSchema_TTT::function_def = {
/*name*/"tan_grad",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dy", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* tan_grad(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("tan_grad");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TanGradSchema_TTT> parser("tan_grad");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::TanGrad(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TanhSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Tanh;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t TanhSchema_TT::max_args;
constexpr size_t TanhSchema_TT::max_pos_args;
constexpr char const* TanhSchema_TT::signature;
FunctionDef TanhSchema_TT::function_def = {
/*name*/"tanh",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* tanh(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("tanh");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TanhSchema_TT> parser("tanh");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Tanh(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TanhGradSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& dy);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::TanhGrad;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor dy)";
  static FunctionDef function_def;
};

constexpr size_t TanhGradSchema_TTT::max_args;
constexpr size_t TanhGradSchema_TTT::max_pos_args;
constexpr char const* TanhGradSchema_TTT::signature;
FunctionDef TanhGradSchema_TTT::function_def = {
/*name*/"tanh_grad",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dy", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* tanh_grad(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("tanh_grad");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TanhGradSchema_TTT> parser("tanh_grad");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::TanhGrad(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ThresholdSchema_TTDD {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, double threshold, double value);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Threshold;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x, *, Double threshold, Double value)";
  static FunctionDef function_def;
};

constexpr size_t ThresholdSchema_TTDD::max_args;
constexpr size_t ThresholdSchema_TTDD::max_pos_args;
constexpr char const* ThresholdSchema_TTDD::signature;
FunctionDef ThresholdSchema_TTDD::function_def = {
/*name*/"threshold",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"threshold", /*value_type*/ValueTypeOf<double>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"value", /*value_type*/ValueTypeOf<double>(), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* threshold(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("threshold");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ThresholdSchema_TTDD> parser("threshold");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Threshold(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<double>(), r[2].As<double>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct EluSchema_TTD {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, double alpha);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Elu;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Double alpha)";
  static FunctionDef function_def;
};

constexpr size_t EluSchema_TTD::max_args;
constexpr size_t EluSchema_TTD::max_pos_args;
constexpr char const* EluSchema_TTD::signature;
FunctionDef EluSchema_TTD::function_def = {
/*name*/"elu",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"alpha", /*value_type*/ValueTypeOf<double>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* elu(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("elu");
  PythonFrameGuard pf;
  static PythonArgParser<functional::EluSchema_TTD> parser("elu");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Elu(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<double>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct CeluSchema_TTDB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, double alpha, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Celu;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x, *, Double alpha=1.0, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t CeluSchema_TTDB::max_args;
constexpr size_t CeluSchema_TTDB::max_pos_args;
constexpr char const* CeluSchema_TTDB::signature;
FunctionDef CeluSchema_TTDB::function_def = {
/*name*/"celu",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"alpha", /*default_value*/double(1.0), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* celu(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("celu");
  PythonFrameGuard pf;
  static PythonArgParser<functional::CeluSchema_TTDB> parser("celu");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Celu(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<double>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct GeluSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Gelu;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t GeluSchema_TT::max_args;
constexpr size_t GeluSchema_TT::max_pos_args;
constexpr char const* GeluSchema_TT::signature;
FunctionDef GeluSchema_TT::function_def = {
/*name*/"gelu",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* gelu(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("gelu");
  PythonFrameGuard pf;
  static PythonArgParser<functional::GeluSchema_TT> parser("gelu");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Gelu(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct GeluWithApproximateSchema_TTS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::string& approximate);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GeluWithApproximate;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, String approximate=\"none\")";
  static FunctionDef function_def;
};

constexpr size_t GeluWithApproximateSchema_TTS::max_args;
constexpr size_t GeluWithApproximateSchema_TTS::max_pos_args;
constexpr char const* GeluWithApproximateSchema_TTS::signature;
FunctionDef GeluWithApproximateSchema_TTS::function_def = {
/*name*/"gelu_with_approximate",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"approximate", /*default_value*/std::string("none"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* gelu_with_approximate(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("gelu_with_approximate");
  PythonFrameGuard pf;
  static PythonArgParser<functional::GeluWithApproximateSchema_TTS> parser("gelu_with_approximate");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::GeluWithApproximate(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct GluSchema_TTI64 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int64_t dim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Glu;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Int64 dim=-1)";
  static FunctionDef function_def;
};

constexpr size_t GluSchema_TTI64::max_args;
constexpr size_t GluSchema_TTI64::max_pos_args;
constexpr char const* GluSchema_TTI64::signature;
FunctionDef GluSchema_TTI64::function_def = {
/*name*/"glu",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/int64_t(-1), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* glu(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("glu");
  PythonFrameGuard pf;
  static PythonArgParser<functional::GluSchema_TTI64> parser("glu");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Glu(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SigmoidSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Sigmoid;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t SigmoidSchema_TT::max_args;
constexpr size_t SigmoidSchema_TT::max_pos_args;
constexpr char const* SigmoidSchema_TT::signature;
FunctionDef SigmoidSchema_TT::function_def = {
/*name*/"sigmoid",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* sigmoid(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("sigmoid");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SigmoidSchema_TT> parser("sigmoid");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Sigmoid(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SigmoidGradSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& y, const std::shared_ptr<one::Tensor>& dy);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::SigmoidGrad;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor y, Tensor dy)";
  static FunctionDef function_def;
};

constexpr size_t SigmoidGradSchema_TTT::max_args;
constexpr size_t SigmoidGradSchema_TTT::max_pos_args;
constexpr char const* SigmoidGradSchema_TTT::signature;
FunctionDef SigmoidGradSchema_TTT::function_def = {
/*name*/"sigmoid_grad",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"y", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dy", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* sigmoid_grad(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("sigmoid_grad");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SigmoidGradSchema_TTT> parser("sigmoid_grad");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::SigmoidGrad(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct HardSigmoidSchema_TTB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::HardSigmoid;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t HardSigmoidSchema_TTB::max_args;
constexpr size_t HardSigmoidSchema_TTB::max_pos_args;
constexpr char const* HardSigmoidSchema_TTB::signature;
FunctionDef HardSigmoidSchema_TTB::function_def = {
/*name*/"hardsigmoid",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* hardsigmoid(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("hardsigmoid");
  PythonFrameGuard pf;
  static PythonArgParser<functional::HardSigmoidSchema_TTB> parser("hardsigmoid");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::HardSigmoid(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct HardShrinkSchema_TTDB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, double lambd, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::HardShrink;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x, *, Double lambd=0.5, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t HardShrinkSchema_TTDB::max_args;
constexpr size_t HardShrinkSchema_TTDB::max_pos_args;
constexpr char const* HardShrinkSchema_TTDB::signature;
FunctionDef HardShrinkSchema_TTDB::function_def = {
/*name*/"hardshrink",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"lambd", /*default_value*/double(0.5), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* hardshrink(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("hardshrink");
  PythonFrameGuard pf;
  static PythonArgParser<functional::HardShrinkSchema_TTDB> parser("hardshrink");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::HardShrink(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<double>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SoftmaxSchema_TTI64 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const Optional<int64_t>& dim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Softmax;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Int64 dim=None)";
  static FunctionDef function_def;
};

constexpr size_t SoftmaxSchema_TTI64::max_args;
constexpr size_t SoftmaxSchema_TTI64::max_pos_args;
constexpr char const* SoftmaxSchema_TTI64::signature;
FunctionDef SoftmaxSchema_TTI64::function_def = {
/*name*/"softmax",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/Optional<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* softmax(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("softmax");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SoftmaxSchema_TTI64> parser("softmax");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Softmax(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<int64_t>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct LogSoftmaxSchema_TTI64 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const Optional<int64_t>& dim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::LogSoftmax;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Int64 dim=None)";
  static FunctionDef function_def;
};

constexpr size_t LogSoftmaxSchema_TTI64::max_args;
constexpr size_t LogSoftmaxSchema_TTI64::max_pos_args;
constexpr char const* LogSoftmaxSchema_TTI64::signature;
FunctionDef LogSoftmaxSchema_TTI64::function_def = {
/*name*/"log_softmax",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/Optional<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* log_softmax(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("log_softmax");
  PythonFrameGuard pf;
  static PythonArgParser<functional::LogSoftmaxSchema_TTI64> parser("log_softmax");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::LogSoftmax(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<int64_t>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct HardSwishSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::HardSwish;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t HardSwishSchema_TT::max_args;
constexpr size_t HardSwishSchema_TT::max_pos_args;
constexpr char const* HardSwishSchema_TT::signature;
FunctionDef HardSwishSchema_TT::function_def = {
/*name*/"hardswish",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* hardswish(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("hardswish");
  PythonFrameGuard pf;
  static PythonArgParser<functional::HardSwishSchema_TT> parser("hardswish");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::HardSwish(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct LeakyReluSchema_TTFB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, float alpha, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::LeakyRelu;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Float alpha, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t LeakyReluSchema_TTFB::max_args;
constexpr size_t LeakyReluSchema_TTFB::max_pos_args;
constexpr char const* LeakyReluSchema_TTFB::signature;
FunctionDef LeakyReluSchema_TTFB::function_def = {
/*name*/"leaky_relu",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"alpha", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* leaky_relu(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("leaky_relu");
  PythonFrameGuard pf;
  static PythonArgParser<functional::LeakyReluSchema_TTFB> parser("leaky_relu");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::LeakyRelu(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<float>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct NormalSchema_TFFShTDtDeGB {
  using FType = Maybe<one::Tensor> (float mean, float std, const Shape& size, const Optional<one::Tensor>& out, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Normal;
  static constexpr size_t max_args = 8;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Float mean, Float std, Shape size, *, Tensor out=None, DataType dtype=None, Device device=None, Generator generator=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t NormalSchema_TFFShTDtDeGB::max_args;
constexpr size_t NormalSchema_TFFShTDtDeGB::max_pos_args;
constexpr char const* NormalSchema_TFFShTDtDeGB::signature;
FunctionDef NormalSchema_TFFShTDtDeGB::function_def = {
/*name*/"normal",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"mean", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"std", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"size", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"out", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"device", /*default_value*/Optional<Symbol<Device>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct Normal2Schema_TFFI32TDtDeGB {
  using FType = Maybe<one::Tensor> (float mean, float std, int32_t size, const Optional<one::Tensor>& out, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Normal2;
  static constexpr size_t max_args = 8;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Float mean, Float std, Int32 size, *, Tensor out=None, DataType dtype=None, Device device=None, Generator generator=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t Normal2Schema_TFFI32TDtDeGB::max_args;
constexpr size_t Normal2Schema_TFFI32TDtDeGB::max_pos_args;
constexpr char const* Normal2Schema_TFFI32TDtDeGB::signature;
FunctionDef Normal2Schema_TFFI32TDtDeGB::function_def = {
/*name*/"normal",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"mean", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"std", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"size", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"out", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"device", /*default_value*/Optional<Symbol<Device>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct GlobalNormalSchema_TFFShTPSbplDtGB {
  using FType = Maybe<one::Tensor> (float mean, float std, const Shape& size, const Optional<one::Tensor>& out, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GlobalNormal;
  static constexpr size_t max_args = 9;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Float mean, Float std, Shape size, *, Tensor out=None, Placement placement, SbpList sbp, DataType dtype=None, Generator generator=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t GlobalNormalSchema_TFFShTPSbplDtGB::max_args;
constexpr size_t GlobalNormalSchema_TFFShTPSbplDtGB::max_pos_args;
constexpr char const* GlobalNormalSchema_TFFShTPSbplDtGB::signature;
FunctionDef GlobalNormalSchema_TFFShTPSbplDtGB::function_def = {
/*name*/"normal",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"mean", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"std", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"size", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"out", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"placement", /*value_type*/ValueTypeOf<Symbol<ParallelDesc>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"sbp", /*value_type*/ValueTypeOf<std::vector<Symbol<SbpParallel>>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct GlobalNormal2Schema_TFFI32TPSbplDtGB {
  using FType = Maybe<one::Tensor> (float mean, float std, int32_t size, const Optional<one::Tensor>& out, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GlobalNormal2;
  static constexpr size_t max_args = 9;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Float mean, Float std, Int32 size, *, Tensor out=None, Placement placement, SbpList sbp, DataType dtype=None, Generator generator=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t GlobalNormal2Schema_TFFI32TPSbplDtGB::max_args;
constexpr size_t GlobalNormal2Schema_TFFI32TPSbplDtGB::max_pos_args;
constexpr char const* GlobalNormal2Schema_TFFI32TPSbplDtGB::signature;
FunctionDef GlobalNormal2Schema_TFFI32TPSbplDtGB::function_def = {
/*name*/"normal",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"mean", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"std", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"size", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"out", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"placement", /*value_type*/ValueTypeOf<Symbol<ParallelDesc>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"sbp", /*value_type*/ValueTypeOf<std::vector<Symbol<SbpParallel>>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* normal(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("normal");
  PythonFrameGuard pf;
  static PythonArgParser<functional::NormalSchema_TFFShTDtDeGB, functional::Normal2Schema_TFFI32TDtDeGB, functional::GlobalNormalSchema_TFFShTPSbplDtGB, functional::GlobalNormal2Schema_TFFI32TPSbplDtGB> parser("normal");
  ParsedArgs<9> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Normal(r[0].As<float>(), r[1].As<float>(), r[2].As<Shape>(), r[3].As<Optional<one::Tensor>>(), r[4].As<Optional<Symbol<DType>>>(), r[5].As<Optional<Symbol<Device>>>(), r[6].As<Optional<one::Generator>>(), r[7].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::Normal2(r[0].As<float>(), r[1].As<float>(), r[2].As<int32_t>(), r[3].As<Optional<one::Tensor>>(), r[4].As<Optional<Symbol<DType>>>(), r[5].As<Optional<Symbol<Device>>>(), r[6].As<Optional<one::Generator>>(), r[7].As<bool>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::GlobalNormal(r[0].As<float>(), r[1].As<float>(), r[2].As<Shape>(), r[3].As<Optional<one::Tensor>>(), r[4].As<Symbol<ParallelDesc>>(), r[5].As<std::vector<Symbol<SbpParallel>>>(), r[6].As<Optional<Symbol<DType>>>(), r[7].As<Optional<one::Generator>>(), r[8].As<bool>()));
  }
  if (idx == 3) {
    return CastToPyObject(functional::GlobalNormal2(r[0].As<float>(), r[1].As<float>(), r[2].As<int32_t>(), r[3].As<Optional<one::Tensor>>(), r[4].As<Symbol<ParallelDesc>>(), r[5].As<std::vector<Symbol<SbpParallel>>>(), r[6].As<Optional<Symbol<DType>>>(), r[7].As<Optional<one::Generator>>(), r[8].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct NormalizationSchema_TTTTTTI32FFB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const Optional<one::Tensor>& moving_mean, const Optional<one::Tensor>& moving_variance, const Optional<one::Tensor>& gamma, const Optional<one::Tensor>& beta, int32_t axis, float epsilon, float momentum, bool is_training);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Normalization;
  static constexpr size_t max_args = 9;
  static constexpr size_t max_pos_args = 9;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor moving_mean=None, Tensor moving_variance=None, Tensor gamma=None, Tensor beta=None, Int32 axis=1, Float epsilon=1e-5, Float momentum=0.9, Bool is_training=False)";
  static FunctionDef function_def;
};

constexpr size_t NormalizationSchema_TTTTTTI32FFB::max_args;
constexpr size_t NormalizationSchema_TTTTTTI32FFB::max_pos_args;
constexpr char const* NormalizationSchema_TTTTTTI32FFB::signature;
FunctionDef NormalizationSchema_TTTTTTI32FFB::function_def = {
/*name*/"normalization",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"moving_mean", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"moving_variance", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"gamma", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"beta", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"axis", /*default_value*/int32_t(1), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"epsilon", /*default_value*/float(1e-5), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"momentum", /*default_value*/float(0.9), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"is_training", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* normalization(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("normalization");
  PythonFrameGuard pf;
  static PythonArgParser<functional::NormalizationSchema_TTTTTTI32FFB> parser("normalization");
  ParsedArgs<9> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Normalization(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<one::Tensor>>(), r[2].As<Optional<one::Tensor>>(), r[3].As<Optional<one::Tensor>>(), r[4].As<Optional<one::Tensor>>(), r[5].As<int32_t>(), r[6].As<float>(), r[7].As<float>(), r[8].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct NormalizationAddReluSchema_TTTTTTTI32FFB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const Optional<one::Tensor>& addend, const Optional<one::Tensor>& moving_mean, const Optional<one::Tensor>& moving_variance, const std::shared_ptr<one::Tensor>& gamma, const std::shared_ptr<one::Tensor>& beta, int32_t axis, float epsilon, float momentum, bool is_training);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::NormalizationAddRelu;
  static constexpr size_t max_args = 10;
  static constexpr size_t max_pos_args = 10;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor addend=None, Tensor moving_mean=None, Tensor moving_variance=None, Tensor gamma, Tensor beta, Int32 axis=1, Float epsilon=1e-5, Float momentum=0.9, Bool is_training=False)";
  static FunctionDef function_def;
};

constexpr size_t NormalizationAddReluSchema_TTTTTTTI32FFB::max_args;
constexpr size_t NormalizationAddReluSchema_TTTTTTTI32FFB::max_pos_args;
constexpr char const* NormalizationAddReluSchema_TTTTTTTI32FFB::signature;
FunctionDef NormalizationAddReluSchema_TTTTTTTI32FFB::function_def = {
/*name*/"normalization_add_relu",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"addend", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"moving_mean", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"moving_variance", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"gamma", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"beta", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"axis", /*default_value*/int32_t(1), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"epsilon", /*default_value*/float(1e-5), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"momentum", /*default_value*/float(0.9), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"is_training", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* normalization_add_relu(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("normalization_add_relu");
  PythonFrameGuard pf;
  static PythonArgParser<functional::NormalizationAddReluSchema_TTTTTTTI32FFB> parser("normalization_add_relu");
  ParsedArgs<10> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::NormalizationAddRelu(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<one::Tensor>>(), r[2].As<Optional<one::Tensor>>(), r[3].As<Optional<one::Tensor>>(), r[4].As<std::shared_ptr<one::Tensor>>(), r[5].As<std::shared_ptr<one::Tensor>>(), r[6].As<int32_t>(), r[7].As<float>(), r[8].As<float>(), r[9].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct EyeSchema_TScScDtDeB {
  using FType = Maybe<one::Tensor> (const Scalar& n, const Optional<Scalar>& m, const Symbol<DType>& dtype, const Optional<Symbol<Device>>& device, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Eye;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Scalar n, Scalar m=None, *, DataType dtype=kFloat, Device device=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t EyeSchema_TScScDtDeB::max_args;
constexpr size_t EyeSchema_TScScDtDeB::max_pos_args;
constexpr char const* EyeSchema_TScScDtDeB::signature;
FunctionDef EyeSchema_TScScDtDeB::function_def = {
/*name*/"eye",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"n", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"m", /*default_value*/Optional<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"dtype", /*default_value*/Symbol<DType>(DType::Float()), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"device", /*default_value*/Optional<Symbol<Device>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct EyeSchema_TScScDtSB {
  using FType = Maybe<one::Tensor> (const Scalar& n, const Optional<Scalar>& m, const Symbol<DType>& dtype, const std::string& device, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Eye;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Scalar n, Scalar m=None, *, DataType dtype=kFloat, String device, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t EyeSchema_TScScDtSB::max_args;
constexpr size_t EyeSchema_TScScDtSB::max_pos_args;
constexpr char const* EyeSchema_TScScDtSB::signature;
FunctionDef EyeSchema_TScScDtSB::function_def = {
/*name*/"eye",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"n", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"m", /*default_value*/Optional<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"dtype", /*default_value*/Symbol<DType>(DType::Float()), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"device", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct EyeSchema_TScScDtBPSbpl {
  using FType = Maybe<one::Tensor> (const Scalar& n, const Optional<Scalar>& m, const Symbol<DType>& dtype, bool requires_grad, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Eye;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Scalar n, Scalar m=None, *, DataType dtype=kFloat, Bool requires_grad=False, Placement placement, SbpList sbp)";
  static FunctionDef function_def;
};

constexpr size_t EyeSchema_TScScDtBPSbpl::max_args;
constexpr size_t EyeSchema_TScScDtBPSbpl::max_pos_args;
constexpr char const* EyeSchema_TScScDtBPSbpl::signature;
FunctionDef EyeSchema_TScScDtBPSbpl::function_def = {
/*name*/"eye",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"n", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"m", /*default_value*/Optional<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"dtype", /*default_value*/Symbol<DType>(DType::Float()), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"placement", /*value_type*/ValueTypeOf<Symbol<ParallelDesc>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"sbp", /*value_type*/ValueTypeOf<std::vector<Symbol<SbpParallel>>>(), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct EyeSchema_TScScDtBPSbp {
  using FType = Maybe<one::Tensor> (const Scalar& n, const Optional<Scalar>& m, const Symbol<DType>& dtype, bool requires_grad, const Symbol<ParallelDesc>& placement, const Symbol<SbpParallel>& sbp);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Eye;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Scalar n, Scalar m=None, *, DataType dtype=kFloat, Bool requires_grad=False, Placement placement, Sbp sbp)";
  static FunctionDef function_def;
};

constexpr size_t EyeSchema_TScScDtBPSbp::max_args;
constexpr size_t EyeSchema_TScScDtBPSbp::max_pos_args;
constexpr char const* EyeSchema_TScScDtBPSbp::signature;
FunctionDef EyeSchema_TScScDtBPSbp::function_def = {
/*name*/"eye",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"n", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"m", /*default_value*/Optional<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"dtype", /*default_value*/Symbol<DType>(DType::Float()), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"placement", /*value_type*/ValueTypeOf<Symbol<ParallelDesc>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"sbp", /*value_type*/ValueTypeOf<Symbol<SbpParallel>>(), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* eye(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("eye");
  PythonFrameGuard pf;
  static PythonArgParser<functional::EyeSchema_TScScDtDeB, functional::EyeSchema_TScScDtSB, functional::EyeSchema_TScScDtBPSbpl, functional::EyeSchema_TScScDtBPSbp> parser("eye");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Eye(r[0].As<Scalar>(), r[1].As<Optional<Scalar>>(), r[2].As<Symbol<DType>>(), r[3].As<Optional<Symbol<Device>>>(), r[4].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::Eye(r[0].As<Scalar>(), r[1].As<Optional<Scalar>>(), r[2].As<Symbol<DType>>(), r[3].As<std::string>(), r[4].As<bool>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::Eye(r[0].As<Scalar>(), r[1].As<Optional<Scalar>>(), r[2].As<Symbol<DType>>(), r[3].As<bool>(), r[4].As<Symbol<ParallelDesc>>(), r[5].As<std::vector<Symbol<SbpParallel>>>()));
  }
  if (idx == 3) {
    return CastToPyObject(functional::Eye(r[0].As<Scalar>(), r[1].As<Optional<Scalar>>(), r[2].As<Symbol<DType>>(), r[3].As<bool>(), r[4].As<Symbol<ParallelDesc>>(), r[5].As<Symbol<SbpParallel>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct EyeInplaceSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::EyeInplace;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t EyeInplaceSchema_TT::max_args;
constexpr size_t EyeInplaceSchema_TT::max_pos_args;
constexpr char const* EyeInplaceSchema_TT::signature;
FunctionDef EyeInplaceSchema_TT::function_def = {
/*name*/"eye_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* eye_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("eye_");
  PythonFrameGuard pf;
  static PythonArgParser<functional::EyeInplaceSchema_TT> parser("eye_");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::EyeInplace(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ErfinvSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Erfinv;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t ErfinvSchema_TT::max_args;
constexpr size_t ErfinvSchema_TT::max_pos_args;
constexpr char const* ErfinvSchema_TT::signature;
FunctionDef ErfinvSchema_TT::function_def = {
/*name*/"erfinv",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* erfinv(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("erfinv");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ErfinvSchema_TT> parser("erfinv");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Erfinv(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ErfinvInplaceSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ErfinvInplace;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t ErfinvInplaceSchema_TT::max_args;
constexpr size_t ErfinvInplaceSchema_TT::max_pos_args;
constexpr char const* ErfinvInplaceSchema_TT::signature;
FunctionDef ErfinvInplaceSchema_TT::function_def = {
/*name*/"erfinv_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* erfinv_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("erfinv_");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ErfinvInplaceSchema_TT> parser("erfinv_");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ErfinvInplace(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ArangeSchema_TScScScDtDe {
  using FType = Maybe<one::Tensor> (const Scalar& start, const Scalar& end, const Scalar& step, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Arange;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Scalar start, Scalar end, Scalar step=1, *, DataType dtype=None, Device device=None)";
  static FunctionDef function_def;
};

constexpr size_t ArangeSchema_TScScScDtDe::max_args;
constexpr size_t ArangeSchema_TScScScDtDe::max_pos_args;
constexpr char const* ArangeSchema_TScScScDtDe::signature;
FunctionDef ArangeSchema_TScScScDtDe::function_def = {
/*name*/"arange",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"start", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"end", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"step", /*default_value*/Scalar(1), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"device", /*default_value*/Optional<Symbol<Device>>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

struct ArangeSchema_TScDtDe {
  using FType = Maybe<one::Tensor> (const Scalar& end, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Arange;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Scalar end, *, DataType dtype=None, Device device=None)";
  static FunctionDef function_def;
};

constexpr size_t ArangeSchema_TScDtDe::max_args;
constexpr size_t ArangeSchema_TScDtDe::max_pos_args;
constexpr char const* ArangeSchema_TScDtDe::signature;
FunctionDef ArangeSchema_TScDtDe::function_def = {
/*name*/"arange",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"end", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"device", /*default_value*/Optional<Symbol<Device>>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

PyObject* arange(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("arange");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ArangeSchema_TScScScDtDe, functional::ArangeSchema_TScDtDe> parser("arange");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Arange(r[0].As<Scalar>(), r[1].As<Scalar>(), r[2].As<Scalar>(), r[3].As<Optional<Symbol<DType>>>(), r[4].As<Optional<Symbol<Device>>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::Arange(r[0].As<Scalar>(), r[1].As<Optional<Symbol<DType>>>(), r[2].As<Optional<Symbol<Device>>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct GlobalArangeSchema_TScScScDtPSbpl {
  using FType = Maybe<one::Tensor> (const Scalar& start, const Scalar& end, const Scalar& step, const Optional<Symbol<DType>>& dtype, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GlobalArange;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Scalar start, Scalar end, Scalar step=1, *, DataType dtype=None, Placement placement, SbpList sbp)";
  static FunctionDef function_def;
};

constexpr size_t GlobalArangeSchema_TScScScDtPSbpl::max_args;
constexpr size_t GlobalArangeSchema_TScScScDtPSbpl::max_pos_args;
constexpr char const* GlobalArangeSchema_TScScScDtPSbpl::signature;
FunctionDef GlobalArangeSchema_TScScScDtPSbpl::function_def = {
/*name*/"global_arange",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"start", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"end", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"step", /*default_value*/Scalar(1), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"placement", /*value_type*/ValueTypeOf<Symbol<ParallelDesc>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"sbp", /*value_type*/ValueTypeOf<std::vector<Symbol<SbpParallel>>>(), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct GlobalArangeSchema_TScDtPSbpl {
  using FType = Maybe<one::Tensor> (const Scalar& end, const Optional<Symbol<DType>>& dtype, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GlobalArange;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Scalar end, *, DataType dtype=None, Placement placement, SbpList sbp)";
  static FunctionDef function_def;
};

constexpr size_t GlobalArangeSchema_TScDtPSbpl::max_args;
constexpr size_t GlobalArangeSchema_TScDtPSbpl::max_pos_args;
constexpr char const* GlobalArangeSchema_TScDtPSbpl::signature;
FunctionDef GlobalArangeSchema_TScDtPSbpl::function_def = {
/*name*/"global_arange",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"end", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"placement", /*value_type*/ValueTypeOf<Symbol<ParallelDesc>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"sbp", /*value_type*/ValueTypeOf<std::vector<Symbol<SbpParallel>>>(), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* global_arange(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("global_arange");
  PythonFrameGuard pf;
  static PythonArgParser<functional::GlobalArangeSchema_TScScScDtPSbpl, functional::GlobalArangeSchema_TScDtPSbpl> parser("global_arange");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::GlobalArange(r[0].As<Scalar>(), r[1].As<Scalar>(), r[2].As<Scalar>(), r[3].As<Optional<Symbol<DType>>>(), r[4].As<Symbol<ParallelDesc>>(), r[5].As<std::vector<Symbol<SbpParallel>>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::GlobalArange(r[0].As<Scalar>(), r[1].As<Optional<Symbol<DType>>>(), r[2].As<Symbol<ParallelDesc>>(), r[3].As<std::vector<Symbol<SbpParallel>>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct FlattenSchema_TTI32I32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, int32_t start_dim, int32_t end_dim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Flatten;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Int32 start_dim=0, Int32 end_dim=-1)";
  static FunctionDef function_def;
};

constexpr size_t FlattenSchema_TTI32I32::max_args;
constexpr size_t FlattenSchema_TTI32I32::max_pos_args;
constexpr char const* FlattenSchema_TTI32I32::signature;
FunctionDef FlattenSchema_TTI32I32::function_def = {
/*name*/"flatten",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"start_dim", /*default_value*/int32_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"end_dim", /*default_value*/int32_t(-1), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* flatten(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("flatten");
  PythonFrameGuard pf;
  static PythonArgParser<functional::FlattenSchema_TTI32I32> parser("flatten");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Flatten(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>(), r[2].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ArgMaxSchema_TTI32BDt {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const Optional<int32_t>& dim, const Optional<bool>& keepdim, const Optional<Symbol<DType>>& dtype);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ArgMax;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor x, Int32 dim=None, Bool keepdim=None, DataType dtype=None)";
  static FunctionDef function_def;
};

constexpr size_t ArgMaxSchema_TTI32BDt::max_args;
constexpr size_t ArgMaxSchema_TTI32BDt::max_pos_args;
constexpr char const* ArgMaxSchema_TTI32BDt::signature;
FunctionDef ArgMaxSchema_TTI32BDt::function_def = {
/*name*/"argmax",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/Optional<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"keepdim", /*default_value*/Optional<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* argmax(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("argmax");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ArgMaxSchema_TTI32BDt> parser("argmax");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ArgMax(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<int32_t>>(), r[2].As<Optional<bool>>(), r[3].As<Optional<Symbol<DType>>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ArgMinSchema_TTI32BDt {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const Optional<int32_t>& dim, const Optional<bool>& keepdim, const Optional<Symbol<DType>>& dtype);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ArgMin;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor x, Int32 dim=None, Bool keepdim=None, DataType dtype=None)";
  static FunctionDef function_def;
};

constexpr size_t ArgMinSchema_TTI32BDt::max_args;
constexpr size_t ArgMinSchema_TTI32BDt::max_pos_args;
constexpr char const* ArgMinSchema_TTI32BDt::signature;
FunctionDef ArgMinSchema_TTI32BDt::function_def = {
/*name*/"argmin",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/Optional<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"keepdim", /*default_value*/Optional<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* argmin(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("argmin");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ArgMinSchema_TTI32BDt> parser("argmin");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ArgMin(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<int32_t>>(), r[2].As<Optional<bool>>(), r[3].As<Optional<Symbol<DType>>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ArgWhereSchema_TtTDt {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& x, const Symbol<DType>& dtype);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::ArgWhere;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "TensorTuple (Tensor x, DataType dtype=kInt32)";
  static FunctionDef function_def;
};

constexpr size_t ArgWhereSchema_TtTDt::max_args;
constexpr size_t ArgWhereSchema_TtTDt::max_pos_args;
constexpr char const* ArgWhereSchema_TtTDt::signature;
FunctionDef ArgWhereSchema_TtTDt::function_def = {
/*name*/"argwhere",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Symbol<DType>(DType::Int32()), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* argwhere(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("argwhere");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ArgWhereSchema_TtTDt> parser("argwhere");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ArgWhere(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Symbol<DType>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct NonZeroSchema_TtTB {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& x, bool as_tuple);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::NonZero;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "TensorTuple (Tensor x, Bool as_tuple=False)";
  static FunctionDef function_def;
};

constexpr size_t NonZeroSchema_TtTB::max_args;
constexpr size_t NonZeroSchema_TtTB::max_pos_args;
constexpr char const* NonZeroSchema_TtTB::signature;
FunctionDef NonZeroSchema_TtTB::function_def = {
/*name*/"nonzero",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"as_tuple", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* nonzero(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("nonzero");
  PythonFrameGuard pf;
  static PythonArgParser<functional::NonZeroSchema_TtTB> parser("nonzero");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::NonZero(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BroadcastLikeSchema_TTTI32l {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& like, const std::vector<int32_t>& broadcast_axes);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BroadcastLike;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor like, Int32List broadcast_axes=[])";
  static FunctionDef function_def;
};

constexpr size_t BroadcastLikeSchema_TTTI32l::max_args;
constexpr size_t BroadcastLikeSchema_TTTI32l::max_pos_args;
constexpr char const* BroadcastLikeSchema_TTTI32l::signature;
FunctionDef BroadcastLikeSchema_TTTI32l::function_def = {
/*name*/"broadcast_like",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"like", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"broadcast_axes", /*default_value*/std::vector<int32_t>({}), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* broadcast_like(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("broadcast_like");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BroadcastLikeSchema_TTTI32l> parser("broadcast_like");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BroadcastLike(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::vector<int32_t>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct CastSchema_TTDtB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const Symbol<DType>& dtype, bool pin_memory);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Cast;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, DataType dtype, Bool pin_memory=False)";
  static FunctionDef function_def;
};

constexpr size_t CastSchema_TTDtB::max_args;
constexpr size_t CastSchema_TTDtB::max_pos_args;
constexpr char const* CastSchema_TTDtB::signature;
FunctionDef CastSchema_TTDtB::function_def = {
/*name*/"cast",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*value_type*/ValueTypeOf<Symbol<DType>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"pin_memory", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* cast(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("cast");
  PythonFrameGuard pf;
  static PythonArgParser<functional::CastSchema_TTDtB> parser("cast");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Cast(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Symbol<DType>>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ConstantSchema_TShScDtDe {
  using FType = Maybe<one::Tensor> (const Shape& shape, const Scalar& value, const Symbol<DType>& dtype, const Optional<Symbol<Device>>& device);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Constant;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Shape shape, Scalar value, *, DataType dtype, Device device=None)";
  static FunctionDef function_def;
};

constexpr size_t ConstantSchema_TShScDtDe::max_args;
constexpr size_t ConstantSchema_TShScDtDe::max_pos_args;
constexpr char const* ConstantSchema_TShScDtDe::signature;
FunctionDef ConstantSchema_TShScDtDe::function_def = {
/*name*/"constant",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"shape", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"value", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*value_type*/ValueTypeOf<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"device", /*default_value*/Optional<Symbol<Device>>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

PyObject* constant(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("constant");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ConstantSchema_TShScDtDe> parser("constant");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Constant(r[0].As<Shape>(), r[1].As<Scalar>(), r[2].As<Symbol<DType>>(), r[3].As<Optional<Symbol<Device>>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct GlobalConstantSchema_TShScDtPSbpl {
  using FType = Maybe<one::Tensor> (const Shape& shape, const Scalar& value, const Symbol<DType>& dtype, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GlobalConstant;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Shape shape, Scalar value, *, DataType dtype, Placement placement, SbpList sbp)";
  static FunctionDef function_def;
};

constexpr size_t GlobalConstantSchema_TShScDtPSbpl::max_args;
constexpr size_t GlobalConstantSchema_TShScDtPSbpl::max_pos_args;
constexpr char const* GlobalConstantSchema_TShScDtPSbpl::signature;
FunctionDef GlobalConstantSchema_TShScDtPSbpl::function_def = {
/*name*/"global_constant",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"shape", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"value", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*value_type*/ValueTypeOf<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"placement", /*value_type*/ValueTypeOf<Symbol<ParallelDesc>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"sbp", /*value_type*/ValueTypeOf<std::vector<Symbol<SbpParallel>>>(), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* global_constant(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("global_constant");
  PythonFrameGuard pf;
  static PythonArgParser<functional::GlobalConstantSchema_TShScDtPSbpl> parser("global_constant");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::GlobalConstant(r[0].As<Shape>(), r[1].As<Scalar>(), r[2].As<Symbol<DType>>(), r[3].As<Symbol<ParallelDesc>>(), r[4].As<std::vector<Symbol<SbpParallel>>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct EmptySchema_TShDtDeB {
  using FType = Maybe<one::Tensor> (const Shape& shape, const Symbol<DType>& dtype, const Optional<Symbol<Device>>& device, bool pin_memory);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Empty;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Shape shape, *, DataType dtype, Device device=None, Bool pin_memory=False)";
  static FunctionDef function_def;
};

constexpr size_t EmptySchema_TShDtDeB::max_args;
constexpr size_t EmptySchema_TShDtDeB::max_pos_args;
constexpr char const* EmptySchema_TShDtDeB::signature;
FunctionDef EmptySchema_TShDtDeB::function_def = {
/*name*/"empty",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"shape", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*value_type*/ValueTypeOf<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"device", /*default_value*/Optional<Symbol<Device>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"pin_memory", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* empty(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("empty");
  PythonFrameGuard pf;
  static PythonArgParser<functional::EmptySchema_TShDtDeB> parser("empty");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Empty(r[0].As<Shape>(), r[1].As<Symbol<DType>>(), r[2].As<Optional<Symbol<Device>>>(), r[3].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct GlobalEmptySchema_TShDtPSbpl {
  using FType = Maybe<one::Tensor> (const Shape& shape, const Symbol<DType>& dtype, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GlobalEmpty;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Shape shape, *, DataType dtype, Placement placement, SbpList sbp)";
  static FunctionDef function_def;
};

constexpr size_t GlobalEmptySchema_TShDtPSbpl::max_args;
constexpr size_t GlobalEmptySchema_TShDtPSbpl::max_pos_args;
constexpr char const* GlobalEmptySchema_TShDtPSbpl::signature;
FunctionDef GlobalEmptySchema_TShDtPSbpl::function_def = {
/*name*/"global_empty",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"shape", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*value_type*/ValueTypeOf<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"placement", /*value_type*/ValueTypeOf<Symbol<ParallelDesc>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"sbp", /*value_type*/ValueTypeOf<std::vector<Symbol<SbpParallel>>>(), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* global_empty(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("global_empty");
  PythonFrameGuard pf;
  static PythonArgParser<functional::GlobalEmptySchema_TShDtPSbpl> parser("global_empty");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::GlobalEmpty(r[0].As<Shape>(), r[1].As<Symbol<DType>>(), r[2].As<Symbol<ParallelDesc>>(), r[3].As<std::vector<Symbol<SbpParallel>>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BernoulliSchema_TTDtGB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Symbol<DType>& dtype, const Optional<one::Generator>& generator, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Bernoulli;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input, *, DataType dtype=kFloat, Generator generator=None, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t BernoulliSchema_TTDtGB::max_args;
constexpr size_t BernoulliSchema_TTDtGB::max_pos_args;
constexpr char const* BernoulliSchema_TTDtGB::signature;
FunctionDef BernoulliSchema_TTDtGB::function_def = {
/*name*/"bernoulli",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Symbol<DType>(DType::Float()), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct BernoulliProbSchema_TTDDtGB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, double p, const Symbol<DType>& dtype, const Optional<one::Generator>& generator, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BernoulliProb;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Double p, *, DataType dtype=kFloat, Generator generator=None, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t BernoulliProbSchema_TTDDtGB::max_args;
constexpr size_t BernoulliProbSchema_TTDDtGB::max_pos_args;
constexpr char const* BernoulliProbSchema_TTDDtGB::signature;
FunctionDef BernoulliProbSchema_TTDDtGB::function_def = {
/*name*/"bernoulli",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"p", /*value_type*/ValueTypeOf<double>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Symbol<DType>(DType::Float()), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* bernoulli(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("bernoulli");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BernoulliSchema_TTDtGB, functional::BernoulliProbSchema_TTDDtGB> parser("bernoulli");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Bernoulli(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Symbol<DType>>(), r[2].As<Optional<one::Generator>>(), r[3].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::BernoulliProb(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<double>(), r[2].As<Symbol<DType>>(), r[3].As<Optional<one::Generator>>(), r[4].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BernoulliInplaceSchema_TTDtG {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Symbol<DType>& dtype, const Optional<one::Generator>& generator);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BernoulliInplace;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input, *, DataType dtype=kFloat, Generator generator=None)";
  static FunctionDef function_def;
};

constexpr size_t BernoulliInplaceSchema_TTDtG::max_args;
constexpr size_t BernoulliInplaceSchema_TTDtG::max_pos_args;
constexpr char const* BernoulliInplaceSchema_TTDtG::signature;
FunctionDef BernoulliInplaceSchema_TTDtG::function_def = {
/*name*/"bernoulli_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Symbol<DType>(DType::Float()), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

struct BernoulliProbInplaceSchema_TTDDtG {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, double p, const Symbol<DType>& dtype, const Optional<one::Generator>& generator);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BernoulliProbInplace;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Double p, *, DataType dtype=kFloat, Generator generator=None)";
  static FunctionDef function_def;
};

constexpr size_t BernoulliProbInplaceSchema_TTDDtG::max_args;
constexpr size_t BernoulliProbInplaceSchema_TTDDtG::max_pos_args;
constexpr char const* BernoulliProbInplaceSchema_TTDDtG::signature;
FunctionDef BernoulliProbInplaceSchema_TTDDtG::function_def = {
/*name*/"bernoulli_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"p", /*value_type*/ValueTypeOf<double>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Symbol<DType>(DType::Float()), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

PyObject* bernoulli_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("bernoulli_");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BernoulliInplaceSchema_TTDtG, functional::BernoulliProbInplaceSchema_TTDDtG> parser("bernoulli_");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BernoulliInplace(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Symbol<DType>>(), r[2].As<Optional<one::Generator>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::BernoulliProbInplace(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<double>(), r[2].As<Symbol<DType>>(), r[3].As<Optional<one::Generator>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ConcatSchema_TTtI64 {
  using FType = Maybe<one::Tensor> (const TensorTuple& inputs, int64_t dim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Concat;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (TensorTuple inputs, Int64 dim=0)";
  static FunctionDef function_def;
};

constexpr size_t ConcatSchema_TTtI64::max_args;
constexpr size_t ConcatSchema_TTtI64::max_pos_args;
constexpr char const* ConcatSchema_TTtI64::signature;
FunctionDef ConcatSchema_TTtI64::function_def = {
/*name*/"concat",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"inputs", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/int64_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* concat(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("concat");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ConcatSchema_TTtI64> parser("concat");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Concat(r[0].As<TensorTuple>(), r[1].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BiasAddSchema_TTTI32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& bias, int32_t axis);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BiasAdd;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor bias, Int32 axis=1)";
  static FunctionDef function_def;
};

constexpr size_t BiasAddSchema_TTTI32::max_args;
constexpr size_t BiasAddSchema_TTTI32::max_pos_args;
constexpr char const* BiasAddSchema_TTTI32::signature;
FunctionDef BiasAddSchema_TTTI32::function_def = {
/*name*/"bias_add",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bias", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"axis", /*default_value*/int32_t(1), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* bias_add(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("bias_add");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BiasAddSchema_TTTI32> parser("bias_add");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BiasAdd(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct Conv1dSchema_TTTTI32lI32lI32lI32S {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& dilation, int32_t groups, const std::string& channel_pos);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Conv1d;
  static constexpr size_t max_args = 8;
  static constexpr size_t max_pos_args = 8;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor weight, Tensor bias=None, Int32List stride=1, Int32List padding=0, Int32List dilation=1, Int32 groups=1, String channel_pos=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t Conv1dSchema_TTTTI32lI32lI32lI32S::max_args;
constexpr size_t Conv1dSchema_TTTTI32lI32lI32lI32S::max_pos_args;
constexpr char const* Conv1dSchema_TTTTI32lI32lI32lI32S::signature;
FunctionDef Conv1dSchema_TTTTI32lI32lI32lI32S::function_def = {
/*name*/"conv1d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bias", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"stride", /*default_value*/std::vector<int32_t>({1}), /*size*/1, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"padding", /*default_value*/std::vector<int32_t>({0}), /*size*/1, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dilation", /*default_value*/std::vector<int32_t>({1}), /*size*/1, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"groups", /*default_value*/int32_t(1), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"channel_pos", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* conv1d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("conv1d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::Conv1dSchema_TTTTI32lI32lI32lI32S> parser("conv1d");
  ParsedArgs<8> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Conv1d(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Optional<one::Tensor>>(), r[3].As<std::vector<int32_t>>(), r[4].As<std::vector<int32_t>>(), r[5].As<std::vector<int32_t>>(), r[6].As<int32_t>(), r[7].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct Conv2dSchema_TTTTI32lI32lI32lI32S {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& dilation, int32_t groups, const std::string& channel_pos);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Conv2d;
  static constexpr size_t max_args = 8;
  static constexpr size_t max_pos_args = 8;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor weight, Tensor bias=None, Int32List stride=1, Int32List padding=0, Int32List dilation=1, Int32 groups=1, String channel_pos=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t Conv2dSchema_TTTTI32lI32lI32lI32S::max_args;
constexpr size_t Conv2dSchema_TTTTI32lI32lI32lI32S::max_pos_args;
constexpr char const* Conv2dSchema_TTTTI32lI32lI32lI32S::signature;
FunctionDef Conv2dSchema_TTTTI32lI32lI32lI32S::function_def = {
/*name*/"conv2d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bias", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"stride", /*default_value*/std::vector<int32_t>({1, 1}), /*size*/2, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"padding", /*default_value*/std::vector<int32_t>({0, 0}), /*size*/2, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dilation", /*default_value*/std::vector<int32_t>({1, 1}), /*size*/2, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"groups", /*default_value*/int32_t(1), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"channel_pos", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* conv2d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("conv2d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::Conv2dSchema_TTTTI32lI32lI32lI32S> parser("conv2d");
  ParsedArgs<8> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Conv2d(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Optional<one::Tensor>>(), r[3].As<std::vector<int32_t>>(), r[4].As<std::vector<int32_t>>(), r[5].As<std::vector<int32_t>>(), r[6].As<int32_t>(), r[7].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct Conv3dSchema_TTTTI32lI32lI32lI32S {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& dilation, int32_t groups, const std::string& channel_pos);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Conv3d;
  static constexpr size_t max_args = 8;
  static constexpr size_t max_pos_args = 8;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor weight, Tensor bias=None, Int32List stride=1, Int32List padding=0, Int32List dilation=1, Int32 groups=1, String channel_pos=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t Conv3dSchema_TTTTI32lI32lI32lI32S::max_args;
constexpr size_t Conv3dSchema_TTTTI32lI32lI32lI32S::max_pos_args;
constexpr char const* Conv3dSchema_TTTTI32lI32lI32lI32S::signature;
FunctionDef Conv3dSchema_TTTTI32lI32lI32lI32S::function_def = {
/*name*/"conv3d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bias", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"stride", /*default_value*/std::vector<int32_t>({1, 1, 1}), /*size*/3, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"padding", /*default_value*/std::vector<int32_t>({0, 0, 0}), /*size*/3, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dilation", /*default_value*/std::vector<int32_t>({1, 1, 1}), /*size*/3, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"groups", /*default_value*/int32_t(1), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"channel_pos", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* conv3d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("conv3d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::Conv3dSchema_TTTTI32lI32lI32lI32S> parser("conv3d");
  ParsedArgs<8> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Conv3d(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Optional<one::Tensor>>(), r[3].As<std::vector<int32_t>>(), r[4].As<std::vector<int32_t>>(), r[5].As<std::vector<int32_t>>(), r[6].As<int32_t>(), r[7].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct FakeQuantizationSchema_TTTTSI32S {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& scale, const std::shared_ptr<one::Tensor>& zero_point, const std::string& quantization_formula, int32_t quantization_bit, const std::string& quantization_scheme);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::FakeQuantization;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 6;
  static constexpr char const* signature = "Tensor (Tensor in, Tensor scale, Tensor zero_point, String quantization_formula, Int32 quantization_bit, String quantization_scheme)";
  static FunctionDef function_def;
};

constexpr size_t FakeQuantizationSchema_TTTTSI32S::max_args;
constexpr size_t FakeQuantizationSchema_TTTTSI32S::max_pos_args;
constexpr char const* FakeQuantizationSchema_TTTTSI32S::signature;
FunctionDef FakeQuantizationSchema_TTTTSI32S::function_def = {
/*name*/"fake_quantization",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"in", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"scale", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"zero_point", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"quantization_formula", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"quantization_bit", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"quantization_scheme", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* fake_quantization(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("fake_quantization");
  PythonFrameGuard pf;
  static PythonArgParser<functional::FakeQuantizationSchema_TTTTSI32S> parser("fake_quantization");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::FakeQuantization(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<std::string>(), r[4].As<int32_t>(), r[5].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct QuantizationSchema_TTTTSI32S {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& scale, const std::shared_ptr<one::Tensor>& zero_point, const std::string& quantization_formula, int32_t quantization_bit, const std::string& quantization_scheme);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Quantization;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 6;
  static constexpr char const* signature = "Tensor (Tensor in, Tensor scale, Tensor zero_point, String quantization_formula, Int32 quantization_bit, String quantization_scheme)";
  static FunctionDef function_def;
};

constexpr size_t QuantizationSchema_TTTTSI32S::max_args;
constexpr size_t QuantizationSchema_TTTTSI32S::max_pos_args;
constexpr char const* QuantizationSchema_TTTTSI32S::signature;
FunctionDef QuantizationSchema_TTTTSI32S::function_def = {
/*name*/"quantization",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"in", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"scale", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"zero_point", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"quantization_formula", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"quantization_bit", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"quantization_scheme", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* quantization(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("quantization");
  PythonFrameGuard pf;
  static PythonArgParser<functional::QuantizationSchema_TTTTSI32S> parser("quantization");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Quantization(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<std::string>(), r[4].As<int32_t>(), r[5].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MinMaxObserverSchema_TtTSI32SB {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& in, const std::string& quantization_formula, int32_t quantization_bit, const std::string& quantization_scheme, bool per_layer_quantization);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::MinMaxObserver;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "TensorTuple (Tensor in, String quantization_formula, Int32 quantization_bit, String quantization_scheme, Bool per_layer_quantization)";
  static FunctionDef function_def;
};

constexpr size_t MinMaxObserverSchema_TtTSI32SB::max_args;
constexpr size_t MinMaxObserverSchema_TtTSI32SB::max_pos_args;
constexpr char const* MinMaxObserverSchema_TtTSI32SB::signature;
FunctionDef MinMaxObserverSchema_TtTSI32SB::function_def = {
/*name*/"min_max_observer",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"in", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"quantization_formula", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"quantization_bit", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"quantization_scheme", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"per_layer_quantization", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* min_max_observer(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("min_max_observer");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MinMaxObserverSchema_TtTSI32SB> parser("min_max_observer");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::MinMaxObserver(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::string>(), r[2].As<int32_t>(), r[3].As<std::string>(), r[4].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MovingAverageMinMaxObserverSchema_TtTTTTBI64SI32SF {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& current_train_step, const std::shared_ptr<one::Tensor>& moving_max, const std::shared_ptr<one::Tensor>& moving_min, bool training, int64_t stop_update_after_iters, const std::string& quantization_formula, int32_t quantization_bit, const std::string& quantization_scheme, float momentum);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::MovingAverageMinMaxObserver;
  static constexpr size_t max_args = 10;
  static constexpr size_t max_pos_args = 10;
  static constexpr char const* signature = "TensorTuple (Tensor in, Tensor current_train_step, Tensor moving_max, Tensor moving_min, Bool training, Int64 stop_update_after_iters, String quantization_formula, Int32 quantization_bit, String quantization_scheme, Float momentum)";
  static FunctionDef function_def;
};

constexpr size_t MovingAverageMinMaxObserverSchema_TtTTTTBI64SI32SF::max_args;
constexpr size_t MovingAverageMinMaxObserverSchema_TtTTTTBI64SI32SF::max_pos_args;
constexpr char const* MovingAverageMinMaxObserverSchema_TtTTTTBI64SI32SF::signature;
FunctionDef MovingAverageMinMaxObserverSchema_TtTTTTBI64SI32SF::function_def = {
/*name*/"moving_average_min_max_observer",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"in", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"current_train_step", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"moving_max", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"moving_min", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"training", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"stop_update_after_iters", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"quantization_formula", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"quantization_bit", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"quantization_scheme", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"momentum", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* moving_average_min_max_observer(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("moving_average_min_max_observer");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MovingAverageMinMaxObserverSchema_TtTTTTBI64SI32SF> parser("moving_average_min_max_observer");
  ParsedArgs<10> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::MovingAverageMinMaxObserver(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<std::shared_ptr<one::Tensor>>(), r[4].As<bool>(), r[5].As<int64_t>(), r[6].As<std::string>(), r[7].As<int32_t>(), r[8].As<std::string>(), r[9].As<float>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct Deconv1dSchema_TTTTI32lI32lI32lI32I32lS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& output_padding, int32_t groups, const std::vector<int32_t>& dilation, const std::string& data_format);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Deconv1d;
  static constexpr size_t max_args = 9;
  static constexpr size_t max_pos_args = 9;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor weight, Tensor bias=None, Int32List stride=1, Int32List padding=0, Int32List output_padding=0, Int32 groups=1, Int32List dilation=1, String data_format=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t Deconv1dSchema_TTTTI32lI32lI32lI32I32lS::max_args;
constexpr size_t Deconv1dSchema_TTTTI32lI32lI32lI32I32lS::max_pos_args;
constexpr char const* Deconv1dSchema_TTTTI32lI32lI32lI32I32lS::signature;
FunctionDef Deconv1dSchema_TTTTI32lI32lI32lI32I32lS::function_def = {
/*name*/"deconv1d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bias", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"stride", /*default_value*/std::vector<int32_t>({1}), /*size*/1, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"padding", /*default_value*/std::vector<int32_t>({0}), /*size*/1, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"output_padding", /*default_value*/std::vector<int32_t>({0}), /*size*/1, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"groups", /*default_value*/int32_t(1), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dilation", /*default_value*/std::vector<int32_t>({1}), /*size*/1, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"data_format", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* deconv1d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("deconv1d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::Deconv1dSchema_TTTTI32lI32lI32lI32I32lS> parser("deconv1d");
  ParsedArgs<9> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Deconv1d(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Optional<one::Tensor>>(), r[3].As<std::vector<int32_t>>(), r[4].As<std::vector<int32_t>>(), r[5].As<std::vector<int32_t>>(), r[6].As<int32_t>(), r[7].As<std::vector<int32_t>>(), r[8].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct Deconv2dSchema_TTTTI32lI32lI32lI32I32lS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& output_padding, int32_t groups, const std::vector<int32_t>& dilation, const std::string& data_format);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Deconv2d;
  static constexpr size_t max_args = 9;
  static constexpr size_t max_pos_args = 9;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor weight, Tensor bias=None, Int32List stride=1, Int32List padding=0, Int32List output_padding=0, Int32 groups=1, Int32List dilation=1, String data_format=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t Deconv2dSchema_TTTTI32lI32lI32lI32I32lS::max_args;
constexpr size_t Deconv2dSchema_TTTTI32lI32lI32lI32I32lS::max_pos_args;
constexpr char const* Deconv2dSchema_TTTTI32lI32lI32lI32I32lS::signature;
FunctionDef Deconv2dSchema_TTTTI32lI32lI32lI32I32lS::function_def = {
/*name*/"deconv2d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bias", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"stride", /*default_value*/std::vector<int32_t>({1, 1}), /*size*/2, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"padding", /*default_value*/std::vector<int32_t>({0, 0}), /*size*/2, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"output_padding", /*default_value*/std::vector<int32_t>({0, 0}), /*size*/2, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"groups", /*default_value*/int32_t(1), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dilation", /*default_value*/std::vector<int32_t>({1, 1}), /*size*/2, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"data_format", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* deconv2d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("deconv2d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::Deconv2dSchema_TTTTI32lI32lI32lI32I32lS> parser("deconv2d");
  ParsedArgs<9> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Deconv2d(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Optional<one::Tensor>>(), r[3].As<std::vector<int32_t>>(), r[4].As<std::vector<int32_t>>(), r[5].As<std::vector<int32_t>>(), r[6].As<int32_t>(), r[7].As<std::vector<int32_t>>(), r[8].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct Deconv3dSchema_TTTTI32lI32lI32lI32I32lS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const Optional<one::Tensor>& bias, const std::vector<int32_t>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& output_padding, int32_t groups, const std::vector<int32_t>& dilation, const std::string& data_format);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Deconv3d;
  static constexpr size_t max_args = 9;
  static constexpr size_t max_pos_args = 9;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor weight, Tensor bias=None, Int32List stride=1, Int32List padding=0, Int32List output_padding=0, Int32 groups=1, Int32List dilation=1, String data_format=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t Deconv3dSchema_TTTTI32lI32lI32lI32I32lS::max_args;
constexpr size_t Deconv3dSchema_TTTTI32lI32lI32lI32I32lS::max_pos_args;
constexpr char const* Deconv3dSchema_TTTTI32lI32lI32lI32I32lS::signature;
FunctionDef Deconv3dSchema_TTTTI32lI32lI32lI32I32lS::function_def = {
/*name*/"deconv3d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bias", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"stride", /*default_value*/std::vector<int32_t>({1, 1, 1}), /*size*/3, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"padding", /*default_value*/std::vector<int32_t>({0, 0, 0}), /*size*/3, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"output_padding", /*default_value*/std::vector<int32_t>({0, 0, 0}), /*size*/3, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"groups", /*default_value*/int32_t(1), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dilation", /*default_value*/std::vector<int32_t>({1, 1, 1}), /*size*/3, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"data_format", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* deconv3d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("deconv3d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::Deconv3dSchema_TTTTI32lI32lI32lI32I32lS> parser("deconv3d");
  ParsedArgs<9> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Deconv3d(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Optional<one::Tensor>>(), r[3].As<std::vector<int32_t>>(), r[4].As<std::vector<int32_t>>(), r[5].As<std::vector<int32_t>>(), r[6].As<int32_t>(), r[7].As<std::vector<int32_t>>(), r[8].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ExpandSchema_TTSh {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const Shape& shape);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Expand;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Shape shape)";
  static FunctionDef function_def;
};

constexpr size_t ExpandSchema_TTSh::max_args;
constexpr size_t ExpandSchema_TTSh::max_pos_args;
constexpr char const* ExpandSchema_TTSh::signature;
FunctionDef ExpandSchema_TTSh::function_def = {
/*name*/"expand",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"shape", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* expand(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("expand");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ExpandSchema_TTSh> parser("expand");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Expand(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Shape>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RepeatSchema_TTSh {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Shape& repeat_shape);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Repeat;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Shape repeat_shape)";
  static FunctionDef function_def;
};

constexpr size_t RepeatSchema_TTSh::max_args;
constexpr size_t RepeatSchema_TTSh::max_pos_args;
constexpr char const* RepeatSchema_TTSh::signature;
FunctionDef RepeatSchema_TTSh::function_def = {
/*name*/"repeat",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"repeat_shape", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* repeat(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("repeat");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RepeatSchema_TTSh> parser("repeat");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Repeat(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Shape>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RepeatInterLeaveIntSchema_TTI32I32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int32_t repeats, const Optional<int32_t>& dim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::RepeatInterLeaveInt;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Int32 repeats, Int32 dim=None)";
  static FunctionDef function_def;
};

constexpr size_t RepeatInterLeaveIntSchema_TTI32I32::max_args;
constexpr size_t RepeatInterLeaveIntSchema_TTI32I32::max_pos_args;
constexpr char const* RepeatInterLeaveIntSchema_TTI32I32::signature;
FunctionDef RepeatInterLeaveIntSchema_TTI32I32::function_def = {
/*name*/"repeat_interleave",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"repeats", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/Optional<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

struct RepeatInterLeaveTensorSchema_TTTI32I32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& repeats, int32_t dim, const Optional<int32_t>& output_size);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::RepeatInterLeaveTensor;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor repeats, Int32 dim, Int32 output_size=None)";
  static FunctionDef function_def;
};

constexpr size_t RepeatInterLeaveTensorSchema_TTTI32I32::max_args;
constexpr size_t RepeatInterLeaveTensorSchema_TTTI32I32::max_pos_args;
constexpr char const* RepeatInterLeaveTensorSchema_TTTI32I32::signature;
FunctionDef RepeatInterLeaveTensorSchema_TTTI32I32::function_def = {
/*name*/"repeat_interleave",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"repeats", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"output_size", /*default_value*/Optional<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* repeat_interleave(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("repeat_interleave");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RepeatInterLeaveIntSchema_TTI32I32, functional::RepeatInterLeaveTensorSchema_TTTI32I32> parser("repeat_interleave");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::RepeatInterLeaveInt(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>(), r[2].As<Optional<int32_t>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::RepeatInterLeaveTensor(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<int32_t>(), r[3].As<Optional<int32_t>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TileSchema_TTSh {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Shape& dims);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Tile;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Shape dims)";
  static FunctionDef function_def;
};

constexpr size_t TileSchema_TTSh::max_args;
constexpr size_t TileSchema_TTSh::max_pos_args;
constexpr char const* TileSchema_TTSh::signature;
FunctionDef TileSchema_TTSh::function_def = {
/*name*/"tile",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dims", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* tile(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("tile");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TileSchema_TTSh> parser("tile");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Tile(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Shape>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RollSchema_TTI32lI32l {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& shifts, const Optional<std::vector<int32_t>>& dims);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Roll;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Int32List shifts, Int32List dims=None)";
  static FunctionDef function_def;
};

constexpr size_t RollSchema_TTI32lI32l::max_args;
constexpr size_t RollSchema_TTI32lI32l::max_pos_args;
constexpr char const* RollSchema_TTI32lI32l::signature;
FunctionDef RollSchema_TTI32lI32l::function_def = {
/*name*/"roll",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"shifts", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dims", /*default_value*/Optional<std::vector<int32_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* roll(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("roll");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RollSchema_TTI32lI32l> parser("roll");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Roll(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<Optional<std::vector<int32_t>>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ExpandDimsSchema_TTI32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int32_t dim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ExpandDims;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Int32 dim)";
  static FunctionDef function_def;
};

constexpr size_t ExpandDimsSchema_TTI32::max_args;
constexpr size_t ExpandDimsSchema_TTI32::max_pos_args;
constexpr char const* ExpandDimsSchema_TTI32::signature;
FunctionDef ExpandDimsSchema_TTI32::function_def = {
/*name*/"expand_dims",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* expand_dims(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("expand_dims");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ExpandDimsSchema_TTI32> parser("expand_dims");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ExpandDims(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct UnsqueezeSchema_TTI32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int32_t dim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Unsqueeze;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Int32 dim)";
  static FunctionDef function_def;
};

constexpr size_t UnsqueezeSchema_TTI32::max_args;
constexpr size_t UnsqueezeSchema_TTI32::max_pos_args;
constexpr char const* UnsqueezeSchema_TTI32::signature;
FunctionDef UnsqueezeSchema_TTI32::function_def = {
/*name*/"unsqueeze",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* unsqueeze(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("unsqueeze");
  PythonFrameGuard pf;
  static PythonArgParser<functional::UnsqueezeSchema_TTI32> parser("unsqueeze");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Unsqueeze(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SqueezeSchema_TTI32l {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const Optional<std::vector<int32_t>>& dim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Squeeze;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Int32List dim=None)";
  static FunctionDef function_def;
};

constexpr size_t SqueezeSchema_TTI32l::max_args;
constexpr size_t SqueezeSchema_TTI32l::max_pos_args;
constexpr char const* SqueezeSchema_TTI32l::signature;
FunctionDef SqueezeSchema_TTI32l::function_def = {
/*name*/"squeeze",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/Optional<std::vector<int32_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* squeeze(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("squeeze");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SqueezeSchema_TTI32l> parser("squeeze");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Squeeze(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<std::vector<int32_t>>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ExpSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Exp;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t ExpSchema_TT::max_args;
constexpr size_t ExpSchema_TT::max_pos_args;
constexpr char const* ExpSchema_TT::signature;
FunctionDef ExpSchema_TT::function_def = {
/*name*/"exp",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* exp(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("exp");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ExpSchema_TT> parser("exp");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Exp(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct GatherSchema_TTTI64 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& indices, int64_t axis);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Gather;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor indices, Int64 axis)";
  static FunctionDef function_def;
};

constexpr size_t GatherSchema_TTTI64::max_args;
constexpr size_t GatherSchema_TTTI64::max_pos_args;
constexpr char const* GatherSchema_TTTI64::signature;
FunctionDef GatherSchema_TTTI64::function_def = {
/*name*/"gather",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"indices", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"axis", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* gather(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("gather");
  PythonFrameGuard pf;
  static PythonArgParser<functional::GatherSchema_TTTI64> parser("gather");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Gather(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct DimGatherSchema_TTI64TB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int64_t dim, const std::shared_ptr<one::Tensor>& index, bool sparse_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::DimGather;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor input, Int64 dim, Tensor index, Bool sparse_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t DimGatherSchema_TTI64TB::max_args;
constexpr size_t DimGatherSchema_TTI64TB::max_pos_args;
constexpr char const* DimGatherSchema_TTI64TB::signature;
FunctionDef DimGatherSchema_TTI64TB::function_def = {
/*name*/"dim_gather",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"index", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"sparse_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* dim_gather(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("dim_gather");
  PythonFrameGuard pf;
  static PythonArgParser<functional::DimGatherSchema_TTI64TB> parser("dim_gather");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::DimGather(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct EmbeddingReNormSchema_TTTDD {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& indices, double max_norm, double norm_type);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::EmbeddingReNorm;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor in, Tensor indices, Double max_norm, Double norm_type)";
  static FunctionDef function_def;
};

constexpr size_t EmbeddingReNormSchema_TTTDD::max_args;
constexpr size_t EmbeddingReNormSchema_TTTDD::max_pos_args;
constexpr char const* EmbeddingReNormSchema_TTTDD::signature;
FunctionDef EmbeddingReNormSchema_TTTDD::function_def = {
/*name*/"embedding_renorm_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"in", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"indices", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"max_norm", /*value_type*/ValueTypeOf<double>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"norm_type", /*value_type*/ValueTypeOf<double>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* embedding_renorm_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("embedding_renorm_");
  PythonFrameGuard pf;
  static PythonArgParser<functional::EmbeddingReNormSchema_TTTDD> parser("embedding_renorm_");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::EmbeddingReNorm(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<double>(), r[3].As<double>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct EmbeddingSchema_TTTI64B {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& indices, const Optional<int64_t>& padding_idx, bool scale_grad_by_freq);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Embedding;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor weight, Tensor indices, Int64 padding_idx=None, Bool scale_grad_by_freq=False)";
  static FunctionDef function_def;
};

constexpr size_t EmbeddingSchema_TTTI64B::max_args;
constexpr size_t EmbeddingSchema_TTTI64B::max_pos_args;
constexpr char const* EmbeddingSchema_TTTI64B::signature;
FunctionDef EmbeddingSchema_TTTI64B::function_def = {
/*name*/"embedding",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"weight", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"indices", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"padding_idx", /*default_value*/Optional<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"scale_grad_by_freq", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* embedding(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("embedding");
  PythonFrameGuard pf;
  static PythonArgParser<functional::EmbeddingSchema_TTTI64B> parser("embedding");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Embedding(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Optional<int64_t>>(), r[3].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ArgSortSchema_TTS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& in, const std::string& direction);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ArgSort;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor in, String direction)";
  static FunctionDef function_def;
};

constexpr size_t ArgSortSchema_TTS::max_args;
constexpr size_t ArgSortSchema_TTS::max_pos_args;
constexpr char const* ArgSortSchema_TTS::signature;
FunctionDef ArgSortSchema_TTS::function_def = {
/*name*/"arg_sort",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"in", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"direction", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* arg_sort(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("arg_sort");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ArgSortSchema_TTS> parser("arg_sort");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ArgSort(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct GatherNdSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& params, const std::shared_ptr<one::Tensor>& indices);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GatherNd;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor params, Tensor indices)";
  static FunctionDef function_def;
};

constexpr size_t GatherNdSchema_TTT::max_args;
constexpr size_t GatherNdSchema_TTT::max_pos_args;
constexpr char const* GatherNdSchema_TTT::signature;
FunctionDef GatherNdSchema_TTT::function_def = {
/*name*/"gather_nd",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"params", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"indices", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* gather_nd(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("gather_nd");
  PythonFrameGuard pf;
  static PythonArgParser<functional::GatherNdSchema_TTT> parser("gather_nd");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::GatherNd(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ScatterNdSchema_TTTSh {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& indices, const std::shared_ptr<one::Tensor>& updates, const Shape& shape);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScatterNd;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor indices, Tensor updates, Shape shape)";
  static FunctionDef function_def;
};

constexpr size_t ScatterNdSchema_TTTSh::max_args;
constexpr size_t ScatterNdSchema_TTTSh::max_pos_args;
constexpr char const* ScatterNdSchema_TTTSh::signature;
FunctionDef ScatterNdSchema_TTTSh::function_def = {
/*name*/"scatternd",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"indices", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"updates", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"shape", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* scatternd(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("scatternd");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ScatterNdSchema_TTTSh> parser("scatternd");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ScatterNd(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Shape>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TensorScatterNdUpdateSchema_TTTTB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& tensor, const std::shared_ptr<one::Tensor>& indices, const std::shared_ptr<one::Tensor>& updates, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::TensorScatterNdUpdate;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor tensor, Tensor indices, Tensor updates, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t TensorScatterNdUpdateSchema_TTTTB::max_args;
constexpr size_t TensorScatterNdUpdateSchema_TTTTB::max_pos_args;
constexpr char const* TensorScatterNdUpdateSchema_TTTTB::signature;
FunctionDef TensorScatterNdUpdateSchema_TTTTB::function_def = {
/*name*/"tensor_scatter_nd_update",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"tensor", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"indices", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"updates", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* tensor_scatter_nd_update(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("tensor_scatter_nd_update");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TensorScatterNdUpdateSchema_TTTTB> parser("tensor_scatter_nd_update");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::TensorScatterNdUpdate(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ScatterNdLikeSchema_TTTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& like, const std::shared_ptr<one::Tensor>& updates, const std::shared_ptr<one::Tensor>& indices);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScatterNdLike;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor like, Tensor updates, Tensor indices)";
  static FunctionDef function_def;
};

constexpr size_t ScatterNdLikeSchema_TTTT::max_args;
constexpr size_t ScatterNdLikeSchema_TTTT::max_pos_args;
constexpr char const* ScatterNdLikeSchema_TTTT::signature;
FunctionDef ScatterNdLikeSchema_TTTT::function_def = {
/*name*/"scatterndlike",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"like", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"updates", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"indices", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* scatterndlike(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("scatterndlike");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ScatterNdLikeSchema_TTTT> parser("scatterndlike");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ScatterNdLike(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MatMulSchema_TTTBBD {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other, bool transpose_a, bool transpose_b, double alpha);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::MatMul;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other, Bool transpose_a=False, Bool transpose_b=False, Double alpha=1.0)";
  static FunctionDef function_def;
};

constexpr size_t MatMulSchema_TTTBBD::max_args;
constexpr size_t MatMulSchema_TTTBBD::max_pos_args;
constexpr char const* MatMulSchema_TTTBBD::signature;
FunctionDef MatMulSchema_TTTBBD::function_def = {
/*name*/"matmul",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"transpose_a", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"transpose_b", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"alpha", /*default_value*/double(1.0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* matmul(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("matmul");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MatMulSchema_TTTBBD> parser("matmul");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::MatMul(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<bool>(), r[3].As<bool>(), r[4].As<double>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MatMulNoBroadCastSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& mat2);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::MatMulNoBroadCast;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor mat2)";
  static FunctionDef function_def;
};

constexpr size_t MatMulNoBroadCastSchema_TTT::max_args;
constexpr size_t MatMulNoBroadCastSchema_TTT::max_pos_args;
constexpr char const* MatMulNoBroadCastSchema_TTT::signature;
FunctionDef MatMulNoBroadCastSchema_TTT::function_def = {
/*name*/"mm",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"mat2", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* mm(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("mm");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MatMulNoBroadCastSchema_TTT> parser("mm");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::MatMulNoBroadCast(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct FusedMLPSchema_TTTtTtB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const TensorTuple& weights, const TensorTuple& biases, bool skip_final_activation);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::FusedMLP;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor x, TensorTuple weights, TensorTuple biases, Bool skip_final_activation)";
  static FunctionDef function_def;
};

constexpr size_t FusedMLPSchema_TTTtTtB::max_args;
constexpr size_t FusedMLPSchema_TTTtTtB::max_pos_args;
constexpr char const* FusedMLPSchema_TTTtTtB::signature;
FunctionDef FusedMLPSchema_TTTtTtB::function_def = {
/*name*/"fused_mlp",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weights", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"biases", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"skip_final_activation", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* fused_mlp(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("fused_mlp");
  PythonFrameGuard pf;
  static PythonArgParser<functional::FusedMLPSchema_TTTtTtB> parser("fused_mlp");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::FusedMLP(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<TensorTuple>(), r[2].As<TensorTuple>(), r[3].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct FusedMatmulBiasAddReluDropoutSchema_TTTtTtBFlG {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const TensorTuple& weights, const TensorTuple& biases, bool skip_final_activation, const std::vector<float>& dropout_rate_list, const Optional<one::Generator>& generator);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::FusedMatmulBiasAddReluDropout;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 6;
  static constexpr char const* signature = "Tensor (Tensor x, TensorTuple weights, TensorTuple biases, Bool skip_final_activation, FloatList dropout_rate_list, Generator generator=None)";
  static FunctionDef function_def;
};

constexpr size_t FusedMatmulBiasAddReluDropoutSchema_TTTtTtBFlG::max_args;
constexpr size_t FusedMatmulBiasAddReluDropoutSchema_TTTtTtBFlG::max_pos_args;
constexpr char const* FusedMatmulBiasAddReluDropoutSchema_TTTtTtBFlG::signature;
FunctionDef FusedMatmulBiasAddReluDropoutSchema_TTTtTtBFlG::function_def = {
/*name*/"fused_matmul_bias_add_relu_dropout",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weights", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"biases", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"skip_final_activation", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dropout_rate_list", /*value_type*/ValueTypeOf<std::vector<float>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* fused_matmul_bias_add_relu_dropout(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("fused_matmul_bias_add_relu_dropout");
  PythonFrameGuard pf;
  static PythonArgParser<functional::FusedMatmulBiasAddReluDropoutSchema_TTTtTtBFlG> parser("fused_matmul_bias_add_relu_dropout");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::FusedMatmulBiasAddReluDropout(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<TensorTuple>(), r[2].As<TensorTuple>(), r[3].As<bool>(), r[4].As<std::vector<float>>(), r[5].As<Optional<one::Generator>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BatchMatMulSchema_TTTBBD {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& a, const std::shared_ptr<one::Tensor>& b, bool transpose_a, bool transpose_b, double alpha);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BatchMatMul;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Tensor (Tensor a, Tensor b, Bool transpose_a=False, Bool transpose_b=False, Double alpha=1.0)";
  static FunctionDef function_def;
};

constexpr size_t BatchMatMulSchema_TTTBBD::max_args;
constexpr size_t BatchMatMulSchema_TTTBBD::max_pos_args;
constexpr char const* BatchMatMulSchema_TTTBBD::signature;
FunctionDef BatchMatMulSchema_TTTBBD::function_def = {
/*name*/"batch_matmul",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"a", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"b", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"transpose_a", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"transpose_b", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"alpha", /*default_value*/double(1.0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* batch_matmul(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("batch_matmul");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BatchMatMulSchema_TTTBBD> parser("batch_matmul");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BatchMatMul(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<bool>(), r[3].As<bool>(), r[4].As<double>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MatrixVectorProductSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& vec);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::MatrixVectorProduct;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor vec)";
  static FunctionDef function_def;
};

constexpr size_t MatrixVectorProductSchema_TTT::max_args;
constexpr size_t MatrixVectorProductSchema_TTT::max_pos_args;
constexpr char const* MatrixVectorProductSchema_TTT::signature;
FunctionDef MatrixVectorProductSchema_TTT::function_def = {
/*name*/"matrix_vector_product",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"vec", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* matrix_vector_product(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("matrix_vector_product");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MatrixVectorProductSchema_TTT> parser("matrix_vector_product");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::MatrixVectorProduct(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TensorDotSchema_TTTI32lI32l {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& a, const std::shared_ptr<one::Tensor>& b, const std::vector<int32_t>& dims_a, const std::vector<int32_t>& dims_b);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::TensorDot;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor a, Tensor b, Int32List dims_a, Int32List dims_b)";
  static FunctionDef function_def;
};

constexpr size_t TensorDotSchema_TTTI32lI32l::max_args;
constexpr size_t TensorDotSchema_TTTI32lI32l::max_pos_args;
constexpr char const* TensorDotSchema_TTTI32lI32l::signature;
FunctionDef TensorDotSchema_TTTI32lI32l::function_def = {
/*name*/"tensordot",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"a", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"b", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dims_a", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dims_b", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct TensorDotIntDimsSchema_TTTI32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& a, const std::shared_ptr<one::Tensor>& b, int32_t dims);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::TensorDotIntDims;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor a, Tensor b, Int32 dims)";
  static FunctionDef function_def;
};

constexpr size_t TensorDotIntDimsSchema_TTTI32::max_args;
constexpr size_t TensorDotIntDimsSchema_TTTI32::max_pos_args;
constexpr char const* TensorDotIntDimsSchema_TTTI32::signature;
FunctionDef TensorDotIntDimsSchema_TTTI32::function_def = {
/*name*/"tensordot",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"a", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"b", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dims", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* tensordot(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("tensordot");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TensorDotSchema_TTTI32lI32l, functional::TensorDotIntDimsSchema_TTTI32> parser("tensordot");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::TensorDot(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::vector<int32_t>>(), r[3].As<std::vector<int32_t>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::TensorDotIntDims(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct L1LossSchema_TTTS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const std::string& reduction);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::L1Loss;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor target, String reduction=\"mean\")";
  static FunctionDef function_def;
};

constexpr size_t L1LossSchema_TTTS::max_args;
constexpr size_t L1LossSchema_TTTS::max_pos_args;
constexpr char const* L1LossSchema_TTTS::signature;
FunctionDef L1LossSchema_TTTS::function_def = {
/*name*/"l1_loss",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"target", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"reduction", /*default_value*/std::string("mean"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* l1_loss(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("l1_loss");
  PythonFrameGuard pf;
  static PythonArgParser<functional::L1LossSchema_TTTS> parser("l1_loss");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::L1Loss(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MseLossSchema_TTTS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const std::string& reduction);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::MseLoss;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor target, String reduction=\"mean\")";
  static FunctionDef function_def;
};

constexpr size_t MseLossSchema_TTTS::max_args;
constexpr size_t MseLossSchema_TTTS::max_pos_args;
constexpr char const* MseLossSchema_TTTS::signature;
FunctionDef MseLossSchema_TTTS::function_def = {
/*name*/"mse_loss",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"target", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"reduction", /*default_value*/std::string("mean"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* mse_loss(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("mse_loss");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MseLossSchema_TTTS> parser("mse_loss");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::MseLoss(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct KLDivLossSchema_TTTBS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, bool log_target, const std::string& reduction);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::KLDivLoss;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor target, Bool log_target, String reduction=\"mean\")";
  static FunctionDef function_def;
};

constexpr size_t KLDivLossSchema_TTTBS::max_args;
constexpr size_t KLDivLossSchema_TTTBS::max_pos_args;
constexpr char const* KLDivLossSchema_TTTBS::signature;
FunctionDef KLDivLossSchema_TTTBS::function_def = {
/*name*/"kl_div_loss",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"target", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"log_target", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"reduction", /*default_value*/std::string("mean"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* kl_div_loss(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("kl_div_loss");
  PythonFrameGuard pf;
  static PythonArgParser<functional::KLDivLossSchema_TTTBS> parser("kl_div_loss");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::KLDivLoss(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<bool>(), r[3].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct NLLLossSchema_TTTTI64S {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, int64_t ignore_index, const std::string& reduction);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::NLLLoss;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor target, Tensor weight=None, Int64 ignore_index, String reduction)";
  static FunctionDef function_def;
};

constexpr size_t NLLLossSchema_TTTTI64S::max_args;
constexpr size_t NLLLossSchema_TTTTI64S::max_pos_args;
constexpr char const* NLLLossSchema_TTTTI64S::signature;
FunctionDef NLLLossSchema_TTTTI64S::function_def = {
/*name*/"nll_loss",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"target", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"ignore_index", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"reduction", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* nll_loss(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("nll_loss");
  PythonFrameGuard pf;
  static PythonArgParser<functional::NLLLossSchema_TTTTI64S> parser("nll_loss");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::NLLLoss(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Optional<one::Tensor>>(), r[3].As<int64_t>(), r[4].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BinaryCrossEntropyLossSchema_TTTTS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, const std::string& reduction);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BinaryCrossEntropyLoss;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor target, Tensor weight=None, String reduction=\"mean\")";
  static FunctionDef function_def;
};

constexpr size_t BinaryCrossEntropyLossSchema_TTTTS::max_args;
constexpr size_t BinaryCrossEntropyLossSchema_TTTTS::max_pos_args;
constexpr char const* BinaryCrossEntropyLossSchema_TTTTS::signature;
FunctionDef BinaryCrossEntropyLossSchema_TTTTS::function_def = {
/*name*/"binary_cross_entropy_loss",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"target", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"reduction", /*default_value*/std::string("mean"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* binary_cross_entropy_loss(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("binary_cross_entropy_loss");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BinaryCrossEntropyLossSchema_TTTTS> parser("binary_cross_entropy_loss");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BinaryCrossEntropyLoss(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Optional<one::Tensor>>(), r[3].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BinaryCrossEntropyWithLogitsLossSchema_TTTTTS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, const Optional<one::Tensor>& pos_weight, const std::string& reduction);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BinaryCrossEntropyWithLogitsLoss;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor target, Tensor weight=None, Tensor pos_weight=None, String reduction=\"mean\")";
  static FunctionDef function_def;
};

constexpr size_t BinaryCrossEntropyWithLogitsLossSchema_TTTTTS::max_args;
constexpr size_t BinaryCrossEntropyWithLogitsLossSchema_TTTTTS::max_pos_args;
constexpr char const* BinaryCrossEntropyWithLogitsLossSchema_TTTTTS::signature;
FunctionDef BinaryCrossEntropyWithLogitsLossSchema_TTTTTS::function_def = {
/*name*/"binary_cross_entropy_with_logits_loss",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"target", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"pos_weight", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"reduction", /*default_value*/std::string("mean"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* binary_cross_entropy_with_logits_loss(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("binary_cross_entropy_with_logits_loss");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BinaryCrossEntropyWithLogitsLossSchema_TTTTTS> parser("binary_cross_entropy_with_logits_loss");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BinaryCrossEntropyWithLogitsLoss(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Optional<one::Tensor>>(), r[3].As<Optional<one::Tensor>>(), r[4].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BinaryCrossEntropyWithLogitsLossGradSchema_TTTTTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, const Optional<one::Tensor>& pos_weight);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BinaryCrossEntropyWithLogitsLossGrad;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Tensor (Tensor dy, Tensor input, Tensor target, Tensor weight=None, Tensor pos_weight=None)";
  static FunctionDef function_def;
};

constexpr size_t BinaryCrossEntropyWithLogitsLossGradSchema_TTTTTT::max_args;
constexpr size_t BinaryCrossEntropyWithLogitsLossGradSchema_TTTTTT::max_pos_args;
constexpr char const* BinaryCrossEntropyWithLogitsLossGradSchema_TTTTTT::signature;
FunctionDef BinaryCrossEntropyWithLogitsLossGradSchema_TTTTTT::function_def = {
/*name*/"binary_cross_entropy_with_logits_loss_grad",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"dy", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"target", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"pos_weight", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* binary_cross_entropy_with_logits_loss_grad(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("binary_cross_entropy_with_logits_loss_grad");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BinaryCrossEntropyWithLogitsLossGradSchema_TTTTTT> parser("binary_cross_entropy_with_logits_loss_grad");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BinaryCrossEntropyWithLogitsLossGrad(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<Optional<one::Tensor>>(), r[4].As<Optional<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SparseCrossEntropySchema_TTTI64 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& prediction, const std::shared_ptr<one::Tensor>& label, int64_t depth);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::SparseCrossEntropy;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor prediction, Tensor label, Int64 depth)";
  static FunctionDef function_def;
};

constexpr size_t SparseCrossEntropySchema_TTTI64::max_args;
constexpr size_t SparseCrossEntropySchema_TTTI64::max_pos_args;
constexpr char const* SparseCrossEntropySchema_TTTI64::signature;
FunctionDef SparseCrossEntropySchema_TTTI64::function_def = {
/*name*/"sparse_cross_entropy",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"prediction", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"label", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"depth", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* sparse_cross_entropy(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("sparse_cross_entropy");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SparseCrossEntropySchema_TTTI64> parser("sparse_cross_entropy");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::SparseCrossEntropy(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SparseCrossEntropyMsSchema_TTTI64 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& prediction, const std::shared_ptr<one::Tensor>& label, int64_t depth);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::SparseCrossEntropyMs;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor prediction, Tensor label, Int64 depth)";
  static FunctionDef function_def;
};

constexpr size_t SparseCrossEntropyMsSchema_TTTI64::max_args;
constexpr size_t SparseCrossEntropyMsSchema_TTTI64::max_pos_args;
constexpr char const* SparseCrossEntropyMsSchema_TTTI64::signature;
FunctionDef SparseCrossEntropyMsSchema_TTTI64::function_def = {
/*name*/"distributed_sparse_cross_entropy",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"prediction", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"label", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"depth", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* distributed_sparse_cross_entropy(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("distributed_sparse_cross_entropy");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SparseCrossEntropyMsSchema_TTTI64> parser("distributed_sparse_cross_entropy");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::SparseCrossEntropyMs(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct CrossEntropySchema_TTTTI64SD {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& target, const Optional<one::Tensor>& weight, int64_t ignore_index, const std::string& reduction, double label_smoothing);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::CrossEntropy;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 6;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor target, Tensor weight=None, Int64 ignore_index=-100, String reduction=\"mean\", Double label_smoothing=0.0)";
  static FunctionDef function_def;
};

constexpr size_t CrossEntropySchema_TTTTI64SD::max_args;
constexpr size_t CrossEntropySchema_TTTTI64SD::max_pos_args;
constexpr char const* CrossEntropySchema_TTTTI64SD::signature;
FunctionDef CrossEntropySchema_TTTTI64SD::function_def = {
/*name*/"cross_entropy",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"target", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"ignore_index", /*default_value*/int64_t(-100), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"reduction", /*default_value*/std::string("mean"), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"label_smoothing", /*default_value*/double(0.0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* cross_entropy(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("cross_entropy");
  PythonFrameGuard pf;
  static PythonArgParser<functional::CrossEntropySchema_TTTTI64SD> parser("cross_entropy");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::CrossEntropy(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Optional<one::Tensor>>(), r[3].As<int64_t>(), r[4].As<std::string>(), r[5].As<double>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SparseSoftmaxCrossEntropySchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& logits, const std::shared_ptr<one::Tensor>& label);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::SparseSoftmaxCrossEntropy;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor logits, Tensor label)";
  static FunctionDef function_def;
};

constexpr size_t SparseSoftmaxCrossEntropySchema_TTT::max_args;
constexpr size_t SparseSoftmaxCrossEntropySchema_TTT::max_pos_args;
constexpr char const* SparseSoftmaxCrossEntropySchema_TTT::signature;
FunctionDef SparseSoftmaxCrossEntropySchema_TTT::function_def = {
/*name*/"sparse_softmax_cross_entropy",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"logits", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"label", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* sparse_softmax_cross_entropy(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("sparse_softmax_cross_entropy");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SparseSoftmaxCrossEntropySchema_TTT> parser("sparse_softmax_cross_entropy");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::SparseSoftmaxCrossEntropy(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SoftmaxCrossEntropySchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& logits, const std::shared_ptr<one::Tensor>& label);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::SoftmaxCrossEntropy;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor logits, Tensor label)";
  static FunctionDef function_def;
};

constexpr size_t SoftmaxCrossEntropySchema_TTT::max_args;
constexpr size_t SoftmaxCrossEntropySchema_TTT::max_pos_args;
constexpr char const* SoftmaxCrossEntropySchema_TTT::signature;
FunctionDef SoftmaxCrossEntropySchema_TTT::function_def = {
/*name*/"softmax_cross_entropy",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"logits", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"label", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* softmax_cross_entropy(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("softmax_cross_entropy");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SoftmaxCrossEntropySchema_TTT> parser("softmax_cross_entropy");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::SoftmaxCrossEntropy(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SoftmaxCrossEntropyGradSchema_TTTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& dy, const std::shared_ptr<one::Tensor>& label, const std::shared_ptr<one::Tensor>& prob);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::SoftmaxCrossEntropyGrad;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor dy, Tensor label, Tensor prob)";
  static FunctionDef function_def;
};

constexpr size_t SoftmaxCrossEntropyGradSchema_TTTT::max_args;
constexpr size_t SoftmaxCrossEntropyGradSchema_TTTT::max_pos_args;
constexpr char const* SoftmaxCrossEntropyGradSchema_TTTT::signature;
FunctionDef SoftmaxCrossEntropyGradSchema_TTTT::function_def = {
/*name*/"softmax_cross_entropy_grad",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"dy", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"label", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"prob", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* softmax_cross_entropy_grad(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("softmax_cross_entropy_grad");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SoftmaxCrossEntropyGradSchema_TTTT> parser("softmax_cross_entropy_grad");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::SoftmaxCrossEntropyGrad(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SmoothL1LossSchema_TTTFS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& logits, const std::shared_ptr<one::Tensor>& label, float beta, const std::string& reduction);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::SmoothL1Loss;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor logits, Tensor label, Float beta, String reduction)";
  static FunctionDef function_def;
};

constexpr size_t SmoothL1LossSchema_TTTFS::max_args;
constexpr size_t SmoothL1LossSchema_TTTFS::max_pos_args;
constexpr char const* SmoothL1LossSchema_TTTFS::signature;
FunctionDef SmoothL1LossSchema_TTTFS::function_def = {
/*name*/"smooth_l1_loss",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"logits", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"label", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"beta", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"reduction", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* smooth_l1_loss(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("smooth_l1_loss");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SmoothL1LossSchema_TTTFS> parser("smooth_l1_loss");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::SmoothL1Loss(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<float>(), r[3].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct CombinedMarginLossSchema_TTTFFF {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& label, float m1, float m2, float m3);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::CombinedMarginLoss;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor label, Float m1, Float m2, Float m3)";
  static FunctionDef function_def;
};

constexpr size_t CombinedMarginLossSchema_TTTFFF::max_args;
constexpr size_t CombinedMarginLossSchema_TTTFFF::max_pos_args;
constexpr char const* CombinedMarginLossSchema_TTTFFF::signature;
FunctionDef CombinedMarginLossSchema_TTTFFF::function_def = {
/*name*/"combined_margin_loss",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"label", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"m1", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"m2", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"m3", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* combined_margin_loss(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("combined_margin_loss");
  PythonFrameGuard pf;
  static PythonArgParser<functional::CombinedMarginLossSchema_TTTFFF> parser("combined_margin_loss");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::CombinedMarginLoss(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<float>(), r[3].As<float>(), r[4].As<float>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TripletMarginLossSchema_TTTTFFFBS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& anchor, const std::shared_ptr<one::Tensor>& positive, const std::shared_ptr<one::Tensor>& negative, float margin, float p, float eps, bool swap, const std::string& reduction);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::TripletMarginLoss;
  static constexpr size_t max_args = 8;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor anchor, Tensor positive, Tensor negative, *, Float margin, Float p, Float eps, Bool swap, String reduction)";
  static FunctionDef function_def;
};

constexpr size_t TripletMarginLossSchema_TTTTFFFBS::max_args;
constexpr size_t TripletMarginLossSchema_TTTTFFFBS::max_pos_args;
constexpr char const* TripletMarginLossSchema_TTTTFFFBS::signature;
FunctionDef TripletMarginLossSchema_TTTTFFFBS::function_def = {
/*name*/"triplet_margin_loss",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"anchor", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"positive", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"negative", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"margin", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"p", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"eps", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"swap", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"reduction", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* triplet_margin_loss(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("triplet_margin_loss");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TripletMarginLossSchema_TTTTFFFBS> parser("triplet_margin_loss");
  ParsedArgs<8> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::TripletMarginLoss(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<float>(), r[4].As<float>(), r[5].As<float>(), r[6].As<bool>(), r[7].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MarginRankingLossSchema_TTTTFS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input_1, const std::shared_ptr<one::Tensor>& input_2, const std::shared_ptr<one::Tensor>& target, float margin, const std::string& reduction);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::MarginRankingLoss;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Tensor (Tensor input_1, Tensor input_2, Tensor target, Float margin, String reduction)";
  static FunctionDef function_def;
};

constexpr size_t MarginRankingLossSchema_TTTTFS::max_args;
constexpr size_t MarginRankingLossSchema_TTTTFS::max_pos_args;
constexpr char const* MarginRankingLossSchema_TTTTFS::signature;
FunctionDef MarginRankingLossSchema_TTTTFS::function_def = {
/*name*/"margin_ranking_loss",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input_1", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"input_2", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"target", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"margin", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"reduction", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* margin_ranking_loss(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("margin_ranking_loss");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MarginRankingLossSchema_TTTTFS> parser("margin_ranking_loss");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::MarginRankingLoss(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<float>(), r[4].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct CtcLossSchema_TTTTTI64I64BS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& log_probs, const std::shared_ptr<one::Tensor>& targets, const std::shared_ptr<one::Tensor>& input_lengths, const std::shared_ptr<one::Tensor>& target_lengths, int64_t max_target_length, int64_t blank, bool zero_infinity, const std::string& reduction);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::CtcLoss;
  static constexpr size_t max_args = 8;
  static constexpr size_t max_pos_args = 8;
  static constexpr char const* signature = "Tensor (Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, Int64 max_target_length, Int64 blank, Bool zero_infinity, String reduction)";
  static FunctionDef function_def;
};

constexpr size_t CtcLossSchema_TTTTTI64I64BS::max_args;
constexpr size_t CtcLossSchema_TTTTTI64I64BS::max_pos_args;
constexpr char const* CtcLossSchema_TTTTTI64I64BS::signature;
FunctionDef CtcLossSchema_TTTTTI64I64BS::function_def = {
/*name*/"ctc_loss",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"log_probs", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"targets", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"input_lengths", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"target_lengths", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"max_target_length", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"blank", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"zero_infinity", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"reduction", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* ctc_loss(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("ctc_loss");
  PythonFrameGuard pf;
  static PythonArgParser<functional::CtcLossSchema_TTTTTI64I64BS> parser("ctc_loss");
  ParsedArgs<8> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::CtcLoss(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<std::shared_ptr<one::Tensor>>(), r[4].As<int64_t>(), r[5].As<int64_t>(), r[6].As<bool>(), r[7].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AffineGridSchema_TTShB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& theta, const Shape& size, bool align_corners);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::AffineGrid;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor theta, *, Shape size, Bool align_corners)";
  static FunctionDef function_def;
};

constexpr size_t AffineGridSchema_TTShB::max_args;
constexpr size_t AffineGridSchema_TTShB::max_pos_args;
constexpr char const* AffineGridSchema_TTShB::signature;
FunctionDef AffineGridSchema_TTShB::function_def = {
/*name*/"affine_grid",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"theta", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"size", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"align_corners", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* affine_grid(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("affine_grid");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AffineGridSchema_TTShB> parser("affine_grid");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::AffineGrid(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Shape>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct GridSampleSchema_TTTSSB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& grid, const std::string& interpolation_mode, const std::string& padding_mode, bool align_corners);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GridSample;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor grid, *, String interpolation_mode, String padding_mode, Bool align_corners)";
  static FunctionDef function_def;
};

constexpr size_t GridSampleSchema_TTTSSB::max_args;
constexpr size_t GridSampleSchema_TTTSSB::max_pos_args;
constexpr char const* GridSampleSchema_TTTSSB::signature;
FunctionDef GridSampleSchema_TTTSSB::function_def = {
/*name*/"grid_sample",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"grid", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"interpolation_mode", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"padding_mode", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"align_corners", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* grid_sample(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("grid_sample");
  PythonFrameGuard pf;
  static PythonArgParser<functional::GridSampleSchema_TTTSSB> parser("grid_sample");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::GridSample(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::string>(), r[3].As<std::string>(), r[4].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct WhereSchema_TTTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& condition, const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Where;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor condition, Tensor x, Tensor y)";
  static FunctionDef function_def;
};

constexpr size_t WhereSchema_TTTT::max_args;
constexpr size_t WhereSchema_TTTT::max_pos_args;
constexpr char const* WhereSchema_TTTT::signature;
FunctionDef WhereSchema_TTTT::function_def = {
/*name*/"where",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"condition", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"y", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct WhereScalarXSchema_TTScT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& condition, const Scalar& x, const std::shared_ptr<one::Tensor>& y);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::WhereScalarX;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor condition, Scalar x, Tensor y)";
  static FunctionDef function_def;
};

constexpr size_t WhereScalarXSchema_TTScT::max_args;
constexpr size_t WhereScalarXSchema_TTScT::max_pos_args;
constexpr char const* WhereScalarXSchema_TTScT::signature;
FunctionDef WhereScalarXSchema_TTScT::function_def = {
/*name*/"where",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"condition", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"y", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct WhereScalarYSchema_TTTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& condition, const std::shared_ptr<one::Tensor>& x, const Scalar& y);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::WhereScalarY;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor condition, Tensor x, Scalar y)";
  static FunctionDef function_def;
};

constexpr size_t WhereScalarYSchema_TTTSc::max_args;
constexpr size_t WhereScalarYSchema_TTTSc::max_pos_args;
constexpr char const* WhereScalarYSchema_TTTSc::signature;
FunctionDef WhereScalarYSchema_TTTSc::function_def = {
/*name*/"where",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"condition", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"y", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct WhereScalarXYSchema_TTScSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& condition, const Scalar& x, const Scalar& y);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::WhereScalarXY;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor condition, Scalar x, Scalar y)";
  static FunctionDef function_def;
};

constexpr size_t WhereScalarXYSchema_TTScSc::max_args;
constexpr size_t WhereScalarXYSchema_TTScSc::max_pos_args;
constexpr char const* WhereScalarXYSchema_TTScSc::signature;
FunctionDef WhereScalarXYSchema_TTScSc::function_def = {
/*name*/"where",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"condition", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"y", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* where(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("where");
  PythonFrameGuard pf;
  static PythonArgParser<functional::WhereSchema_TTTT, functional::WhereScalarXSchema_TTScT, functional::WhereScalarYSchema_TTTSc, functional::WhereScalarXYSchema_TTScSc> parser("where");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Where(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::WhereScalarX(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>(), r[2].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::WhereScalarY(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Scalar>()));
  }
  if (idx == 3) {
    return CastToPyObject(functional::WhereScalarXY(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>(), r[2].As<Scalar>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MaskedFillSchema_TTTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& mask, const Scalar& value);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::MaskedFill;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor mask, Scalar value)";
  static FunctionDef function_def;
};

constexpr size_t MaskedFillSchema_TTTSc::max_args;
constexpr size_t MaskedFillSchema_TTTSc::max_pos_args;
constexpr char const* MaskedFillSchema_TTTSc::signature;
FunctionDef MaskedFillSchema_TTTSc::function_def = {
/*name*/"masked_fill",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"mask", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"value", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* masked_fill(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("masked_fill");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MaskedFillSchema_TTTSc> parser("masked_fill");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::MaskedFill(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Scalar>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MaskedFillInplaceSchema_TTTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& mask, const Scalar& value);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::MaskedFillInplace;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor mask, Scalar value)";
  static FunctionDef function_def;
};

constexpr size_t MaskedFillInplaceSchema_TTTSc::max_args;
constexpr size_t MaskedFillInplaceSchema_TTTSc::max_pos_args;
constexpr char const* MaskedFillInplaceSchema_TTTSc::signature;
FunctionDef MaskedFillInplaceSchema_TTTSc::function_def = {
/*name*/"masked_fill_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"mask", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"value", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* masked_fill_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("masked_fill_");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MaskedFillInplaceSchema_TTTSc> parser("masked_fill_");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::MaskedFillInplace(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Scalar>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MovedimIntSchema_TTI32I32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int32_t source, int32_t destination);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::MovedimInt;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Int32 source, Int32 destination)";
  static FunctionDef function_def;
};

constexpr size_t MovedimIntSchema_TTI32I32::max_args;
constexpr size_t MovedimIntSchema_TTI32I32::max_pos_args;
constexpr char const* MovedimIntSchema_TTI32I32::signature;
FunctionDef MovedimIntSchema_TTI32I32::function_def = {
/*name*/"movedim",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"source", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"destination", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct MovedimVecSchema_TTI32lI32l {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& source, const std::vector<int32_t>& destination);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::MovedimVec;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Int32List source, Int32List destination)";
  static FunctionDef function_def;
};

constexpr size_t MovedimVecSchema_TTI32lI32l::max_args;
constexpr size_t MovedimVecSchema_TTI32lI32l::max_pos_args;
constexpr char const* MovedimVecSchema_TTI32lI32l::signature;
FunctionDef MovedimVecSchema_TTI32lI32l::function_def = {
/*name*/"movedim",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"source", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"destination", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* movedim(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("movedim");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MovedimIntSchema_TTI32I32, functional::MovedimVecSchema_TTI32lI32l> parser("movedim");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::MovedimInt(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>(), r[2].As<int32_t>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::MovedimVec(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<std::vector<int32_t>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TensorSplitIntSchema_TtTI32I32 {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, int32_t indices_or_sections, int32_t dim);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::TensorSplitInt;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "TensorTuple (Tensor input, Int32 indices_or_sections, Int32 dim=0)";
  static FunctionDef function_def;
};

constexpr size_t TensorSplitIntSchema_TtTI32I32::max_args;
constexpr size_t TensorSplitIntSchema_TtTI32I32::max_pos_args;
constexpr char const* TensorSplitIntSchema_TtTI32I32::signature;
FunctionDef TensorSplitIntSchema_TtTI32I32::function_def = {
/*name*/"tensor_split",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"indices_or_sections", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/int32_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct TensorSplitVecSchema_TtTI32lI32 {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& indices_or_sections, int32_t dim);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::TensorSplitVec;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "TensorTuple (Tensor input, Int32List indices_or_sections, Int32 dim=0)";
  static FunctionDef function_def;
};

constexpr size_t TensorSplitVecSchema_TtTI32lI32::max_args;
constexpr size_t TensorSplitVecSchema_TtTI32lI32::max_pos_args;
constexpr char const* TensorSplitVecSchema_TtTI32lI32::signature;
FunctionDef TensorSplitVecSchema_TtTI32lI32::function_def = {
/*name*/"tensor_split",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"indices_or_sections", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/int32_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* tensor_split(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("tensor_split");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TensorSplitIntSchema_TtTI32I32, functional::TensorSplitVecSchema_TtTI32lI32> parser("tensor_split");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::TensorSplitInt(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>(), r[2].As<int32_t>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::TensorSplitVec(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct HsplitIntSchema_TtTI32 {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, int32_t indices_or_sections);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::HsplitInt;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "TensorTuple (Tensor input, Int32 indices_or_sections)";
  static FunctionDef function_def;
};

constexpr size_t HsplitIntSchema_TtTI32::max_args;
constexpr size_t HsplitIntSchema_TtTI32::max_pos_args;
constexpr char const* HsplitIntSchema_TtTI32::signature;
FunctionDef HsplitIntSchema_TtTI32::function_def = {
/*name*/"hsplit",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"indices_or_sections", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct HsplitVecSchema_TtTI32l {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& indices_or_sections);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::HsplitVec;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "TensorTuple (Tensor input, Int32List indices_or_sections)";
  static FunctionDef function_def;
};

constexpr size_t HsplitVecSchema_TtTI32l::max_args;
constexpr size_t HsplitVecSchema_TtTI32l::max_pos_args;
constexpr char const* HsplitVecSchema_TtTI32l::signature;
FunctionDef HsplitVecSchema_TtTI32l::function_def = {
/*name*/"hsplit",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"indices_or_sections", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* hsplit(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("hsplit");
  PythonFrameGuard pf;
  static PythonArgParser<functional::HsplitIntSchema_TtTI32, functional::HsplitVecSchema_TtTI32l> parser("hsplit");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::HsplitInt(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::HsplitVec(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct VsplitIntSchema_TtTI32 {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, int32_t indices_or_sections);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::VsplitInt;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "TensorTuple (Tensor input, Int32 indices_or_sections)";
  static FunctionDef function_def;
};

constexpr size_t VsplitIntSchema_TtTI32::max_args;
constexpr size_t VsplitIntSchema_TtTI32::max_pos_args;
constexpr char const* VsplitIntSchema_TtTI32::signature;
FunctionDef VsplitIntSchema_TtTI32::function_def = {
/*name*/"vsplit",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"indices_or_sections", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct VsplitVecSchema_TtTI32l {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& indices_or_sections);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::VsplitVec;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "TensorTuple (Tensor input, Int32List indices_or_sections)";
  static FunctionDef function_def;
};

constexpr size_t VsplitVecSchema_TtTI32l::max_args;
constexpr size_t VsplitVecSchema_TtTI32l::max_pos_args;
constexpr char const* VsplitVecSchema_TtTI32l::signature;
FunctionDef VsplitVecSchema_TtTI32l::function_def = {
/*name*/"vsplit",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"indices_or_sections", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* vsplit(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("vsplit");
  PythonFrameGuard pf;
  static PythonArgParser<functional::VsplitIntSchema_TtTI32, functional::VsplitVecSchema_TtTI32l> parser("vsplit");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::VsplitInt(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::VsplitVec(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct NegativeSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Negative;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t NegativeSchema_TT::max_args;
constexpr size_t NegativeSchema_TT::max_pos_args;
constexpr char const* NegativeSchema_TT::signature;
FunctionDef NegativeSchema_TT::function_def = {
/*name*/"negative",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* negative(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("negative");
  PythonFrameGuard pf;
  static PythonArgParser<functional::NegativeSchema_TT> parser("negative");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Negative(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct LayerNormAffineSchema_TTTTI64I64D {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& gamma, const std::shared_ptr<one::Tensor>& beta, int64_t begin_norm_axis, int64_t begin_params_axis, double epsilon);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::LayerNormAffine;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 6;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor gamma, Tensor beta, Int64 begin_norm_axis, Int64 begin_params_axis, Double epsilon)";
  static FunctionDef function_def;
};

constexpr size_t LayerNormAffineSchema_TTTTI64I64D::max_args;
constexpr size_t LayerNormAffineSchema_TTTTI64I64D::max_pos_args;
constexpr char const* LayerNormAffineSchema_TTTTI64I64D::signature;
FunctionDef LayerNormAffineSchema_TTTTI64I64D::function_def = {
/*name*/"layer_norm_affine",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"gamma", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"beta", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"begin_norm_axis", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"begin_params_axis", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"epsilon", /*value_type*/ValueTypeOf<double>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* layer_norm_affine(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("layer_norm_affine");
  PythonFrameGuard pf;
  static PythonArgParser<functional::LayerNormAffineSchema_TTTTI64I64D> parser("layer_norm_affine");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::LayerNormAffine(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<int64_t>(), r[4].As<int64_t>(), r[5].As<double>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct LayerNormSchema_TTI64I64D {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, int64_t begin_norm_axis, int64_t begin_params_axis, double epsilon);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::LayerNorm;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor x, Int64 begin_norm_axis, Int64 begin_params_axis, Double epsilon)";
  static FunctionDef function_def;
};

constexpr size_t LayerNormSchema_TTI64I64D::max_args;
constexpr size_t LayerNormSchema_TTI64I64D::max_pos_args;
constexpr char const* LayerNormSchema_TTI64I64D::signature;
FunctionDef LayerNormSchema_TTI64I64D::function_def = {
/*name*/"layer_norm",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"begin_norm_axis", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"begin_params_axis", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"epsilon", /*value_type*/ValueTypeOf<double>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* layer_norm(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("layer_norm");
  PythonFrameGuard pf;
  static PythonArgParser<functional::LayerNormSchema_TTI64I64D> parser("layer_norm");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::LayerNorm(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>(), r[2].As<int64_t>(), r[3].As<double>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct GroupNormSchema_TTTTBI32D {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const Optional<one::Tensor>& gamma, const Optional<one::Tensor>& beta, bool affine, int32_t num_groups, double epsilon);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GroupNorm;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 6;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor gamma=None, Tensor beta=None, Bool affine, Int32 num_groups, Double epsilon)";
  static FunctionDef function_def;
};

constexpr size_t GroupNormSchema_TTTTBI32D::max_args;
constexpr size_t GroupNormSchema_TTTTBI32D::max_pos_args;
constexpr char const* GroupNormSchema_TTTTBI32D::signature;
FunctionDef GroupNormSchema_TTTTBI32D::function_def = {
/*name*/"group_norm",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"gamma", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"beta", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"affine", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"num_groups", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"epsilon", /*value_type*/ValueTypeOf<double>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* group_norm(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("group_norm");
  PythonFrameGuard pf;
  static PythonArgParser<functional::GroupNormSchema_TTTTBI32D> parser("group_norm");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::GroupNorm(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<one::Tensor>>(), r[2].As<Optional<one::Tensor>>(), r[3].As<bool>(), r[4].As<int32_t>(), r[5].As<double>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TFAvgPool2DSchema_TTI32lI32lSI32lI32lSB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& stride, const std::string& padding, const std::vector<int32_t>& padding_before, const std::vector<int32_t>& padding_after, const std::string& data_format, bool ceil_mode);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::TFAvgPool2D;
  static constexpr size_t max_args = 8;
  static constexpr size_t max_pos_args = 8;
  static constexpr char const* signature = "Tensor (Tensor x, Int32List kernel_size, Int32List stride, String padding, Int32List padding_before, Int32List padding_after, String data_format=\"channels_first\", Bool ceil_mode=False)";
  static FunctionDef function_def;
};

constexpr size_t TFAvgPool2DSchema_TTI32lI32lSI32lI32lSB::max_args;
constexpr size_t TFAvgPool2DSchema_TTI32lI32lSI32lI32lSB::max_pos_args;
constexpr char const* TFAvgPool2DSchema_TTI32lI32lSI32lI32lSB::signature;
FunctionDef TFAvgPool2DSchema_TTI32lI32lSI32lI32lSB::function_def = {
/*name*/"avg_pool2d_nhwc",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"kernel_size", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"stride", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"padding", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"padding_before", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"padding_after", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"data_format", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"ceil_mode", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* avg_pool2d_nhwc(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("avg_pool2d_nhwc");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TFAvgPool2DSchema_TTI32lI32lSI32lI32lSB> parser("avg_pool2d_nhwc");
  ParsedArgs<8> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::TFAvgPool2D(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<std::vector<int32_t>>(), r[3].As<std::string>(), r[4].As<std::vector<int32_t>>(), r[5].As<std::vector<int32_t>>(), r[6].As<std::string>(), r[7].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AdaptiveAvgPool1DSchema_TTI64l {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& output_size);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::AdaptiveAvgPool1D;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Int64List output_size)";
  static FunctionDef function_def;
};

constexpr size_t AdaptiveAvgPool1DSchema_TTI64l::max_args;
constexpr size_t AdaptiveAvgPool1DSchema_TTI64l::max_pos_args;
constexpr char const* AdaptiveAvgPool1DSchema_TTI64l::signature;
FunctionDef AdaptiveAvgPool1DSchema_TTI64l::function_def = {
/*name*/"adaptive_avg_pool1d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"output_size", /*value_type*/ValueTypeOf<std::vector<int64_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* adaptive_avg_pool1d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("adaptive_avg_pool1d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AdaptiveAvgPool1DSchema_TTI64l> parser("adaptive_avg_pool1d");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::AdaptiveAvgPool1D(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int64_t>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AdaptiveAvgPool2DSchema_TTI64l {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& output_size);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::AdaptiveAvgPool2D;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Int64List output_size)";
  static FunctionDef function_def;
};

constexpr size_t AdaptiveAvgPool2DSchema_TTI64l::max_args;
constexpr size_t AdaptiveAvgPool2DSchema_TTI64l::max_pos_args;
constexpr char const* AdaptiveAvgPool2DSchema_TTI64l::signature;
FunctionDef AdaptiveAvgPool2DSchema_TTI64l::function_def = {
/*name*/"adaptive_avg_pool2d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"output_size", /*value_type*/ValueTypeOf<std::vector<int64_t>>(), /*size*/2, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* adaptive_avg_pool2d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("adaptive_avg_pool2d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AdaptiveAvgPool2DSchema_TTI64l> parser("adaptive_avg_pool2d");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::AdaptiveAvgPool2D(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int64_t>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AdaptiveAvgPool3DSchema_TTI64l {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& output_size);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::AdaptiveAvgPool3D;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Int64List output_size)";
  static FunctionDef function_def;
};

constexpr size_t AdaptiveAvgPool3DSchema_TTI64l::max_args;
constexpr size_t AdaptiveAvgPool3DSchema_TTI64l::max_pos_args;
constexpr char const* AdaptiveAvgPool3DSchema_TTI64l::signature;
FunctionDef AdaptiveAvgPool3DSchema_TTI64l::function_def = {
/*name*/"adaptive_avg_pool3d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"output_size", /*value_type*/ValueTypeOf<std::vector<int64_t>>(), /*size*/3, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* adaptive_avg_pool3d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("adaptive_avg_pool3d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AdaptiveAvgPool3DSchema_TTI64l> parser("adaptive_avg_pool3d");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::AdaptiveAvgPool3D(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int64_t>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MaxPool1DSchema_TtTI32lI32lI32lI32lBBS {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& kernel_size, const Optional<std::vector<int32_t>>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& dilation, bool return_indices, bool ceil_mode, const std::string& data_format);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::MaxPool1D;
  static constexpr size_t max_args = 8;
  static constexpr size_t max_pos_args = 8;
  static constexpr char const* signature = "TensorTuple (Tensor input, Int32List kernel_size, Int32List stride=None, Int32List padding=0, Int32List dilation=1, Bool return_indices=True, Bool ceil_mode=False, String data_format=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t MaxPool1DSchema_TtTI32lI32lI32lI32lBBS::max_args;
constexpr size_t MaxPool1DSchema_TtTI32lI32lI32lI32lBBS::max_pos_args;
constexpr char const* MaxPool1DSchema_TtTI32lI32lI32lI32lBBS::signature;
FunctionDef MaxPool1DSchema_TtTI32lI32lI32lI32lBBS::function_def = {
/*name*/"max_pool1d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"kernel_size", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"stride", /*default_value*/Optional<std::vector<int32_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"padding", /*default_value*/std::vector<int32_t>({0}), /*size*/1, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dilation", /*default_value*/std::vector<int32_t>({1}), /*size*/1, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"return_indices", /*default_value*/bool(true), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"ceil_mode", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"data_format", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* max_pool1d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("max_pool1d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MaxPool1DSchema_TtTI32lI32lI32lI32lBBS> parser("max_pool1d");
  ParsedArgs<8> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::MaxPool1D(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<Optional<std::vector<int32_t>>>(), r[3].As<std::vector<int32_t>>(), r[4].As<std::vector<int32_t>>(), r[5].As<bool>(), r[6].As<bool>(), r[7].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MaxPool2DSchema_TtTI32lI32lI32lI32lBBS {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& kernel_size, const Optional<std::vector<int32_t>>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& dilation, bool return_indices, bool ceil_mode, const std::string& data_format);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::MaxPool2D;
  static constexpr size_t max_args = 8;
  static constexpr size_t max_pos_args = 8;
  static constexpr char const* signature = "TensorTuple (Tensor input, Int32List kernel_size, Int32List stride=None, Int32List padding=0, Int32List dilation=1, Bool return_indices=True, Bool ceil_mode=False, String data_format=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t MaxPool2DSchema_TtTI32lI32lI32lI32lBBS::max_args;
constexpr size_t MaxPool2DSchema_TtTI32lI32lI32lI32lBBS::max_pos_args;
constexpr char const* MaxPool2DSchema_TtTI32lI32lI32lI32lBBS::signature;
FunctionDef MaxPool2DSchema_TtTI32lI32lI32lI32lBBS::function_def = {
/*name*/"max_pool2d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"kernel_size", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/2, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"stride", /*default_value*/Optional<std::vector<int32_t>>(), /*size*/2, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"padding", /*default_value*/std::vector<int32_t>({0, 0}), /*size*/2, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dilation", /*default_value*/std::vector<int32_t>({1, 1}), /*size*/2, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"return_indices", /*default_value*/bool(true), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"ceil_mode", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"data_format", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* max_pool2d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("max_pool2d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MaxPool2DSchema_TtTI32lI32lI32lI32lBBS> parser("max_pool2d");
  ParsedArgs<8> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::MaxPool2D(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<Optional<std::vector<int32_t>>>(), r[3].As<std::vector<int32_t>>(), r[4].As<std::vector<int32_t>>(), r[5].As<bool>(), r[6].As<bool>(), r[7].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MaxPool3DSchema_TtTI32lI32lI32lI32lBBS {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& kernel_size, const Optional<std::vector<int32_t>>& stride, const std::vector<int32_t>& padding, const std::vector<int32_t>& dilation, bool return_indices, bool ceil_mode, const std::string& data_format);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::MaxPool3D;
  static constexpr size_t max_args = 8;
  static constexpr size_t max_pos_args = 8;
  static constexpr char const* signature = "TensorTuple (Tensor input, Int32List kernel_size, Int32List stride=None, Int32List padding=0, Int32List dilation=1, Bool return_indices=True, Bool ceil_mode=False, String data_format=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t MaxPool3DSchema_TtTI32lI32lI32lI32lBBS::max_args;
constexpr size_t MaxPool3DSchema_TtTI32lI32lI32lI32lBBS::max_pos_args;
constexpr char const* MaxPool3DSchema_TtTI32lI32lI32lI32lBBS::signature;
FunctionDef MaxPool3DSchema_TtTI32lI32lI32lI32lBBS::function_def = {
/*name*/"max_pool3d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"kernel_size", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/3, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"stride", /*default_value*/Optional<std::vector<int32_t>>(), /*size*/3, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"padding", /*default_value*/std::vector<int32_t>({0, 0, 0}), /*size*/3, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dilation", /*default_value*/std::vector<int32_t>({1, 1, 1}), /*size*/3, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"return_indices", /*default_value*/bool(true), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"ceil_mode", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"data_format", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* max_pool3d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("max_pool3d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MaxPool3DSchema_TtTI32lI32lI32lI32lBBS> parser("max_pool3d");
  ParsedArgs<8> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::MaxPool3D(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<Optional<std::vector<int32_t>>>(), r[3].As<std::vector<int32_t>>(), r[4].As<std::vector<int32_t>>(), r[5].As<bool>(), r[6].As<bool>(), r[7].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct PReluSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& alpha);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::PRelu;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor alpha)";
  static FunctionDef function_def;
};

constexpr size_t PReluSchema_TTT::max_args;
constexpr size_t PReluSchema_TTT::max_pos_args;
constexpr char const* PReluSchema_TTT::signature;
FunctionDef PReluSchema_TTT::function_def = {
/*name*/"prelu",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"alpha", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* prelu(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("prelu");
  PythonFrameGuard pf;
  static PythonArgParser<functional::PReluSchema_TTT> parser("prelu");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::PRelu(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ReshapeSchema_TTSh {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const Shape& shape);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Reshape;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Shape shape)";
  static FunctionDef function_def;
};

constexpr size_t ReshapeSchema_TTSh::max_args;
constexpr size_t ReshapeSchema_TTSh::max_pos_args;
constexpr char const* ReshapeSchema_TTSh::signature;
FunctionDef ReshapeSchema_TTSh::function_def = {
/*name*/"reshape",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"shape", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* reshape(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("reshape");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ReshapeSchema_TTSh> parser("reshape");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Reshape(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Shape>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ViewSchema_TTSh {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const Shape& shape);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::View;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Shape shape)";
  static FunctionDef function_def;
};

constexpr size_t ViewSchema_TTSh::max_args;
constexpr size_t ViewSchema_TTSh::max_pos_args;
constexpr char const* ViewSchema_TTSh::signature;
FunctionDef ViewSchema_TTSh::function_def = {
/*name*/"view",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"shape", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* view(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("view");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ViewSchema_TTSh> parser("view");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::View(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Shape>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ToContiguousSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ToContiguous;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input)";
  static FunctionDef function_def;
};

constexpr size_t ToContiguousSchema_TT::max_args;
constexpr size_t ToContiguousSchema_TT::max_pos_args;
constexpr char const* ToContiguousSchema_TT::signature;
FunctionDef ToContiguousSchema_TT::function_def = {
/*name*/"contiguous",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* contiguous(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("contiguous");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ToContiguousSchema_TT> parser("contiguous");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ToContiguous(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct InplaceToContiguousSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::InplaceToContiguous;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input)";
  static FunctionDef function_def;
};

constexpr size_t InplaceToContiguousSchema_TT::max_args;
constexpr size_t InplaceToContiguousSchema_TT::max_pos_args;
constexpr char const* InplaceToContiguousSchema_TT::signature;
FunctionDef InplaceToContiguousSchema_TT::function_def = {
/*name*/"contiguous_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* contiguous_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("contiguous_");
  PythonFrameGuard pf;
  static PythonArgParser<functional::InplaceToContiguousSchema_TT> parser("contiguous_");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::InplaceToContiguous(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SliceView1dContiguousSchema_TTI64I64 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, int64_t start, int64_t end);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::SliceView1dContiguous;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Int64 start, Int64 end)";
  static FunctionDef function_def;
};

constexpr size_t SliceView1dContiguousSchema_TTI64I64::max_args;
constexpr size_t SliceView1dContiguousSchema_TTI64I64::max_pos_args;
constexpr char const* SliceView1dContiguousSchema_TTI64I64::signature;
FunctionDef SliceView1dContiguousSchema_TTI64I64::function_def = {
/*name*/"slice_view_1d_contiguous",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"start", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"end", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* slice_view_1d_contiguous(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("slice_view_1d_contiguous");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SliceView1dContiguousSchema_TTI64I64> parser("slice_view_1d_contiguous");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::SliceView1dContiguous(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>(), r[2].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct NarrowSchema_TTI64I64I64 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int64_t dim, int64_t start, int64_t length);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Narrow;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor input, Int64 dim, Int64 start, Int64 length)";
  static FunctionDef function_def;
};

constexpr size_t NarrowSchema_TTI64I64I64::max_args;
constexpr size_t NarrowSchema_TTI64I64I64::max_pos_args;
constexpr char const* NarrowSchema_TTI64I64I64::signature;
FunctionDef NarrowSchema_TTI64I64I64::function_def = {
/*name*/"narrow",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"start", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"length", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* narrow(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("narrow");
  PythonFrameGuard pf;
  static PythonArgParser<functional::NarrowSchema_TTI64I64I64> parser("narrow");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Narrow(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>(), r[2].As<int64_t>(), r[3].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SliceSchema_TTI64lI64lI64lB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& start, const std::vector<int64_t>& stop, const std::vector<int64_t>& step, const Optional<bool>& enable_view_slice);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Slice;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Tensor (Tensor x, Int64List start, Int64List stop, Int64List step, Bool enable_view_slice=None)";
  static FunctionDef function_def;
};

constexpr size_t SliceSchema_TTI64lI64lI64lB::max_args;
constexpr size_t SliceSchema_TTI64lI64lI64lB::max_pos_args;
constexpr char const* SliceSchema_TTI64lI64lI64lB::signature;
FunctionDef SliceSchema_TTI64lI64lI64lB::function_def = {
/*name*/"slice",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"start", /*value_type*/ValueTypeOf<std::vector<int64_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"stop", /*value_type*/ValueTypeOf<std::vector<int64_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"step", /*value_type*/ValueTypeOf<std::vector<int64_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"enable_view_slice", /*default_value*/Optional<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* slice(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("slice");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SliceSchema_TTI64lI64lI64lB> parser("slice");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Slice(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int64_t>>(), r[2].As<std::vector<int64_t>>(), r[3].As<std::vector<int64_t>>(), r[4].As<Optional<bool>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SliceUpdateSchema_TTTI64lI64lI64lB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& ref, const std::shared_ptr<one::Tensor>& value, const std::vector<int64_t>& start, const std::vector<int64_t>& stop, const std::vector<int64_t>& step, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::SliceUpdate;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 6;
  static constexpr char const* signature = "Tensor (Tensor ref, Tensor value, Int64List start, Int64List stop, Int64List step, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t SliceUpdateSchema_TTTI64lI64lI64lB::max_args;
constexpr size_t SliceUpdateSchema_TTTI64lI64lI64lB::max_pos_args;
constexpr char const* SliceUpdateSchema_TTTI64lI64lI64lB::signature;
FunctionDef SliceUpdateSchema_TTTI64lI64lI64lB::function_def = {
/*name*/"slice_update",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"ref", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"value", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"start", /*value_type*/ValueTypeOf<std::vector<int64_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"stop", /*value_type*/ValueTypeOf<std::vector<int64_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"step", /*value_type*/ValueTypeOf<std::vector<int64_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* slice_update(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("slice_update");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SliceUpdateSchema_TTTI64lI64lI64lB> parser("slice_update");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::SliceUpdate(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::vector<int64_t>>(), r[3].As<std::vector<int64_t>>(), r[4].As<std::vector<int64_t>>(), r[5].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct CopySchema_TTSI64B {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::string& device_type, int64_t device_id, bool pin_memory);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Copy;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor x, String device_type, Int64 device_id, Bool pin_memory=False)";
  static FunctionDef function_def;
};

constexpr size_t CopySchema_TTSI64B::max_args;
constexpr size_t CopySchema_TTSI64B::max_pos_args;
constexpr char const* CopySchema_TTSI64B::signature;
FunctionDef CopySchema_TTSI64B::function_def = {
/*name*/"copy",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"device_type", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"device_id", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"pin_memory", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* copy(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("copy");
  PythonFrameGuard pf;
  static PythonArgParser<functional::CopySchema_TTSI64B> parser("copy");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Copy(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::string>(), r[2].As<int64_t>(), r[3].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ToSchema_TTSDtB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const Optional<std::string>& device, const Optional<Symbol<DType>>& dtype, bool copy);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::To;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor x, String device=None, DataType dtype=None, Bool copy=False)";
  static FunctionDef function_def;
};

constexpr size_t ToSchema_TTSDtB::max_args;
constexpr size_t ToSchema_TTSDtB::max_pos_args;
constexpr char const* ToSchema_TTSDtB::signature;
FunctionDef ToSchema_TTSDtB::function_def = {
/*name*/"to",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"device", /*default_value*/Optional<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"copy", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ToSchema_TTDeDtB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const Optional<Symbol<Device>>& device, const Optional<Symbol<DType>>& dtype, bool copy);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::To;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor x, Device device=None, DataType dtype=None, Bool copy=False)";
  static FunctionDef function_def;
};

constexpr size_t ToSchema_TTDeDtB::max_args;
constexpr size_t ToSchema_TTDeDtB::max_pos_args;
constexpr char const* ToSchema_TTDeDtB::signature;
FunctionDef ToSchema_TTDeDtB::function_def = {
/*name*/"to",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"device", /*default_value*/Optional<Symbol<Device>>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"copy", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ToSchema_TTDtB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const Optional<Symbol<DType>>& dtype, bool copy);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::To;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, DataType dtype=None, Bool copy=False)";
  static FunctionDef function_def;
};

constexpr size_t ToSchema_TTDtB::max_args;
constexpr size_t ToSchema_TTDtB::max_pos_args;
constexpr char const* ToSchema_TTDtB::signature;
FunctionDef ToSchema_TTDtB::function_def = {
/*name*/"to",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"copy", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ToSchema_TTTB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& other, bool copy);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::To;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor other, Bool copy=False)";
  static FunctionDef function_def;
};

constexpr size_t ToSchema_TTTB::max_args;
constexpr size_t ToSchema_TTTB::max_pos_args;
constexpr char const* ToSchema_TTTB::signature;
FunctionDef ToSchema_TTTB::function_def = {
/*name*/"to",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"copy", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct ToSchema_TTS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const Optional<std::string>& device);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::To;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, String device=None)";
  static FunctionDef function_def;
};

constexpr size_t ToSchema_TTS::max_args;
constexpr size_t ToSchema_TTS::max_pos_args;
constexpr char const* ToSchema_TTS::signature;
FunctionDef ToSchema_TTS::function_def = {
/*name*/"to",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"device", /*default_value*/Optional<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* to(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("to");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ToSchema_TTSDtB, functional::ToSchema_TTDeDtB, functional::ToSchema_TTDtB, functional::ToSchema_TTTB, functional::ToSchema_TTS> parser("to");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::To(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<std::string>>(), r[2].As<Optional<Symbol<DType>>>(), r[3].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::To(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<Symbol<Device>>>(), r[2].As<Optional<Symbol<DType>>>(), r[3].As<bool>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::To(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<Symbol<DType>>>(), r[2].As<bool>()));
  }
  if (idx == 3) {
    return CastToPyObject(functional::To(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<bool>()));
  }
  if (idx == 4) {
    return CastToPyObject(functional::To(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<std::string>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct FlipSchema_TTI32l {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& dims);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Flip;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Int32List dims)";
  static FunctionDef function_def;
};

constexpr size_t FlipSchema_TTI32l::max_args;
constexpr size_t FlipSchema_TTI32l::max_pos_args;
constexpr char const* FlipSchema_TTI32l::signature;
FunctionDef FlipSchema_TTI32l::function_def = {
/*name*/"flip",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dims", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* flip(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("flip");
  PythonFrameGuard pf;
  static PythonArgParser<functional::FlipSchema_TTI32l> parser("flip");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Flip(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct UpsampleSchema_TTDDBSS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, double height_scale, double width_scale, bool align_corners, const std::string& interpolation, const std::string& data_format);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Upsample;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 6;
  static constexpr char const* signature = "Tensor (Tensor x, Double height_scale, Double width_scale, Bool align_corners, String interpolation, String data_format=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t UpsampleSchema_TTDDBSS::max_args;
constexpr size_t UpsampleSchema_TTDDBSS::max_pos_args;
constexpr char const* UpsampleSchema_TTDDBSS::signature;
FunctionDef UpsampleSchema_TTDDBSS::function_def = {
/*name*/"upsample",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"height_scale", /*value_type*/ValueTypeOf<double>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"width_scale", /*value_type*/ValueTypeOf<double>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"align_corners", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"interpolation", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"data_format", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* upsample(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("upsample");
  PythonFrameGuard pf;
  static PythonArgParser<functional::UpsampleSchema_TTDDBSS> parser("upsample");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Upsample(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<double>(), r[2].As<double>(), r[3].As<bool>(), r[4].As<std::string>(), r[5].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct UpsampleLinear1DSchema_TTDBI64lS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, double scale_factor, bool align_corners, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::UpsampleLinear1D;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Tensor (Tensor x, Double scale_factor=0.0, Bool align_corners=False, Int64List output_size=None, String data_format=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t UpsampleLinear1DSchema_TTDBI64lS::max_args;
constexpr size_t UpsampleLinear1DSchema_TTDBI64lS::max_pos_args;
constexpr char const* UpsampleLinear1DSchema_TTDBI64lS::signature;
FunctionDef UpsampleLinear1DSchema_TTDBI64lS::function_def = {
/*name*/"upsample_linear_1d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"scale_factor", /*default_value*/double(0.0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"align_corners", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"output_size", /*default_value*/Optional<std::vector<int64_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"data_format", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* upsample_linear_1d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("upsample_linear_1d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::UpsampleLinear1DSchema_TTDBI64lS> parser("upsample_linear_1d");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::UpsampleLinear1D(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<double>(), r[2].As<bool>(), r[3].As<Optional<std::vector<int64_t>>>(), r[4].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct UpsampleNearest1DSchema_TTDI64lS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, double scale_factor, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::UpsampleNearest1D;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor x, Double scale_factor=0.0, Int64List output_size=None, String data_format=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t UpsampleNearest1DSchema_TTDI64lS::max_args;
constexpr size_t UpsampleNearest1DSchema_TTDI64lS::max_pos_args;
constexpr char const* UpsampleNearest1DSchema_TTDI64lS::signature;
FunctionDef UpsampleNearest1DSchema_TTDI64lS::function_def = {
/*name*/"upsample_nearest_1d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"scale_factor", /*default_value*/double(0.0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"output_size", /*default_value*/Optional<std::vector<int64_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"data_format", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* upsample_nearest_1d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("upsample_nearest_1d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::UpsampleNearest1DSchema_TTDI64lS> parser("upsample_nearest_1d");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::UpsampleNearest1D(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<double>(), r[2].As<Optional<std::vector<int64_t>>>(), r[3].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct UpsampleNearest2DSchema_TTDDI64lS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, double height_scale, double width_scale, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::UpsampleNearest2D;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Tensor (Tensor x, Double height_scale=0.0, Double width_scale=0.0, Int64List output_size=None, String data_format=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t UpsampleNearest2DSchema_TTDDI64lS::max_args;
constexpr size_t UpsampleNearest2DSchema_TTDDI64lS::max_pos_args;
constexpr char const* UpsampleNearest2DSchema_TTDDI64lS::signature;
FunctionDef UpsampleNearest2DSchema_TTDDI64lS::function_def = {
/*name*/"upsample_nearest_2d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"height_scale", /*default_value*/double(0.0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"width_scale", /*default_value*/double(0.0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"output_size", /*default_value*/Optional<std::vector<int64_t>>(), /*size*/2, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"data_format", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* upsample_nearest_2d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("upsample_nearest_2d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::UpsampleNearest2DSchema_TTDDI64lS> parser("upsample_nearest_2d");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::UpsampleNearest2D(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<double>(), r[2].As<double>(), r[3].As<Optional<std::vector<int64_t>>>(), r[4].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct UpsampleBilinear2DSchema_TTDDBI64lS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, double height_scale, double width_scale, bool align_corners, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::UpsampleBilinear2D;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 6;
  static constexpr char const* signature = "Tensor (Tensor x, Double height_scale=0.0, Double width_scale=0.0, Bool align_corners=False, Int64List output_size=None, String data_format=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t UpsampleBilinear2DSchema_TTDDBI64lS::max_args;
constexpr size_t UpsampleBilinear2DSchema_TTDDBI64lS::max_pos_args;
constexpr char const* UpsampleBilinear2DSchema_TTDDBI64lS::signature;
FunctionDef UpsampleBilinear2DSchema_TTDDBI64lS::function_def = {
/*name*/"upsample_bilinear_2d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"height_scale", /*default_value*/double(0.0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"width_scale", /*default_value*/double(0.0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"align_corners", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"output_size", /*default_value*/Optional<std::vector<int64_t>>(), /*size*/2, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"data_format", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* upsample_bilinear_2d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("upsample_bilinear_2d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::UpsampleBilinear2DSchema_TTDDBI64lS> parser("upsample_bilinear_2d");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::UpsampleBilinear2D(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<double>(), r[2].As<double>(), r[3].As<bool>(), r[4].As<Optional<std::vector<int64_t>>>(), r[5].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct UpsampleBicubic2DSchema_TTDDBI64lS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, double height_scale, double width_scale, bool align_corners, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::UpsampleBicubic2D;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 6;
  static constexpr char const* signature = "Tensor (Tensor x, Double height_scale=0.0, Double width_scale=0.0, Bool align_corners=False, Int64List output_size=None, String data_format=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t UpsampleBicubic2DSchema_TTDDBI64lS::max_args;
constexpr size_t UpsampleBicubic2DSchema_TTDDBI64lS::max_pos_args;
constexpr char const* UpsampleBicubic2DSchema_TTDDBI64lS::signature;
FunctionDef UpsampleBicubic2DSchema_TTDDBI64lS::function_def = {
/*name*/"upsample_bicubic_2d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"height_scale", /*default_value*/double(0.0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"width_scale", /*default_value*/double(0.0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"align_corners", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"output_size", /*default_value*/Optional<std::vector<int64_t>>(), /*size*/2, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"data_format", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* upsample_bicubic_2d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("upsample_bicubic_2d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::UpsampleBicubic2DSchema_TTDDBI64lS> parser("upsample_bicubic_2d");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::UpsampleBicubic2D(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<double>(), r[2].As<double>(), r[3].As<bool>(), r[4].As<Optional<std::vector<int64_t>>>(), r[5].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct UpsampleNearest3DSchema_TTDDDI64lS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, double depth_scale, double height_scale, double width_scale, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::UpsampleNearest3D;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 6;
  static constexpr char const* signature = "Tensor (Tensor x, Double depth_scale=0.0, Double height_scale=0.0, Double width_scale=0.0, Int64List output_size=None, String data_format=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t UpsampleNearest3DSchema_TTDDDI64lS::max_args;
constexpr size_t UpsampleNearest3DSchema_TTDDDI64lS::max_pos_args;
constexpr char const* UpsampleNearest3DSchema_TTDDDI64lS::signature;
FunctionDef UpsampleNearest3DSchema_TTDDDI64lS::function_def = {
/*name*/"upsample_nearest_3d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"depth_scale", /*default_value*/double(0.0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"height_scale", /*default_value*/double(0.0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"width_scale", /*default_value*/double(0.0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"output_size", /*default_value*/Optional<std::vector<int64_t>>(), /*size*/3, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"data_format", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* upsample_nearest_3d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("upsample_nearest_3d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::UpsampleNearest3DSchema_TTDDDI64lS> parser("upsample_nearest_3d");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::UpsampleNearest3D(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<double>(), r[2].As<double>(), r[3].As<double>(), r[4].As<Optional<std::vector<int64_t>>>(), r[5].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct UpsampleTrilinear3DSchema_TTDDDBI64lS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, double depth_scale, double height_scale, double width_scale, bool align_corners, const Optional<std::vector<int64_t>>& output_size, const std::string& data_format);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::UpsampleTrilinear3D;
  static constexpr size_t max_args = 7;
  static constexpr size_t max_pos_args = 7;
  static constexpr char const* signature = "Tensor (Tensor x, Double depth_scale=0.0, Double height_scale=0.0, Double width_scale=0.0, Bool align_corners=False, Int64List output_size=None, String data_format=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t UpsampleTrilinear3DSchema_TTDDDBI64lS::max_args;
constexpr size_t UpsampleTrilinear3DSchema_TTDDDBI64lS::max_pos_args;
constexpr char const* UpsampleTrilinear3DSchema_TTDDDBI64lS::signature;
FunctionDef UpsampleTrilinear3DSchema_TTDDDBI64lS::function_def = {
/*name*/"upsample_trilinear_3d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"depth_scale", /*default_value*/double(0.0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"height_scale", /*default_value*/double(0.0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"width_scale", /*default_value*/double(0.0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"align_corners", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"output_size", /*default_value*/Optional<std::vector<int64_t>>(), /*size*/3, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"data_format", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* upsample_trilinear_3d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("upsample_trilinear_3d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::UpsampleTrilinear3DSchema_TTDDDBI64lS> parser("upsample_trilinear_3d");
  ParsedArgs<7> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::UpsampleTrilinear3D(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<double>(), r[2].As<double>(), r[3].As<double>(), r[4].As<bool>(), r[5].As<Optional<std::vector<int64_t>>>(), r[6].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AbsSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Abs;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t AbsSchema_TT::max_args;
constexpr size_t AbsSchema_TT::max_pos_args;
constexpr char const* AbsSchema_TT::signature;
FunctionDef AbsSchema_TT::function_def = {
/*name*/"abs",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* abs(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("abs");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AbsSchema_TT> parser("abs");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Abs(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AcosSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Acos;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t AcosSchema_TT::max_args;
constexpr size_t AcosSchema_TT::max_pos_args;
constexpr char const* AcosSchema_TT::signature;
FunctionDef AcosSchema_TT::function_def = {
/*name*/"acos",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* acos(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("acos");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AcosSchema_TT> parser("acos");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Acos(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AcoshSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Acosh;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t AcoshSchema_TT::max_args;
constexpr size_t AcoshSchema_TT::max_pos_args;
constexpr char const* AcoshSchema_TT::signature;
FunctionDef AcoshSchema_TT::function_def = {
/*name*/"acosh",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* acosh(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("acosh");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AcoshSchema_TT> parser("acosh");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Acosh(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AsinSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Asin;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t AsinSchema_TT::max_args;
constexpr size_t AsinSchema_TT::max_pos_args;
constexpr char const* AsinSchema_TT::signature;
FunctionDef AsinSchema_TT::function_def = {
/*name*/"asin",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* asin(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("asin");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AsinSchema_TT> parser("asin");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Asin(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AsinhSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Asinh;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t AsinhSchema_TT::max_args;
constexpr size_t AsinhSchema_TT::max_pos_args;
constexpr char const* AsinhSchema_TT::signature;
FunctionDef AsinhSchema_TT::function_def = {
/*name*/"asinh",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* asinh(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("asinh");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AsinhSchema_TT> parser("asinh");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Asinh(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AtanSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Atan;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t AtanSchema_TT::max_args;
constexpr size_t AtanSchema_TT::max_pos_args;
constexpr char const* AtanSchema_TT::signature;
FunctionDef AtanSchema_TT::function_def = {
/*name*/"atan",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* atan(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("atan");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AtanSchema_TT> parser("atan");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Atan(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct Atan2Schema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Atan2;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t Atan2Schema_TTT::max_args;
constexpr size_t Atan2Schema_TTT::max_pos_args;
constexpr char const* Atan2Schema_TTT::signature;
FunctionDef Atan2Schema_TTT::function_def = {
/*name*/"atan2",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* atan2(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("atan2");
  PythonFrameGuard pf;
  static PythonArgParser<functional::Atan2Schema_TTT> parser("atan2");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Atan2(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AtanhSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Atanh;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t AtanhSchema_TT::max_args;
constexpr size_t AtanhSchema_TT::max_pos_args;
constexpr char const* AtanhSchema_TT::signature;
FunctionDef AtanhSchema_TT::function_def = {
/*name*/"atanh",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* atanh(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("atanh");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AtanhSchema_TT> parser("atanh");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Atanh(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct CeilSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Ceil;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t CeilSchema_TT::max_args;
constexpr size_t CeilSchema_TT::max_pos_args;
constexpr char const* CeilSchema_TT::signature;
FunctionDef CeilSchema_TT::function_def = {
/*name*/"ceil",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* ceil(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("ceil");
  PythonFrameGuard pf;
  static PythonArgParser<functional::CeilSchema_TT> parser("ceil");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Ceil(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ErfSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Erf;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t ErfSchema_TT::max_args;
constexpr size_t ErfSchema_TT::max_pos_args;
constexpr char const* ErfSchema_TT::signature;
FunctionDef ErfSchema_TT::function_def = {
/*name*/"erf",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* erf(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("erf");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ErfSchema_TT> parser("erf");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Erf(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ErfcSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Erfc;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t ErfcSchema_TT::max_args;
constexpr size_t ErfcSchema_TT::max_pos_args;
constexpr char const* ErfcSchema_TT::signature;
FunctionDef ErfcSchema_TT::function_def = {
/*name*/"erfc",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* erfc(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("erfc");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ErfcSchema_TT> parser("erfc");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Erfc(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct Expm1Schema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Expm1;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t Expm1Schema_TT::max_args;
constexpr size_t Expm1Schema_TT::max_pos_args;
constexpr char const* Expm1Schema_TT::signature;
FunctionDef Expm1Schema_TT::function_def = {
/*name*/"expm1",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* expm1(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("expm1");
  PythonFrameGuard pf;
  static PythonArgParser<functional::Expm1Schema_TT> parser("expm1");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Expm1(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct FloorSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Floor;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t FloorSchema_TT::max_args;
constexpr size_t FloorSchema_TT::max_pos_args;
constexpr char const* FloorSchema_TT::signature;
FunctionDef FloorSchema_TT::function_def = {
/*name*/"floor",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* floor(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("floor");
  PythonFrameGuard pf;
  static PythonArgParser<functional::FloorSchema_TT> parser("floor");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Floor(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct Floor_Schema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Floor_;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t Floor_Schema_TT::max_args;
constexpr size_t Floor_Schema_TT::max_pos_args;
constexpr char const* Floor_Schema_TT::signature;
FunctionDef Floor_Schema_TT::function_def = {
/*name*/"floor_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* floor_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("floor_");
  PythonFrameGuard pf;
  static PythonArgParser<functional::Floor_Schema_TT> parser("floor_");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Floor_(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct LgammaSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Lgamma;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t LgammaSchema_TT::max_args;
constexpr size_t LgammaSchema_TT::max_pos_args;
constexpr char const* LgammaSchema_TT::signature;
FunctionDef LgammaSchema_TT::function_def = {
/*name*/"lgamma",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* lgamma(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("lgamma");
  PythonFrameGuard pf;
  static PythonArgParser<functional::LgammaSchema_TT> parser("lgamma");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Lgamma(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct Log1pSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Log1p;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t Log1pSchema_TT::max_args;
constexpr size_t Log1pSchema_TT::max_pos_args;
constexpr char const* Log1pSchema_TT::signature;
FunctionDef Log1pSchema_TT::function_def = {
/*name*/"log1p",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* log1p(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("log1p");
  PythonFrameGuard pf;
  static PythonArgParser<functional::Log1pSchema_TT> parser("log1p");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Log1p(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct LogSigmoidSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::LogSigmoid;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t LogSigmoidSchema_TT::max_args;
constexpr size_t LogSigmoidSchema_TT::max_pos_args;
constexpr char const* LogSigmoidSchema_TT::signature;
FunctionDef LogSigmoidSchema_TT::function_def = {
/*name*/"logsigmoid",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* logsigmoid(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("logsigmoid");
  PythonFrameGuard pf;
  static PythonArgParser<functional::LogSigmoidSchema_TT> parser("logsigmoid");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::LogSigmoid(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RintSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Rint;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t RintSchema_TT::max_args;
constexpr size_t RintSchema_TT::max_pos_args;
constexpr char const* RintSchema_TT::signature;
FunctionDef RintSchema_TT::function_def = {
/*name*/"rint",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* rint(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("rint");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RintSchema_TT> parser("rint");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Rint(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RoundSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Round;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t RoundSchema_TT::max_args;
constexpr size_t RoundSchema_TT::max_pos_args;
constexpr char const* RoundSchema_TT::signature;
FunctionDef RoundSchema_TT::function_def = {
/*name*/"round",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* round(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("round");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RoundSchema_TT> parser("round");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Round(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SignSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Sign;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t SignSchema_TT::max_args;
constexpr size_t SignSchema_TT::max_pos_args;
constexpr char const* SignSchema_TT::signature;
FunctionDef SignSchema_TT::function_def = {
/*name*/"sign",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* sign(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("sign");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SignSchema_TT> parser("sign");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Sign(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SinhSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Sinh;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t SinhSchema_TT::max_args;
constexpr size_t SinhSchema_TT::max_pos_args;
constexpr char const* SinhSchema_TT::signature;
FunctionDef SinhSchema_TT::function_def = {
/*name*/"sinh",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* sinh(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("sinh");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SinhSchema_TT> parser("sinh");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Sinh(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SoftplusSchema_TTDD {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, double beta, double threshold);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Softplus;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Double beta=1.0, Double threshold=20.0)";
  static FunctionDef function_def;
};

constexpr size_t SoftplusSchema_TTDD::max_args;
constexpr size_t SoftplusSchema_TTDD::max_pos_args;
constexpr char const* SoftplusSchema_TTDD::signature;
FunctionDef SoftplusSchema_TTDD::function_def = {
/*name*/"softplus",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"beta", /*default_value*/double(1.0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"threshold", /*default_value*/double(20.0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* softplus(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("softplus");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SoftplusSchema_TTDD> parser("softplus");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Softplus(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<double>(), r[2].As<double>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SoftShrinkSchema_TTDB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, double alpha, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::SoftShrink;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x, *, Double alpha=0.5, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t SoftShrinkSchema_TTDB::max_args;
constexpr size_t SoftShrinkSchema_TTDB::max_pos_args;
constexpr char const* SoftShrinkSchema_TTDB::signature;
FunctionDef SoftShrinkSchema_TTDB::function_def = {
/*name*/"softshrink",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"alpha", /*default_value*/double(0.5), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* softshrink(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("softshrink");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SoftShrinkSchema_TTDB> parser("softshrink");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::SoftShrink(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<double>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct OneHotSchema_TTI64ScSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int64_t num_classes, const Scalar& on_value, const Scalar& off_value);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::OneHot;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor input, Int64 num_classes=-1, Scalar on_value=1, Scalar off_value=0)";
  static FunctionDef function_def;
};

constexpr size_t OneHotSchema_TTI64ScSc::max_args;
constexpr size_t OneHotSchema_TTI64ScSc::max_pos_args;
constexpr char const* OneHotSchema_TTI64ScSc::signature;
FunctionDef OneHotSchema_TTI64ScSc::function_def = {
/*name*/"one_hot",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"num_classes", /*default_value*/int64_t(-1), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"on_value", /*default_value*/Scalar(1), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"off_value", /*default_value*/Scalar(0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* one_hot(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("one_hot");
  PythonFrameGuard pf;
  static PythonArgParser<functional::OneHotSchema_TTI64ScSc> parser("one_hot");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::OneHot(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>(), r[2].As<Scalar>(), r[3].As<Scalar>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct UnsortedSegmentSumSchema_TTTI64I64 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& segment_ids, int64_t axis, int64_t num_segments);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::UnsortedSegmentSum;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor segment_ids, Int64 axis, Int64 num_segments)";
  static FunctionDef function_def;
};

constexpr size_t UnsortedSegmentSumSchema_TTTI64I64::max_args;
constexpr size_t UnsortedSegmentSumSchema_TTTI64I64::max_pos_args;
constexpr char const* UnsortedSegmentSumSchema_TTTI64I64::signature;
FunctionDef UnsortedSegmentSumSchema_TTTI64I64::function_def = {
/*name*/"unsorted_segment_sum",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"segment_ids", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"axis", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"num_segments", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* unsorted_segment_sum(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("unsorted_segment_sum");
  PythonFrameGuard pf;
  static PythonArgParser<functional::UnsortedSegmentSumSchema_TTTI64I64> parser("unsorted_segment_sum");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::UnsortedSegmentSum(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<int64_t>(), r[3].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TrilSchema_TTI64 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, int64_t diagonal);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Tril;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Int64 diagonal=0)";
  static FunctionDef function_def;
};

constexpr size_t TrilSchema_TTI64::max_args;
constexpr size_t TrilSchema_TTI64::max_pos_args;
constexpr char const* TrilSchema_TTI64::signature;
FunctionDef TrilSchema_TTI64::function_def = {
/*name*/"tril",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"diagonal", /*default_value*/int64_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* tril(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("tril");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TrilSchema_TTI64> parser("tril");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Tril(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TriuSchema_TTI64 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, int64_t diagonal);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Triu;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Int64 diagonal=0)";
  static FunctionDef function_def;
};

constexpr size_t TriuSchema_TTI64::max_args;
constexpr size_t TriuSchema_TTI64::max_pos_args;
constexpr char const* TriuSchema_TTI64::signature;
FunctionDef TriuSchema_TTI64::function_def = {
/*name*/"triu",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"diagonal", /*default_value*/int64_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* triu(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("triu");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TriuSchema_TTI64> parser("triu");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Triu(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct InplaceTriuSchema_TTI64 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, int64_t diagonal);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::InplaceTriu;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Int64 diagonal=0)";
  static FunctionDef function_def;
};

constexpr size_t InplaceTriuSchema_TTI64::max_args;
constexpr size_t InplaceTriuSchema_TTI64::max_pos_args;
constexpr char const* InplaceTriuSchema_TTI64::signature;
FunctionDef InplaceTriuSchema_TTI64::function_def = {
/*name*/"triu_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"diagonal", /*default_value*/int64_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* triu_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("triu_");
  PythonFrameGuard pf;
  static PythonArgParser<functional::InplaceTriuSchema_TTI64> parser("triu_");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::InplaceTriu(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ClampSchema_TTScSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Optional<Scalar>& min, const Optional<Scalar>& max);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Clamp;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar min=None, Scalar max=None)";
  static FunctionDef function_def;
};

constexpr size_t ClampSchema_TTScSc::max_args;
constexpr size_t ClampSchema_TTScSc::max_pos_args;
constexpr char const* ClampSchema_TTScSc::signature;
FunctionDef ClampSchema_TTScSc::function_def = {
/*name*/"clamp",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"min", /*default_value*/Optional<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"max", /*default_value*/Optional<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* clamp(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("clamp");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ClampSchema_TTScSc> parser("clamp");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Clamp(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<Scalar>>(), r[2].As<Optional<Scalar>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ClampInplaceSchema_TTScSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Optional<Scalar>& min, const Optional<Scalar>& max);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ClampInplace;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar min=None, Scalar max=None)";
  static FunctionDef function_def;
};

constexpr size_t ClampInplaceSchema_TTScSc::max_args;
constexpr size_t ClampInplaceSchema_TTScSc::max_pos_args;
constexpr char const* ClampInplaceSchema_TTScSc::signature;
FunctionDef ClampInplaceSchema_TTScSc::function_def = {
/*name*/"clamp_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"min", /*default_value*/Optional<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"max", /*default_value*/Optional<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* clamp_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("clamp_");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ClampInplaceSchema_TTScSc> parser("clamp_");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ClampInplace(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<Scalar>>(), r[2].As<Optional<Scalar>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ClampMinSchema_TTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& min);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ClampMin;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar min)";
  static FunctionDef function_def;
};

constexpr size_t ClampMinSchema_TTSc::max_args;
constexpr size_t ClampMinSchema_TTSc::max_pos_args;
constexpr char const* ClampMinSchema_TTSc::signature;
FunctionDef ClampMinSchema_TTSc::function_def = {
/*name*/"clamp_min",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"min", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* clamp_min(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("clamp_min");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ClampMinSchema_TTSc> parser("clamp_min");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ClampMin(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ClampMinInplaceSchema_TTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& min);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ClampMinInplace;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar min)";
  static FunctionDef function_def;
};

constexpr size_t ClampMinInplaceSchema_TTSc::max_args;
constexpr size_t ClampMinInplaceSchema_TTSc::max_pos_args;
constexpr char const* ClampMinInplaceSchema_TTSc::signature;
FunctionDef ClampMinInplaceSchema_TTSc::function_def = {
/*name*/"clamp_min_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"min", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* clamp_min_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("clamp_min_");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ClampMinInplaceSchema_TTSc> parser("clamp_min_");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ClampMinInplace(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ClampMaxSchema_TTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& max);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ClampMax;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar max)";
  static FunctionDef function_def;
};

constexpr size_t ClampMaxSchema_TTSc::max_args;
constexpr size_t ClampMaxSchema_TTSc::max_pos_args;
constexpr char const* ClampMaxSchema_TTSc::signature;
FunctionDef ClampMaxSchema_TTSc::function_def = {
/*name*/"clamp_max",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"max", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* clamp_max(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("clamp_max");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ClampMaxSchema_TTSc> parser("clamp_max");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ClampMax(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ClampMaxInplaceSchema_TTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& min);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ClampMaxInplace;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar min)";
  static FunctionDef function_def;
};

constexpr size_t ClampMaxInplaceSchema_TTSc::max_args;
constexpr size_t ClampMaxInplaceSchema_TTSc::max_pos_args;
constexpr char const* ClampMaxInplaceSchema_TTSc::signature;
FunctionDef ClampMaxInplaceSchema_TTSc::function_def = {
/*name*/"clamp_max_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"min", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* clamp_max_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("clamp_max_");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ClampMaxInplaceSchema_TTSc> parser("clamp_max_");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ClampMaxInplace(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ClipSchema_TTScSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Optional<Scalar>& min, const Optional<Scalar>& max);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Clip;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar min=None, Scalar max=None)";
  static FunctionDef function_def;
};

constexpr size_t ClipSchema_TTScSc::max_args;
constexpr size_t ClipSchema_TTScSc::max_pos_args;
constexpr char const* ClipSchema_TTScSc::signature;
FunctionDef ClipSchema_TTScSc::function_def = {
/*name*/"clip",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"min", /*default_value*/Optional<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"max", /*default_value*/Optional<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* clip(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("clip");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ClipSchema_TTScSc> parser("clip");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Clip(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<Scalar>>(), r[2].As<Optional<Scalar>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ClipInplaceSchema_TTScSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Optional<Scalar>& min, const Optional<Scalar>& max);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ClipInplace;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar min=None, Scalar max=None)";
  static FunctionDef function_def;
};

constexpr size_t ClipInplaceSchema_TTScSc::max_args;
constexpr size_t ClipInplaceSchema_TTScSc::max_pos_args;
constexpr char const* ClipInplaceSchema_TTScSc::signature;
FunctionDef ClipInplaceSchema_TTScSc::function_def = {
/*name*/"clip_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"min", /*default_value*/Optional<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"max", /*default_value*/Optional<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* clip_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("clip_");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ClipInplaceSchema_TTScSc> parser("clip_");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ClipInplace(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<Scalar>>(), r[2].As<Optional<Scalar>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct VectorNormSchema_TTScI32lBDt {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& ord, const Optional<std::vector<int32_t>>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::VectorNorm;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar ord=2, Int32List dim=None, Bool keepdim=False, *, DataType dtype=None)";
  static FunctionDef function_def;
};

constexpr size_t VectorNormSchema_TTScI32lBDt::max_args;
constexpr size_t VectorNormSchema_TTScI32lBDt::max_pos_args;
constexpr char const* VectorNormSchema_TTScI32lBDt::signature;
FunctionDef VectorNormSchema_TTScI32lBDt::function_def = {
/*name*/"vector_norm",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"ord", /*default_value*/Scalar(2), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/Optional<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

struct VectorNormSchema_TTScScBDt {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& ord, const Scalar& dim, bool keepdim, const Optional<Symbol<DType>>& dtype);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::VectorNorm;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar ord=2, Scalar dim, Bool keepdim=False, *, DataType dtype=None)";
  static FunctionDef function_def;
};

constexpr size_t VectorNormSchema_TTScScBDt::max_args;
constexpr size_t VectorNormSchema_TTScScBDt::max_pos_args;
constexpr char const* VectorNormSchema_TTScScBDt::signature;
FunctionDef VectorNormSchema_TTScScBDt::function_def = {
/*name*/"vector_norm",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"ord", /*default_value*/Scalar(2), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

PyObject* vector_norm(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("vector_norm");
  PythonFrameGuard pf;
  static PythonArgParser<functional::VectorNormSchema_TTScI32lBDt, functional::VectorNormSchema_TTScScBDt> parser("vector_norm");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::VectorNorm(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>(), r[2].As<Optional<std::vector<int32_t>>>(), r[3].As<bool>(), r[4].As<Optional<Symbol<DType>>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::VectorNorm(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>(), r[2].As<Scalar>(), r[3].As<bool>(), r[4].As<Optional<Symbol<DType>>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MatrixNormSchema_TTScI32lBDt {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Scalar& ord, const std::vector<int32_t>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::MatrixNorm;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar ord, Int32List dim, Bool keepdim=False, *, DataType dtype=None)";
  static FunctionDef function_def;
};

constexpr size_t MatrixNormSchema_TTScI32lBDt::max_args;
constexpr size_t MatrixNormSchema_TTScI32lBDt::max_pos_args;
constexpr char const* MatrixNormSchema_TTScI32lBDt::signature;
FunctionDef MatrixNormSchema_TTScI32lBDt::function_def = {
/*name*/"matrix_norm",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"ord", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

struct MatrixNormSchema_TTSI32lBDt {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::string& ord, const std::vector<int32_t>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::MatrixNorm;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor input, String ord, Int32List dim, Bool keepdim=False, *, DataType dtype=None)";
  static FunctionDef function_def;
};

constexpr size_t MatrixNormSchema_TTSI32lBDt::max_args;
constexpr size_t MatrixNormSchema_TTSI32lBDt::max_pos_args;
constexpr char const* MatrixNormSchema_TTSI32lBDt::signature;
FunctionDef MatrixNormSchema_TTSI32lBDt::function_def = {
/*name*/"matrix_norm",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"ord", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

PyObject* matrix_norm(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("matrix_norm");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MatrixNormSchema_TTScI32lBDt, functional::MatrixNormSchema_TTSI32lBDt> parser("matrix_norm");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::MatrixNorm(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>(), r[2].As<std::vector<int32_t>>(), r[3].As<bool>(), r[4].As<Optional<Symbol<DType>>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::MatrixNorm(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::string>(), r[2].As<std::vector<int32_t>>(), r[3].As<bool>(), r[4].As<Optional<Symbol<DType>>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct NormSchema_TTScI32lBDtB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Optional<Scalar>& ord, const Optional<std::vector<int32_t>>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype, bool for_norm);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Norm;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar ord=None, Int32List dim=None, Bool keepdim=False, *, DataType dtype=None, Bool for_norm=False)";
  static FunctionDef function_def;
};

constexpr size_t NormSchema_TTScI32lBDtB::max_args;
constexpr size_t NormSchema_TTScI32lBDtB::max_pos_args;
constexpr char const* NormSchema_TTScI32lBDtB::signature;
FunctionDef NormSchema_TTScI32lBDtB::function_def = {
/*name*/"norm",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"ord", /*default_value*/Optional<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"dim", /*default_value*/Optional<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"for_norm", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct NormSchema_TTSI32lBDt {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::string& ord, const Optional<std::vector<int32_t>>& dim, bool keepdim, const Optional<Symbol<DType>>& dtype);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Norm;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor input, String ord, Int32List dim=None, Bool keepdim=False, *, DataType dtype=None)";
  static FunctionDef function_def;
};

constexpr size_t NormSchema_TTSI32lBDt::max_args;
constexpr size_t NormSchema_TTSI32lBDt::max_pos_args;
constexpr char const* NormSchema_TTSI32lBDt::signature;
FunctionDef NormSchema_TTSI32lBDt::function_def = {
/*name*/"norm",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"ord", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/Optional<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

struct ScalarNormSchema_TTScScBDt {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Optional<Scalar>& ord, const Scalar& dim, bool keepdim, const Optional<Symbol<DType>>& dtype);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarNorm;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor input, Scalar ord=None, Scalar dim, Bool keepdim=False, *, DataType dtype=None)";
  static FunctionDef function_def;
};

constexpr size_t ScalarNormSchema_TTScScBDt::max_args;
constexpr size_t ScalarNormSchema_TTScScBDt::max_pos_args;
constexpr char const* ScalarNormSchema_TTScScBDt::signature;
FunctionDef ScalarNormSchema_TTScScBDt::function_def = {
/*name*/"norm",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"ord", /*default_value*/Optional<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

struct ScalarNormSchema_TTSScBDt {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::string& ord, const Scalar& dim, bool keepdim, const Optional<Symbol<DType>>& dtype);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ScalarNorm;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor input, String ord, Scalar dim, Bool keepdim=False, *, DataType dtype=None)";
  static FunctionDef function_def;
};

constexpr size_t ScalarNormSchema_TTSScBDt::max_args;
constexpr size_t ScalarNormSchema_TTSScBDt::max_pos_args;
constexpr char const* ScalarNormSchema_TTSScBDt::signature;
FunctionDef ScalarNormSchema_TTSScBDt::function_def = {
/*name*/"norm",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"ord", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

PyObject* norm(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("norm");
  PythonFrameGuard pf;
  static PythonArgParser<functional::NormSchema_TTScI32lBDtB, functional::NormSchema_TTSI32lBDt, functional::ScalarNormSchema_TTScScBDt, functional::ScalarNormSchema_TTSScBDt> parser("norm");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Norm(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<Scalar>>(), r[2].As<Optional<std::vector<int32_t>>>(), r[3].As<bool>(), r[4].As<Optional<Symbol<DType>>>(), r[5].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::Norm(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::string>(), r[2].As<Optional<std::vector<int32_t>>>(), r[3].As<bool>(), r[4].As<Optional<Symbol<DType>>>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::ScalarNorm(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<Scalar>>(), r[2].As<Scalar>(), r[3].As<bool>(), r[4].As<Optional<Symbol<DType>>>()));
  }
  if (idx == 3) {
    return CastToPyObject(functional::ScalarNorm(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::string>(), r[2].As<Scalar>(), r[3].As<bool>(), r[4].As<Optional<Symbol<DType>>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct InvSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Inv;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t InvSchema_TT::max_args;
constexpr size_t InvSchema_TT::max_pos_args;
constexpr char const* InvSchema_TT::signature;
FunctionDef InvSchema_TT::function_def = {
/*name*/"inv",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* inv(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("inv");
  PythonFrameGuard pf;
  static PythonArgParser<functional::InvSchema_TT> parser("inv");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Inv(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct LinalgCrossSchema_TTTI64 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other, const Optional<int64_t>& dim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::LinalgCross;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other, Int64 dim=None)";
  static FunctionDef function_def;
};

constexpr size_t LinalgCrossSchema_TTTI64::max_args;
constexpr size_t LinalgCrossSchema_TTTI64::max_pos_args;
constexpr char const* LinalgCrossSchema_TTTI64::signature;
FunctionDef LinalgCrossSchema_TTTI64::function_def = {
/*name*/"linalg_cross",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/Optional<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* linalg_cross(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("linalg_cross");
  PythonFrameGuard pf;
  static PythonArgParser<functional::LinalgCrossSchema_TTTI64> parser("linalg_cross");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::LinalgCross(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Optional<int64_t>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct DropoutSchema_TTFBBGT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, float p, bool training, bool inplace, const Optional<one::Generator>& generator, const Optional<one::Tensor>& addend);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Dropout;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Tensor (Tensor input, Float p=0.5, Bool training=True, Bool inplace=False, Generator generator=None, *, Tensor addend=None)";
  static FunctionDef function_def;
};

constexpr size_t DropoutSchema_TTFBBGT::max_args;
constexpr size_t DropoutSchema_TTFBBGT::max_pos_args;
constexpr char const* DropoutSchema_TTFBBGT::signature;
FunctionDef DropoutSchema_TTFBBGT::function_def = {
/*name*/"dropout",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"p", /*default_value*/float(0.5), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"training", /*default_value*/bool(true), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"addend", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

PyObject* dropout(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("dropout");
  PythonFrameGuard pf;
  static PythonArgParser<functional::DropoutSchema_TTFBBGT> parser("dropout");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Dropout(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<float>(), r[2].As<bool>(), r[3].As<bool>(), r[4].As<Optional<one::Generator>>(), r[5].As<Optional<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct Dropout1dSchema_TTFB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, float p, bool training);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Dropout1d;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Float p=0.5, Bool training=True)";
  static FunctionDef function_def;
};

constexpr size_t Dropout1dSchema_TTFB::max_args;
constexpr size_t Dropout1dSchema_TTFB::max_pos_args;
constexpr char const* Dropout1dSchema_TTFB::signature;
FunctionDef Dropout1dSchema_TTFB::function_def = {
/*name*/"dropout1d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"p", /*default_value*/float(0.5), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"training", /*default_value*/bool(true), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* dropout1d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("dropout1d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::Dropout1dSchema_TTFB> parser("dropout1d");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Dropout1d(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<float>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct Dropout2dSchema_TTFB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, float p, bool training);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Dropout2d;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Float p=0.5, Bool training=True)";
  static FunctionDef function_def;
};

constexpr size_t Dropout2dSchema_TTFB::max_args;
constexpr size_t Dropout2dSchema_TTFB::max_pos_args;
constexpr char const* Dropout2dSchema_TTFB::signature;
FunctionDef Dropout2dSchema_TTFB::function_def = {
/*name*/"dropout2d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"p", /*default_value*/float(0.5), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"training", /*default_value*/bool(true), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* dropout2d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("dropout2d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::Dropout2dSchema_TTFB> parser("dropout2d");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Dropout2d(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<float>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct Dropout3dSchema_TTFB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, float p, bool training);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Dropout3d;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Float p=0.5, Bool training=True)";
  static FunctionDef function_def;
};

constexpr size_t Dropout3dSchema_TTFB::max_args;
constexpr size_t Dropout3dSchema_TTFB::max_pos_args;
constexpr char const* Dropout3dSchema_TTFB::signature;
FunctionDef Dropout3dSchema_TTFB::function_def = {
/*name*/"dropout3d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"p", /*default_value*/float(0.5), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"training", /*default_value*/bool(true), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* dropout3d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("dropout3d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::Dropout3dSchema_TTFB> parser("dropout3d");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Dropout3d(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<float>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct PadSchema_TTI64lSSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& pad, const std::string& mode, const Scalar& value);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Pad;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor x, Int64List pad, String mode=\"constant\", Scalar value=0)";
  static FunctionDef function_def;
};

constexpr size_t PadSchema_TTI64lSSc::max_args;
constexpr size_t PadSchema_TTI64lSSc::max_pos_args;
constexpr char const* PadSchema_TTI64lSSc::signature;
FunctionDef PadSchema_TTI64lSSc::function_def = {
/*name*/"pad",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"pad", /*value_type*/ValueTypeOf<std::vector<int64_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"mode", /*default_value*/std::string("constant"), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"value", /*default_value*/Scalar(0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* pad(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("pad");
  PythonFrameGuard pf;
  static PythonArgParser<functional::PadSchema_TTI64lSSc> parser("pad");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Pad(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int64_t>>(), r[2].As<std::string>(), r[3].As<Scalar>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SiluSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Silu;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t SiluSchema_TT::max_args;
constexpr size_t SiluSchema_TT::max_pos_args;
constexpr char const* SiluSchema_TT::signature;
FunctionDef SiluSchema_TT::function_def = {
/*name*/"silu",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* silu(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("silu");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SiluSchema_TT> parser("silu");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Silu(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MishSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Mish;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t MishSchema_TT::max_args;
constexpr size_t MishSchema_TT::max_pos_args;
constexpr char const* MishSchema_TT::signature;
FunctionDef MishSchema_TT::function_def = {
/*name*/"mish",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* mish(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("mish");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MishSchema_TT> parser("mish");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Mish(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SeluSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Selu;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t SeluSchema_TT::max_args;
constexpr size_t SeluSchema_TT::max_pos_args;
constexpr char const* SeluSchema_TT::signature;
FunctionDef SeluSchema_TT::function_def = {
/*name*/"selu",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* selu(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("selu");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SeluSchema_TT> parser("selu");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Selu(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SoftSignSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::SoftSign;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x)";
  static FunctionDef function_def;
};

constexpr size_t SoftSignSchema_TT::max_args;
constexpr size_t SoftSignSchema_TT::max_pos_args;
constexpr char const* SoftSignSchema_TT::signature;
FunctionDef SoftSignSchema_TT::function_def = {
/*name*/"softsign",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* softsign(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("softsign");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SoftSignSchema_TT> parser("softsign");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::SoftSign(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct DiagSchema_TTI32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, int32_t diagonal);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Diag;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Int32 diagonal=0)";
  static FunctionDef function_def;
};

constexpr size_t DiagSchema_TTI32::max_args;
constexpr size_t DiagSchema_TTI32::max_pos_args;
constexpr char const* DiagSchema_TTI32::signature;
FunctionDef DiagSchema_TTI32::function_def = {
/*name*/"diag",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"diagonal", /*default_value*/int32_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* diag(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("diag");
  PythonFrameGuard pf;
  static PythonArgParser<functional::DiagSchema_TTI32> parser("diag");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Diag(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct DiagonalSchema_TTI32I32I32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, int32_t offset, int32_t dim1, int32_t dim2);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Diagonal;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor x, Int32 offset=0, Int32 dim1=0, Int32 dim2=1)";
  static FunctionDef function_def;
};

constexpr size_t DiagonalSchema_TTI32I32I32::max_args;
constexpr size_t DiagonalSchema_TTI32I32I32::max_pos_args;
constexpr char const* DiagonalSchema_TTI32I32I32::signature;
FunctionDef DiagonalSchema_TTI32I32I32::function_def = {
/*name*/"diagonal",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"offset", /*default_value*/int32_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim1", /*default_value*/int32_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim2", /*default_value*/int32_t(1), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* diagonal(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("diagonal");
  PythonFrameGuard pf;
  static PythonArgParser<functional::DiagonalSchema_TTI32I32I32> parser("diagonal");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Diagonal(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>(), r[2].As<int32_t>(), r[3].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct DimScatterSchema_TTI32TTSB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int32_t dim, const std::shared_ptr<one::Tensor>& index, const std::shared_ptr<one::Tensor>& src, const Optional<std::string>& reduce, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::DimScatter;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor input, Int32 dim, Tensor index, Tensor src, *, String reduce=None, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t DimScatterSchema_TTI32TTSB::max_args;
constexpr size_t DimScatterSchema_TTI32TTSB::max_pos_args;
constexpr char const* DimScatterSchema_TTI32TTSB::signature;
FunctionDef DimScatterSchema_TTI32TTSB::function_def = {
/*name*/"scatter",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"index", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"src", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"reduce", /*default_value*/Optional<std::string>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct DimScatterScalarSchema_TTI32TScSB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int32_t dim, const std::shared_ptr<one::Tensor>& index, const Scalar& src, const Optional<std::string>& reduce, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::DimScatterScalar;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor input, Int32 dim, Tensor index, Scalar src, *, String reduce=None, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t DimScatterScalarSchema_TTI32TScSB::max_args;
constexpr size_t DimScatterScalarSchema_TTI32TScSB::max_pos_args;
constexpr char const* DimScatterScalarSchema_TTI32TScSB::signature;
FunctionDef DimScatterScalarSchema_TTI32TScSB::function_def = {
/*name*/"scatter",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"index", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"src", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"reduce", /*default_value*/Optional<std::string>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* scatter(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("scatter");
  PythonFrameGuard pf;
  static PythonArgParser<functional::DimScatterSchema_TTI32TTSB, functional::DimScatterScalarSchema_TTI32TScSB> parser("scatter");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::DimScatter(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<std::shared_ptr<one::Tensor>>(), r[4].As<Optional<std::string>>(), r[5].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::DimScatterScalar(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<Scalar>(), r[4].As<Optional<std::string>>(), r[5].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct DimScatterAddSchema_TTI32TTB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int32_t dim, const std::shared_ptr<one::Tensor>& index, const std::shared_ptr<one::Tensor>& src, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::DimScatterAdd;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor input, Int32 dim, Tensor index, Tensor src, *, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t DimScatterAddSchema_TTI32TTB::max_args;
constexpr size_t DimScatterAddSchema_TTI32TTB::max_pos_args;
constexpr char const* DimScatterAddSchema_TTI32TTB::signature;
FunctionDef DimScatterAddSchema_TTI32TTB::function_def = {
/*name*/"scatter_add",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"index", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"src", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct DimScatterAddScalarSchema_TTI32TScB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int32_t dim, const std::shared_ptr<one::Tensor>& index, const Scalar& src, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::DimScatterAddScalar;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor input, Int32 dim, Tensor index, Scalar src, *, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t DimScatterAddScalarSchema_TTI32TScB::max_args;
constexpr size_t DimScatterAddScalarSchema_TTI32TScB::max_pos_args;
constexpr char const* DimScatterAddScalarSchema_TTI32TScB::signature;
FunctionDef DimScatterAddScalarSchema_TTI32TScB::function_def = {
/*name*/"scatter_add",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"index", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"src", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* scatter_add(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("scatter_add");
  PythonFrameGuard pf;
  static PythonArgParser<functional::DimScatterAddSchema_TTI32TTB, functional::DimScatterAddScalarSchema_TTI32TScB> parser("scatter_add");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::DimScatterAdd(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<std::shared_ptr<one::Tensor>>(), r[4].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::DimScatterAddScalar(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<Scalar>(), r[4].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TensorSetItemSchema_VTTiT {
  using FType = Maybe<void> (const std::shared_ptr<one::Tensor>& x, const TensorIndex& index, const std::shared_ptr<one::Tensor>& value);
  using R = Maybe<void>;

  static constexpr FType* func = &functional::TensorSetItem;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Void (Tensor x, TensorIndex index, Tensor value)";
  static FunctionDef function_def;
};

constexpr size_t TensorSetItemSchema_VTTiT::max_args;
constexpr size_t TensorSetItemSchema_VTTiT::max_pos_args;
constexpr char const* TensorSetItemSchema_VTTiT::signature;
FunctionDef TensorSetItemSchema_VTTiT::function_def = {
/*name*/"tensor_setitem",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<void>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"index", /*value_type*/ValueTypeOf<TensorIndex>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"value", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* tensor_setitem(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("tensor_setitem");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TensorSetItemSchema_VTTiT> parser("tensor_setitem");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::TensorSetItem(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<TensorIndex>(), r[2].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AvgPool1DSchema_TTI32lI32lI32lBBI32S {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& kernel_size, const Optional<std::vector<int32_t>>& stride, const std::vector<int32_t>& padding, bool ceil_mode, bool count_include_pad, int32_t divisor_override, const std::string& data_format);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::AvgPool1D;
  static constexpr size_t max_args = 8;
  static constexpr size_t max_pos_args = 8;
  static constexpr char const* signature = "Tensor (Tensor input, Int32List kernel_size, Int32List stride=None, Int32List padding=0, Bool ceil_mode=False, Bool count_include_pad=True, Int32 divisor_override=0, String data_format=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t AvgPool1DSchema_TTI32lI32lI32lBBI32S::max_args;
constexpr size_t AvgPool1DSchema_TTI32lI32lI32lBBI32S::max_pos_args;
constexpr char const* AvgPool1DSchema_TTI32lI32lI32lBBI32S::signature;
FunctionDef AvgPool1DSchema_TTI32lI32lI32lBBI32S::function_def = {
/*name*/"avg_pool1d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"kernel_size", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"stride", /*default_value*/Optional<std::vector<int32_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"padding", /*default_value*/std::vector<int32_t>({0}), /*size*/1, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"ceil_mode", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"count_include_pad", /*default_value*/bool(true), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"divisor_override", /*default_value*/int32_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"data_format", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* avg_pool1d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("avg_pool1d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AvgPool1DSchema_TTI32lI32lI32lBBI32S> parser("avg_pool1d");
  ParsedArgs<8> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::AvgPool1D(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<Optional<std::vector<int32_t>>>(), r[3].As<std::vector<int32_t>>(), r[4].As<bool>(), r[5].As<bool>(), r[6].As<int32_t>(), r[7].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AvgPool2DSchema_TTI32lI32lI32lBBI32S {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& kernel_size, const Optional<std::vector<int32_t>>& stride, const std::vector<int32_t>& padding, bool ceil_mode, bool count_include_pad, int32_t divisor_override, const std::string& data_format);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::AvgPool2D;
  static constexpr size_t max_args = 8;
  static constexpr size_t max_pos_args = 8;
  static constexpr char const* signature = "Tensor (Tensor input, Int32List kernel_size, Int32List stride=None, Int32List padding=0, Bool ceil_mode=False, Bool count_include_pad=True, Int32 divisor_override=0, String data_format=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t AvgPool2DSchema_TTI32lI32lI32lBBI32S::max_args;
constexpr size_t AvgPool2DSchema_TTI32lI32lI32lBBI32S::max_pos_args;
constexpr char const* AvgPool2DSchema_TTI32lI32lI32lBBI32S::signature;
FunctionDef AvgPool2DSchema_TTI32lI32lI32lBBI32S::function_def = {
/*name*/"avg_pool2d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"kernel_size", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/2, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"stride", /*default_value*/Optional<std::vector<int32_t>>(), /*size*/2, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"padding", /*default_value*/std::vector<int32_t>({0, 0}), /*size*/2, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"ceil_mode", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"count_include_pad", /*default_value*/bool(true), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"divisor_override", /*default_value*/int32_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"data_format", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* avg_pool2d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("avg_pool2d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AvgPool2DSchema_TTI32lI32lI32lBBI32S> parser("avg_pool2d");
  ParsedArgs<8> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::AvgPool2D(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<Optional<std::vector<int32_t>>>(), r[3].As<std::vector<int32_t>>(), r[4].As<bool>(), r[5].As<bool>(), r[6].As<int32_t>(), r[7].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AvgPool3DSchema_TTI32lI32lI32lBBI32S {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::vector<int32_t>& kernel_size, const Optional<std::vector<int32_t>>& stride, const std::vector<int32_t>& padding, bool ceil_mode, bool count_include_pad, int32_t divisor_override, const std::string& data_format);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::AvgPool3D;
  static constexpr size_t max_args = 8;
  static constexpr size_t max_pos_args = 8;
  static constexpr char const* signature = "Tensor (Tensor input, Int32List kernel_size, Int32List stride=None, Int32List padding=0, Bool ceil_mode=False, Bool count_include_pad=True, Int32 divisor_override=0, String data_format=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t AvgPool3DSchema_TTI32lI32lI32lBBI32S::max_args;
constexpr size_t AvgPool3DSchema_TTI32lI32lI32lBBI32S::max_pos_args;
constexpr char const* AvgPool3DSchema_TTI32lI32lI32lBBI32S::signature;
FunctionDef AvgPool3DSchema_TTI32lI32lI32lBBI32S::function_def = {
/*name*/"avg_pool3d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"kernel_size", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/3, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"stride", /*default_value*/Optional<std::vector<int32_t>>(), /*size*/3, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"padding", /*default_value*/std::vector<int32_t>({0, 0, 0}), /*size*/3, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"ceil_mode", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"count_include_pad", /*default_value*/bool(true), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"divisor_override", /*default_value*/int32_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"data_format", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* avg_pool3d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("avg_pool3d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AvgPool3DSchema_TTI32lI32lI32lBBI32S> parser("avg_pool3d");
  ParsedArgs<8> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::AvgPool3D(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<Optional<std::vector<int32_t>>>(), r[3].As<std::vector<int32_t>>(), r[4].As<bool>(), r[5].As<bool>(), r[6].As<int32_t>(), r[7].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MinimumSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Minimum;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t MinimumSchema_TTT::max_args;
constexpr size_t MinimumSchema_TTT::max_pos_args;
constexpr char const* MinimumSchema_TTT::signature;
FunctionDef MinimumSchema_TTT::function_def = {
/*name*/"minimum",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* minimum(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("minimum");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MinimumSchema_TTT> parser("minimum");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Minimum(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MaximumSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Maximum;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t MaximumSchema_TTT::max_args;
constexpr size_t MaximumSchema_TTT::max_pos_args;
constexpr char const* MaximumSchema_TTT::signature;
FunctionDef MaximumSchema_TTT::function_def = {
/*name*/"maximum",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* maximum(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("maximum");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MaximumSchema_TTT> parser("maximum");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Maximum(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct StackSchema_TTtI64 {
  using FType = Maybe<one::Tensor> (const TensorTuple& inputs, int64_t dim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Stack;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (TensorTuple inputs, Int64 dim=0)";
  static FunctionDef function_def;
};

constexpr size_t StackSchema_TTtI64::max_args;
constexpr size_t StackSchema_TTtI64::max_pos_args;
constexpr char const* StackSchema_TTtI64::signature;
FunctionDef StackSchema_TTtI64::function_def = {
/*name*/"stack",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"inputs", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/int64_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* stack(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("stack");
  PythonFrameGuard pf;
  static PythonArgParser<functional::StackSchema_TTtI64> parser("stack");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Stack(r[0].As<TensorTuple>(), r[1].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AtLeast1DSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::AtLeast1D;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input)";
  static FunctionDef function_def;
};

constexpr size_t AtLeast1DSchema_TT::max_args;
constexpr size_t AtLeast1DSchema_TT::max_pos_args;
constexpr char const* AtLeast1DSchema_TT::signature;
FunctionDef AtLeast1DSchema_TT::function_def = {
/*name*/"atleast_1d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct AtLeast1DSchema_TtTt {
  using FType = Maybe<one::TensorTuple> (const TensorTuple& tensors);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::AtLeast1D;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "TensorTuple (TensorTuple tensors)";
  static FunctionDef function_def;
};

constexpr size_t AtLeast1DSchema_TtTt::max_args;
constexpr size_t AtLeast1DSchema_TtTt::max_pos_args;
constexpr char const* AtLeast1DSchema_TtTt::signature;
FunctionDef AtLeast1DSchema_TtTt::function_def = {
/*name*/"atleast_1d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"tensors", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* atleast_1d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("atleast_1d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AtLeast1DSchema_TT, functional::AtLeast1DSchema_TtTt> parser("atleast_1d");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::AtLeast1D(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::AtLeast1D(r[0].As<TensorTuple>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AtLeast2DSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::AtLeast2D;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input)";
  static FunctionDef function_def;
};

constexpr size_t AtLeast2DSchema_TT::max_args;
constexpr size_t AtLeast2DSchema_TT::max_pos_args;
constexpr char const* AtLeast2DSchema_TT::signature;
FunctionDef AtLeast2DSchema_TT::function_def = {
/*name*/"atleast_2d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct AtLeast2DSchema_TtTt {
  using FType = Maybe<one::TensorTuple> (const TensorTuple& tensors);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::AtLeast2D;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "TensorTuple (TensorTuple tensors)";
  static FunctionDef function_def;
};

constexpr size_t AtLeast2DSchema_TtTt::max_args;
constexpr size_t AtLeast2DSchema_TtTt::max_pos_args;
constexpr char const* AtLeast2DSchema_TtTt::signature;
FunctionDef AtLeast2DSchema_TtTt::function_def = {
/*name*/"atleast_2d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"tensors", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* atleast_2d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("atleast_2d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AtLeast2DSchema_TT, functional::AtLeast2DSchema_TtTt> parser("atleast_2d");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::AtLeast2D(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::AtLeast2D(r[0].As<TensorTuple>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AtLeast3DSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::AtLeast3D;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input)";
  static FunctionDef function_def;
};

constexpr size_t AtLeast3DSchema_TT::max_args;
constexpr size_t AtLeast3DSchema_TT::max_pos_args;
constexpr char const* AtLeast3DSchema_TT::signature;
FunctionDef AtLeast3DSchema_TT::function_def = {
/*name*/"atleast_3d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct AtLeast3DSchema_TtTt {
  using FType = Maybe<one::TensorTuple> (const TensorTuple& tensors);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::AtLeast3D;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "TensorTuple (TensorTuple tensors)";
  static FunctionDef function_def;
};

constexpr size_t AtLeast3DSchema_TtTt::max_args;
constexpr size_t AtLeast3DSchema_TtTt::max_pos_args;
constexpr char const* AtLeast3DSchema_TtTt::signature;
FunctionDef AtLeast3DSchema_TtTt::function_def = {
/*name*/"atleast_3d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"tensors", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* atleast_3d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("atleast_3d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AtLeast3DSchema_TT, functional::AtLeast3DSchema_TtTt> parser("atleast_3d");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::AtLeast3D(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::AtLeast3D(r[0].As<TensorTuple>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct HStackSchema_TTt {
  using FType = Maybe<one::Tensor> (const TensorTuple& tensors);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::HStack;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (TensorTuple tensors)";
  static FunctionDef function_def;
};

constexpr size_t HStackSchema_TTt::max_args;
constexpr size_t HStackSchema_TTt::max_pos_args;
constexpr char const* HStackSchema_TTt::signature;
FunctionDef HStackSchema_TTt::function_def = {
/*name*/"hstack",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"tensors", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* hstack(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("hstack");
  PythonFrameGuard pf;
  static PythonArgParser<functional::HStackSchema_TTt> parser("hstack");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::HStack(r[0].As<TensorTuple>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct VStackSchema_TTt {
  using FType = Maybe<one::Tensor> (const TensorTuple& tensors);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::VStack;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (TensorTuple tensors)";
  static FunctionDef function_def;
};

constexpr size_t VStackSchema_TTt::max_args;
constexpr size_t VStackSchema_TTt::max_pos_args;
constexpr char const* VStackSchema_TTt::signature;
FunctionDef VStackSchema_TTt::function_def = {
/*name*/"vstack",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"tensors", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* vstack(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("vstack");
  PythonFrameGuard pf;
  static PythonArgParser<functional::VStackSchema_TTt> parser("vstack");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::VStack(r[0].As<TensorTuple>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct DStackSchema_TTt {
  using FType = Maybe<one::Tensor> (const TensorTuple& tensors);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::DStack;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (TensorTuple tensors)";
  static FunctionDef function_def;
};

constexpr size_t DStackSchema_TTt::max_args;
constexpr size_t DStackSchema_TTt::max_pos_args;
constexpr char const* DStackSchema_TTt::signature;
FunctionDef DStackSchema_TTt::function_def = {
/*name*/"dstack",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"tensors", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* dstack(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("dstack");
  PythonFrameGuard pf;
  static PythonArgParser<functional::DStackSchema_TTt> parser("dstack");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::DStack(r[0].As<TensorTuple>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ColumnStackSchema_TTt {
  using FType = Maybe<one::Tensor> (const TensorTuple& tensors);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ColumnStack;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (TensorTuple tensors)";
  static FunctionDef function_def;
};

constexpr size_t ColumnStackSchema_TTt::max_args;
constexpr size_t ColumnStackSchema_TTt::max_pos_args;
constexpr char const* ColumnStackSchema_TTt::signature;
FunctionDef ColumnStackSchema_TTt::function_def = {
/*name*/"column_stack",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"tensors", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* column_stack(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("column_stack");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ColumnStackSchema_TTt> parser("column_stack");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ColumnStack(r[0].As<TensorTuple>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RowStackSchema_TTt {
  using FType = Maybe<one::Tensor> (const TensorTuple& tensors);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::RowStack;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (TensorTuple tensors)";
  static FunctionDef function_def;
};

constexpr size_t RowStackSchema_TTt::max_args;
constexpr size_t RowStackSchema_TTt::max_pos_args;
constexpr char const* RowStackSchema_TTt::signature;
FunctionDef RowStackSchema_TTt::function_def = {
/*name*/"row_stack",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"tensors", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* row_stack(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("row_stack");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RowStackSchema_TTt> parser("row_stack");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::RowStack(r[0].As<TensorTuple>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ToGlobalSchema_TTPSbplSbplBB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const std::vector<Symbol<SbpParallel>>& grad_sbp, bool check_meta, bool copy);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ToGlobal;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 6;
  static constexpr char const* signature = "Tensor (Tensor x, Placement placement, SbpList sbp, SbpList grad_sbp, Bool check_meta, Bool copy=False)";
  static FunctionDef function_def;
};

constexpr size_t ToGlobalSchema_TTPSbplSbplBB::max_args;
constexpr size_t ToGlobalSchema_TTPSbplSbplBB::max_pos_args;
constexpr char const* ToGlobalSchema_TTPSbplSbplBB::signature;
FunctionDef ToGlobalSchema_TTPSbplSbplBB::function_def = {
/*name*/"to_global",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"placement", /*value_type*/ValueTypeOf<Symbol<ParallelDesc>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"sbp", /*value_type*/ValueTypeOf<std::vector<Symbol<SbpParallel>>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"grad_sbp", /*value_type*/ValueTypeOf<std::vector<Symbol<SbpParallel>>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"check_meta", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"copy", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* to_global(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("to_global");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ToGlobalSchema_TTPSbplSbplBB> parser("to_global");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ToGlobal(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Symbol<ParallelDesc>>(), r[2].As<std::vector<Symbol<SbpParallel>>>(), r[3].As<std::vector<Symbol<SbpParallel>>>(), r[4].As<bool>(), r[5].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct GlobalToLocalSchema_TTB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, bool copy);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GlobalToLocal;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Bool copy=False)";
  static FunctionDef function_def;
};

constexpr size_t GlobalToLocalSchema_TTB::max_args;
constexpr size_t GlobalToLocalSchema_TTB::max_pos_args;
constexpr char const* GlobalToLocalSchema_TTB::signature;
FunctionDef GlobalToLocalSchema_TTB::function_def = {
/*name*/"to_local",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"copy", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* to_local(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("to_local");
  PythonFrameGuard pf;
  static PythonArgParser<functional::GlobalToLocalSchema_TTB> parser("to_local");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::GlobalToLocal(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct StreamTouchSchema_VTt {
  using FType = Maybe<void> (const TensorTuple& x);
  using R = Maybe<void>;

  static constexpr FType* func = &functional::StreamTouch;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Void (TensorTuple x)";
  static FunctionDef function_def;
};

constexpr size_t StreamTouchSchema_VTt::max_args;
constexpr size_t StreamTouchSchema_VTt::max_pos_args;
constexpr char const* StreamTouchSchema_VTt::signature;
FunctionDef StreamTouchSchema_VTt::function_def = {
/*name*/"stream_touch",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<void>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* stream_touch(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("stream_touch");
  PythonFrameGuard pf;
  static PythonArgParser<functional::StreamTouchSchema_VTt> parser("stream_touch");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::StreamTouch(r[0].As<TensorTuple>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BroadcastTensorsSchema_TtTtI64B {
  using FType = Maybe<one::TensorTuple> (const TensorTuple& inputs, int64_t src_rank, bool inplace);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::BroadcastTensors;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "TensorTuple (TensorTuple inputs, *, Int64 src_rank=0, Bool inplace=True)";
  static FunctionDef function_def;
};

constexpr size_t BroadcastTensorsSchema_TtTtI64B::max_args;
constexpr size_t BroadcastTensorsSchema_TtTtI64B::max_pos_args;
constexpr char const* BroadcastTensorsSchema_TtTtI64B::signature;
FunctionDef BroadcastTensorsSchema_TtTtI64B::function_def = {
/*name*/"broadcast",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"inputs", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"src_rank", /*default_value*/int64_t(0), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(true), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* broadcast(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("broadcast");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BroadcastTensorsSchema_TtTtI64B> parser("broadcast");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BroadcastTensors(r[0].As<TensorTuple>(), r[1].As<int64_t>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct LocalAllReduceSchema_TTB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::LocalAllReduce;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Bool inplace=False)";
  static FunctionDef function_def;
};

constexpr size_t LocalAllReduceSchema_TTB::max_args;
constexpr size_t LocalAllReduceSchema_TTB::max_pos_args;
constexpr char const* LocalAllReduceSchema_TTB::signature;
FunctionDef LocalAllReduceSchema_TTB::function_def = {
/*name*/"local_all_reduce",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* local_all_reduce(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("local_all_reduce");
  PythonFrameGuard pf;
  static PythonArgParser<functional::LocalAllReduceSchema_TTB> parser("local_all_reduce");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::LocalAllReduce(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct LocalReduceSchema_TTI64B {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, int64_t dst, bool inplace);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::LocalReduce;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor x, *, Int64 dst=0, Bool inplace=True)";
  static FunctionDef function_def;
};

constexpr size_t LocalReduceSchema_TTI64B::max_args;
constexpr size_t LocalReduceSchema_TTI64B::max_pos_args;
constexpr char const* LocalReduceSchema_TTI64B::signature;
FunctionDef LocalReduceSchema_TTI64B::function_def = {
/*name*/"local_reduce",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dst", /*default_value*/int64_t(0), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"inplace", /*default_value*/bool(true), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* local_reduce(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("local_reduce");
  PythonFrameGuard pf;
  static PythonArgParser<functional::LocalReduceSchema_TTI64B> parser("local_reduce");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::LocalReduce(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SelectTopNSchema_TtTtI32 {
  using FType = Maybe<one::TensorTuple> (const TensorTuple& inputs, int32_t n);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::SelectTopN;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "TensorTuple (TensorTuple inputs, Int32 n)";
  static FunctionDef function_def;
};

constexpr size_t SelectTopNSchema_TtTtI32::max_args;
constexpr size_t SelectTopNSchema_TtTtI32::max_pos_args;
constexpr char const* SelectTopNSchema_TtTtI32::signature;
FunctionDef SelectTopNSchema_TtTtI32::function_def = {
/*name*/"select_top_n",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"inputs", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"n", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* select_top_n(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("select_top_n");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SelectTopNSchema_TtTtI32> parser("select_top_n");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::SelectTopN(r[0].As<TensorTuple>(), r[1].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct IdentitySchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& in);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Identity;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor in)";
  static FunctionDef function_def;
};

constexpr size_t IdentitySchema_TT::max_args;
constexpr size_t IdentitySchema_TT::max_pos_args;
constexpr char const* IdentitySchema_TT::signature;
FunctionDef IdentitySchema_TT::function_def = {
/*name*/"identity",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"in", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* identity(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("identity");
  PythonFrameGuard pf;
  static PythonArgParser<functional::IdentitySchema_TT> parser("identity");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Identity(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AmpWhiteIdentitySchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& in);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::AmpWhiteIdentity;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor in)";
  static FunctionDef function_def;
};

constexpr size_t AmpWhiteIdentitySchema_TT::max_args;
constexpr size_t AmpWhiteIdentitySchema_TT::max_pos_args;
constexpr char const* AmpWhiteIdentitySchema_TT::signature;
FunctionDef AmpWhiteIdentitySchema_TT::function_def = {
/*name*/"amp_white_identity",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"in", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* amp_white_identity(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("amp_white_identity");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AmpWhiteIdentitySchema_TT> parser("amp_white_identity");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::AmpWhiteIdentity(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AmpBlackIdentitySchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& in);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::AmpBlackIdentity;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor in)";
  static FunctionDef function_def;
};

constexpr size_t AmpBlackIdentitySchema_TT::max_args;
constexpr size_t AmpBlackIdentitySchema_TT::max_pos_args;
constexpr char const* AmpBlackIdentitySchema_TT::signature;
FunctionDef AmpBlackIdentitySchema_TT::function_def = {
/*name*/"amp_black_identity",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"in", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* amp_black_identity(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("amp_black_identity");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AmpBlackIdentitySchema_TT> parser("amp_black_identity");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::AmpBlackIdentity(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ReshapeLikeSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& like);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ReshapeLike;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor in, Tensor like)";
  static FunctionDef function_def;
};

constexpr size_t ReshapeLikeSchema_TTT::max_args;
constexpr size_t ReshapeLikeSchema_TTT::max_pos_args;
constexpr char const* ReshapeLikeSchema_TTT::signature;
FunctionDef ReshapeLikeSchema_TTT::function_def = {
/*name*/"reshape_like",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"in", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"like", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* reshape_like(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("reshape_like");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ReshapeLikeSchema_TTT> parser("reshape_like");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ReshapeLike(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ReduceSumLikeSchema_TTTI32l {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& like, const std::vector<int32_t>& axis);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::ReduceSumLike;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor in, Tensor like, Int32List axis)";
  static FunctionDef function_def;
};

constexpr size_t ReduceSumLikeSchema_TTTI32l::max_args;
constexpr size_t ReduceSumLikeSchema_TTTI32l::max_pos_args;
constexpr char const* ReduceSumLikeSchema_TTTI32l::signature;
FunctionDef ReduceSumLikeSchema_TTTI32l::function_def = {
/*name*/"reduce_sum_like",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"in", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"like", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"axis", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* reduce_sum_like(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("reduce_sum_like");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ReduceSumLikeSchema_TTTI32l> parser("reduce_sum_like");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::ReduceSumLike(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::vector<int32_t>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RandSchema_TShDtDeGB {
  using FType = Maybe<one::Tensor> (const Shape& size, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Rand;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Shape size, *, DataType dtype=None, Device device=None, Generator generator=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t RandSchema_TShDtDeGB::max_args;
constexpr size_t RandSchema_TShDtDeGB::max_pos_args;
constexpr char const* RandSchema_TShDtDeGB::signature;
FunctionDef RandSchema_TShDtDeGB::function_def = {
/*name*/"rand",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"size", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"device", /*default_value*/Optional<Symbol<Device>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct GlobalRandSchema_TShPSbplDtGB {
  using FType = Maybe<one::Tensor> (const Shape& size, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GlobalRand;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Shape size, *, Placement placement, SbpList sbp, DataType dtype=None, Generator generator=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t GlobalRandSchema_TShPSbplDtGB::max_args;
constexpr size_t GlobalRandSchema_TShPSbplDtGB::max_pos_args;
constexpr char const* GlobalRandSchema_TShPSbplDtGB::signature;
FunctionDef GlobalRandSchema_TShPSbplDtGB::function_def = {
/*name*/"rand",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"size", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"placement", /*value_type*/ValueTypeOf<Symbol<ParallelDesc>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"sbp", /*value_type*/ValueTypeOf<std::vector<Symbol<SbpParallel>>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* rand(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("rand");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RandSchema_TShDtDeGB, functional::GlobalRandSchema_TShPSbplDtGB> parser("rand");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Rand(r[0].As<Shape>(), r[1].As<Optional<Symbol<DType>>>(), r[2].As<Optional<Symbol<Device>>>(), r[3].As<Optional<one::Generator>>(), r[4].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::GlobalRand(r[0].As<Shape>(), r[1].As<Symbol<ParallelDesc>>(), r[2].As<std::vector<Symbol<SbpParallel>>>(), r[3].As<Optional<Symbol<DType>>>(), r[4].As<Optional<one::Generator>>(), r[5].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RandNSchema_TShDtDeGB {
  using FType = Maybe<one::Tensor> (const Shape& size, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::RandN;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Shape size, *, DataType dtype=None, Device device=None, Generator generator=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t RandNSchema_TShDtDeGB::max_args;
constexpr size_t RandNSchema_TShDtDeGB::max_pos_args;
constexpr char const* RandNSchema_TShDtDeGB::signature;
FunctionDef RandNSchema_TShDtDeGB::function_def = {
/*name*/"randn",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"size", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"device", /*default_value*/Optional<Symbol<Device>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct GlobalRandNSchema_TShPSbplDtGB {
  using FType = Maybe<one::Tensor> (const Shape& size, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GlobalRandN;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Shape size, *, Placement placement, SbpList sbp, DataType dtype=None, Generator generator=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t GlobalRandNSchema_TShPSbplDtGB::max_args;
constexpr size_t GlobalRandNSchema_TShPSbplDtGB::max_pos_args;
constexpr char const* GlobalRandNSchema_TShPSbplDtGB::signature;
FunctionDef GlobalRandNSchema_TShPSbplDtGB::function_def = {
/*name*/"randn",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"size", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"placement", /*value_type*/ValueTypeOf<Symbol<ParallelDesc>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"sbp", /*value_type*/ValueTypeOf<std::vector<Symbol<SbpParallel>>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* randn(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("randn");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RandNSchema_TShDtDeGB, functional::GlobalRandNSchema_TShPSbplDtGB> parser("randn");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::RandN(r[0].As<Shape>(), r[1].As<Optional<Symbol<DType>>>(), r[2].As<Optional<Symbol<Device>>>(), r[3].As<Optional<one::Generator>>(), r[4].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::GlobalRandN(r[0].As<Shape>(), r[1].As<Symbol<ParallelDesc>>(), r[2].As<std::vector<Symbol<SbpParallel>>>(), r[3].As<Optional<Symbol<DType>>>(), r[4].As<Optional<one::Generator>>(), r[5].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RandnLikeSchema_TTDtDeGB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::RandnLike;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input, *, DataType dtype=None, Device device=None, Generator generator=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t RandnLikeSchema_TTDtDeGB::max_args;
constexpr size_t RandnLikeSchema_TTDtDeGB::max_pos_args;
constexpr char const* RandnLikeSchema_TTDtDeGB::signature;
FunctionDef RandnLikeSchema_TTDtDeGB::function_def = {
/*name*/"randn_like",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"device", /*default_value*/Optional<Symbol<Device>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct GlobalRandnLikeSchema_TTPSbplDtGB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GlobalRandnLike;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input, *, Placement placement, SbpList sbp, DataType dtype=None, Generator generator=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t GlobalRandnLikeSchema_TTPSbplDtGB::max_args;
constexpr size_t GlobalRandnLikeSchema_TTPSbplDtGB::max_pos_args;
constexpr char const* GlobalRandnLikeSchema_TTPSbplDtGB::signature;
FunctionDef GlobalRandnLikeSchema_TTPSbplDtGB::function_def = {
/*name*/"randn_like",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"placement", /*value_type*/ValueTypeOf<Symbol<ParallelDesc>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"sbp", /*value_type*/ValueTypeOf<std::vector<Symbol<SbpParallel>>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* randn_like(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("randn_like");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RandnLikeSchema_TTDtDeGB, functional::GlobalRandnLikeSchema_TTPSbplDtGB> parser("randn_like");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::RandnLike(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<Symbol<DType>>>(), r[2].As<Optional<Symbol<Device>>>(), r[3].As<Optional<one::Generator>>(), r[4].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::GlobalRandnLike(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Symbol<ParallelDesc>>(), r[2].As<std::vector<Symbol<SbpParallel>>>(), r[3].As<Optional<Symbol<DType>>>(), r[4].As<Optional<one::Generator>>(), r[5].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RandIntSchema_TI64I64ShDtDeGB {
  using FType = Maybe<one::Tensor> (int64_t low, int64_t high, const Shape& size, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::RandInt;
  static constexpr size_t max_args = 7;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Int64 low, Int64 high, Shape size, *, DataType dtype=None, Device device=None, Generator generator=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t RandIntSchema_TI64I64ShDtDeGB::max_args;
constexpr size_t RandIntSchema_TI64I64ShDtDeGB::max_pos_args;
constexpr char const* RandIntSchema_TI64I64ShDtDeGB::signature;
FunctionDef RandIntSchema_TI64I64ShDtDeGB::function_def = {
/*name*/"randint",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"low", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"high", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"size", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"device", /*default_value*/Optional<Symbol<Device>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct RandIntSchema_TI64ShDtDeGB {
  using FType = Maybe<one::Tensor> (int64_t high, const Shape& size, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::RandInt;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Int64 high, Shape size, *, DataType dtype=None, Device device=None, Generator generator=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t RandIntSchema_TI64ShDtDeGB::max_args;
constexpr size_t RandIntSchema_TI64ShDtDeGB::max_pos_args;
constexpr char const* RandIntSchema_TI64ShDtDeGB::signature;
FunctionDef RandIntSchema_TI64ShDtDeGB::function_def = {
/*name*/"randint",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"high", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"size", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"device", /*default_value*/Optional<Symbol<Device>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct GlobalRandIntSchema_TI64I64ShPSbplDtGB {
  using FType = Maybe<one::Tensor> (int64_t low, int64_t high, const Shape& size, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GlobalRandInt;
  static constexpr size_t max_args = 8;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Int64 low, Int64 high, Shape size, *, Placement placement, SbpList sbp, DataType dtype=None, Generator generator=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t GlobalRandIntSchema_TI64I64ShPSbplDtGB::max_args;
constexpr size_t GlobalRandIntSchema_TI64I64ShPSbplDtGB::max_pos_args;
constexpr char const* GlobalRandIntSchema_TI64I64ShPSbplDtGB::signature;
FunctionDef GlobalRandIntSchema_TI64I64ShPSbplDtGB::function_def = {
/*name*/"randint",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"low", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"high", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"size", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"placement", /*value_type*/ValueTypeOf<Symbol<ParallelDesc>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"sbp", /*value_type*/ValueTypeOf<std::vector<Symbol<SbpParallel>>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct GlobalRandIntSchema_TI64ShPSbplDtGB {
  using FType = Maybe<one::Tensor> (int64_t high, const Shape& size, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GlobalRandInt;
  static constexpr size_t max_args = 7;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Int64 high, Shape size, *, Placement placement, SbpList sbp, DataType dtype=None, Generator generator=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t GlobalRandIntSchema_TI64ShPSbplDtGB::max_args;
constexpr size_t GlobalRandIntSchema_TI64ShPSbplDtGB::max_pos_args;
constexpr char const* GlobalRandIntSchema_TI64ShPSbplDtGB::signature;
FunctionDef GlobalRandIntSchema_TI64ShPSbplDtGB::function_def = {
/*name*/"randint",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"high", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"size", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"placement", /*value_type*/ValueTypeOf<Symbol<ParallelDesc>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"sbp", /*value_type*/ValueTypeOf<std::vector<Symbol<SbpParallel>>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* randint(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("randint");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RandIntSchema_TI64I64ShDtDeGB, functional::RandIntSchema_TI64ShDtDeGB, functional::GlobalRandIntSchema_TI64I64ShPSbplDtGB, functional::GlobalRandIntSchema_TI64ShPSbplDtGB> parser("randint");
  ParsedArgs<8> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::RandInt(r[0].As<int64_t>(), r[1].As<int64_t>(), r[2].As<Shape>(), r[3].As<Optional<Symbol<DType>>>(), r[4].As<Optional<Symbol<Device>>>(), r[5].As<Optional<one::Generator>>(), r[6].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::RandInt(r[0].As<int64_t>(), r[1].As<Shape>(), r[2].As<Optional<Symbol<DType>>>(), r[3].As<Optional<Symbol<Device>>>(), r[4].As<Optional<one::Generator>>(), r[5].As<bool>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::GlobalRandInt(r[0].As<int64_t>(), r[1].As<int64_t>(), r[2].As<Shape>(), r[3].As<Symbol<ParallelDesc>>(), r[4].As<std::vector<Symbol<SbpParallel>>>(), r[5].As<Optional<Symbol<DType>>>(), r[6].As<Optional<one::Generator>>(), r[7].As<bool>()));
  }
  if (idx == 3) {
    return CastToPyObject(functional::GlobalRandInt(r[0].As<int64_t>(), r[1].As<Shape>(), r[2].As<Symbol<ParallelDesc>>(), r[3].As<std::vector<Symbol<SbpParallel>>>(), r[4].As<Optional<Symbol<DType>>>(), r[5].As<Optional<one::Generator>>(), r[6].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RandIntLikeSchema_TTI64I64DtDeGB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, int64_t low, int64_t high, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::RandIntLike;
  static constexpr size_t max_args = 7;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Int64 low, Int64 high, *, DataType dtype=None, Device device=None, Generator generator=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t RandIntLikeSchema_TTI64I64DtDeGB::max_args;
constexpr size_t RandIntLikeSchema_TTI64I64DtDeGB::max_pos_args;
constexpr char const* RandIntLikeSchema_TTI64I64DtDeGB::signature;
FunctionDef RandIntLikeSchema_TTI64I64DtDeGB::function_def = {
/*name*/"randint_like",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"low", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"high", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"device", /*default_value*/Optional<Symbol<Device>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct RandIntLikeSchema_TTI64DtDeGB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, int64_t high, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Generator>& generator, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::RandIntLike;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Int64 high, *, DataType dtype=None, Device device=None, Generator generator=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t RandIntLikeSchema_TTI64DtDeGB::max_args;
constexpr size_t RandIntLikeSchema_TTI64DtDeGB::max_pos_args;
constexpr char const* RandIntLikeSchema_TTI64DtDeGB::signature;
FunctionDef RandIntLikeSchema_TTI64DtDeGB::function_def = {
/*name*/"randint_like",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"high", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"device", /*default_value*/Optional<Symbol<Device>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct GlobalRandIntLikeSchema_TTI64I64PSbplDtGB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, int64_t low, int64_t high, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GlobalRandIntLike;
  static constexpr size_t max_args = 8;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Int64 low, Int64 high, *, Placement placement, SbpList sbp, DataType dtype=None, Generator generator=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t GlobalRandIntLikeSchema_TTI64I64PSbplDtGB::max_args;
constexpr size_t GlobalRandIntLikeSchema_TTI64I64PSbplDtGB::max_pos_args;
constexpr char const* GlobalRandIntLikeSchema_TTI64I64PSbplDtGB::signature;
FunctionDef GlobalRandIntLikeSchema_TTI64I64PSbplDtGB::function_def = {
/*name*/"randint_like",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"low", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"high", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"placement", /*value_type*/ValueTypeOf<Symbol<ParallelDesc>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"sbp", /*value_type*/ValueTypeOf<std::vector<Symbol<SbpParallel>>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct GlobalRandIntLikeSchema_TTI64PSbplDtGB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, int64_t high, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<Symbol<DType>>& dtype, const Optional<one::Generator>& generator, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GlobalRandIntLike;
  static constexpr size_t max_args = 7;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Int64 high, *, Placement placement, SbpList sbp, DataType dtype=None, Generator generator=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t GlobalRandIntLikeSchema_TTI64PSbplDtGB::max_args;
constexpr size_t GlobalRandIntLikeSchema_TTI64PSbplDtGB::max_pos_args;
constexpr char const* GlobalRandIntLikeSchema_TTI64PSbplDtGB::signature;
FunctionDef GlobalRandIntLikeSchema_TTI64PSbplDtGB::function_def = {
/*name*/"randint_like",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"high", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"placement", /*value_type*/ValueTypeOf<Symbol<ParallelDesc>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"sbp", /*value_type*/ValueTypeOf<std::vector<Symbol<SbpParallel>>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* randint_like(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("randint_like");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RandIntLikeSchema_TTI64I64DtDeGB, functional::RandIntLikeSchema_TTI64DtDeGB, functional::GlobalRandIntLikeSchema_TTI64I64PSbplDtGB, functional::GlobalRandIntLikeSchema_TTI64PSbplDtGB> parser("randint_like");
  ParsedArgs<8> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::RandIntLike(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>(), r[2].As<int64_t>(), r[3].As<Optional<Symbol<DType>>>(), r[4].As<Optional<Symbol<Device>>>(), r[5].As<Optional<one::Generator>>(), r[6].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::RandIntLike(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>(), r[2].As<Optional<Symbol<DType>>>(), r[3].As<Optional<Symbol<Device>>>(), r[4].As<Optional<one::Generator>>(), r[5].As<bool>()));
  }
  if (idx == 2) {
    return CastToPyObject(functional::GlobalRandIntLike(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>(), r[2].As<int64_t>(), r[3].As<Symbol<ParallelDesc>>(), r[4].As<std::vector<Symbol<SbpParallel>>>(), r[5].As<Optional<Symbol<DType>>>(), r[6].As<Optional<one::Generator>>(), r[7].As<bool>()));
  }
  if (idx == 3) {
    return CastToPyObject(functional::GlobalRandIntLike(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>(), r[2].As<Symbol<ParallelDesc>>(), r[3].As<std::vector<Symbol<SbpParallel>>>(), r[4].As<Optional<Symbol<DType>>>(), r[5].As<Optional<one::Generator>>(), r[6].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RandPermSchema_TI32GDtDeB {
  using FType = Maybe<one::Tensor> (int32_t n, const Optional<one::Generator>& generator, const Symbol<DType>& dtype, const Optional<Symbol<Device>>& device, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::RandPerm;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Int32 n, *, Generator generator=None, DataType dtype=kInt64, Device device=None, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t RandPermSchema_TI32GDtDeB::max_args;
constexpr size_t RandPermSchema_TI32GDtDeB::max_pos_args;
constexpr char const* RandPermSchema_TI32GDtDeB::signature;
FunctionDef RandPermSchema_TI32GDtDeB::function_def = {
/*name*/"randperm",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"n", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"dtype", /*default_value*/Symbol<DType>(DType::Int64()), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"device", /*default_value*/Optional<Symbol<Device>>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

struct GlobalRandPermSchema_TI32PSbplGDtB {
  using FType = Maybe<one::Tensor> (int32_t n, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp, const Optional<one::Generator>& generator, const Symbol<DType>& dtype, bool requires_grad);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GlobalRandPerm;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Int32 n, *, Placement placement, SbpList sbp, Generator generator=None, DataType dtype=kInt64, Bool requires_grad=False)";
  static FunctionDef function_def;
};

constexpr size_t GlobalRandPermSchema_TI32PSbplGDtB::max_args;
constexpr size_t GlobalRandPermSchema_TI32PSbplGDtB::max_pos_args;
constexpr char const* GlobalRandPermSchema_TI32PSbplGDtB::signature;
FunctionDef GlobalRandPermSchema_TI32PSbplGDtB::function_def = {
/*name*/"randperm",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"n", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"placement", /*value_type*/ValueTypeOf<Symbol<ParallelDesc>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"sbp", /*value_type*/ValueTypeOf<std::vector<Symbol<SbpParallel>>>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true),
  ArgumentDef(/*name*/"dtype", /*default_value*/Symbol<DType>(DType::Int64()), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"requires_grad", /*default_value*/bool(false), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* randperm(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("randperm");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RandPermSchema_TI32GDtDeB, functional::GlobalRandPermSchema_TI32PSbplGDtB> parser("randperm");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::RandPerm(r[0].As<int32_t>(), r[1].As<Optional<one::Generator>>(), r[2].As<Symbol<DType>>(), r[3].As<Optional<Symbol<Device>>>(), r[4].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::GlobalRandPerm(r[0].As<int32_t>(), r[1].As<Symbol<ParallelDesc>>(), r[2].As<std::vector<Symbol<SbpParallel>>>(), r[3].As<Optional<one::Generator>>(), r[4].As<Symbol<DType>>(), r[5].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct UnfoldTensorSchema_TTI32I32I32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, int32_t dimension, int32_t size, int32_t step);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::UnfoldTensor;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor x, Int32 dimension, Int32 size, Int32 step)";
  static FunctionDef function_def;
};

constexpr size_t UnfoldTensorSchema_TTI32I32I32::max_args;
constexpr size_t UnfoldTensorSchema_TTI32I32I32::max_pos_args;
constexpr char const* UnfoldTensorSchema_TTI32I32I32::signature;
FunctionDef UnfoldTensorSchema_TTI32I32I32::function_def = {
/*name*/"unfold_tensor",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dimension", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"size", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"step", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* unfold_tensor(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("unfold_tensor");
  PythonFrameGuard pf;
  static PythonArgParser<functional::UnfoldTensorSchema_TTI32I32I32> parser("unfold_tensor");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::UnfoldTensor(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>(), r[2].As<int32_t>(), r[3].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct UnfoldSchema_TTI32lI32lI32lI32lS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& dilation, const std::vector<int32_t>& padding, const std::vector<int32_t>& stride, const std::string& data_format);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Unfold;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 6;
  static constexpr char const* signature = "Tensor (Tensor x, Int32List kernel_size, Int32List dilation=1, Int32List padding=0, Int32List stride=1, String data_format=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t UnfoldSchema_TTI32lI32lI32lI32lS::max_args;
constexpr size_t UnfoldSchema_TTI32lI32lI32lI32lS::max_pos_args;
constexpr char const* UnfoldSchema_TTI32lI32lI32lI32lS::signature;
FunctionDef UnfoldSchema_TTI32lI32lI32lI32lS::function_def = {
/*name*/"unfold",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"kernel_size", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/2, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dilation", /*default_value*/std::vector<int32_t>({1, 1}), /*size*/2, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"padding", /*default_value*/std::vector<int32_t>({0, 0}), /*size*/2, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"stride", /*default_value*/std::vector<int32_t>({1, 1}), /*size*/2, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"data_format", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* unfold(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("unfold");
  PythonFrameGuard pf;
  static PythonArgParser<functional::UnfoldSchema_TTI32lI32lI32lI32lS> parser("unfold");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Unfold(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<std::vector<int32_t>>(), r[3].As<std::vector<int32_t>>(), r[4].As<std::vector<int32_t>>(), r[5].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct FoldSchema_TTI32lI32lI32lI32lI32lS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& output_size, const std::vector<int32_t>& kernel_size, const std::vector<int32_t>& dilation, const std::vector<int32_t>& padding, const std::vector<int32_t>& stride, const std::string& data_format);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Fold;
  static constexpr size_t max_args = 7;
  static constexpr size_t max_pos_args = 7;
  static constexpr char const* signature = "Tensor (Tensor x, Int32List output_size, Int32List kernel_size, Int32List dilation=1, Int32List padding=0, Int32List stride=1, String data_format=\"channels_first\")";
  static FunctionDef function_def;
};

constexpr size_t FoldSchema_TTI32lI32lI32lI32lI32lS::max_args;
constexpr size_t FoldSchema_TTI32lI32lI32lI32lI32lS::max_pos_args;
constexpr char const* FoldSchema_TTI32lI32lI32lI32lI32lS::signature;
FunctionDef FoldSchema_TTI32lI32lI32lI32lI32lS::function_def = {
/*name*/"fold",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"output_size", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"kernel_size", /*value_type*/ValueTypeOf<std::vector<int32_t>>(), /*size*/2, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dilation", /*default_value*/std::vector<int32_t>({1, 1}), /*size*/2, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"padding", /*default_value*/std::vector<int32_t>({0, 0}), /*size*/2, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"stride", /*default_value*/std::vector<int32_t>({1, 1}), /*size*/2, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"data_format", /*default_value*/std::string("channels_first"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* fold(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("fold");
  PythonFrameGuard pf;
  static PythonArgParser<functional::FoldSchema_TTI32lI32lI32lI32lI32lS> parser("fold");
  ParsedArgs<7> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Fold(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int32_t>>(), r[2].As<std::vector<int32_t>>(), r[3].As<std::vector<int32_t>>(), r[4].As<std::vector<int32_t>>(), r[5].As<std::vector<int32_t>>(), r[6].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SplitSchema_TtTI64I64 {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& x, int64_t split_size_or_sections, int64_t dim);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::Split;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "TensorTuple (Tensor x, Int64 split_size_or_sections, Int64 dim=0)";
  static FunctionDef function_def;
};

constexpr size_t SplitSchema_TtTI64I64::max_args;
constexpr size_t SplitSchema_TtTI64I64::max_pos_args;
constexpr char const* SplitSchema_TtTI64I64::signature;
FunctionDef SplitSchema_TtTI64I64::function_def = {
/*name*/"split",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"split_size_or_sections", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/int64_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct SplitWithSizeSchema_TtTI64lI64 {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& split_size_or_sections, int64_t dim);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::SplitWithSize;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "TensorTuple (Tensor x, Int64List split_size_or_sections, Int64 dim=0)";
  static FunctionDef function_def;
};

constexpr size_t SplitWithSizeSchema_TtTI64lI64::max_args;
constexpr size_t SplitWithSizeSchema_TtTI64lI64::max_pos_args;
constexpr char const* SplitWithSizeSchema_TtTI64lI64::signature;
FunctionDef SplitWithSizeSchema_TtTI64lI64::function_def = {
/*name*/"split",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"split_size_or_sections", /*value_type*/ValueTypeOf<std::vector<int64_t>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/int64_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* split(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("split");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SplitSchema_TtTI64I64, functional::SplitWithSizeSchema_TtTI64lI64> parser("split");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Split(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>(), r[2].As<int64_t>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::SplitWithSize(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int64_t>>(), r[2].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct UnbindSchema_TtTI64 {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& x, int64_t dim);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::Unbind;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "TensorTuple (Tensor x, Int64 dim=0)";
  static FunctionDef function_def;
};

constexpr size_t UnbindSchema_TtTI64::max_args;
constexpr size_t UnbindSchema_TtTI64::max_pos_args;
constexpr char const* UnbindSchema_TtTI64::signature;
FunctionDef UnbindSchema_TtTI64::function_def = {
/*name*/"unbind",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/int64_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* unbind(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("unbind");
  PythonFrameGuard pf;
  static PythonArgParser<functional::UnbindSchema_TtTI64> parser("unbind");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Unbind(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ChunkSchema_TtTI64I64 {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& x, int64_t chunks, int64_t dim);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::Chunk;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "TensorTuple (Tensor x, Int64 chunks, Int64 dim=0)";
  static FunctionDef function_def;
};

constexpr size_t ChunkSchema_TtTI64I64::max_args;
constexpr size_t ChunkSchema_TtTI64I64::max_pos_args;
constexpr char const* ChunkSchema_TtTI64I64::signature;
FunctionDef ChunkSchema_TtTI64I64::function_def = {
/*name*/"chunk",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"chunks", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/int64_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* chunk(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("chunk");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ChunkSchema_TtTI64I64> parser("chunk");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Chunk(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>(), r[2].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SplitLikeSchema_TtTTtI64 {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& x, const TensorTuple& like, int64_t axis);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::SplitLike;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "TensorTuple (Tensor x, TensorTuple like, Int64 axis)";
  static FunctionDef function_def;
};

constexpr size_t SplitLikeSchema_TtTTtI64::max_args;
constexpr size_t SplitLikeSchema_TtTTtI64::max_pos_args;
constexpr char const* SplitLikeSchema_TtTTtI64::signature;
FunctionDef SplitLikeSchema_TtTTtI64::function_def = {
/*name*/"split_like",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"like", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"axis", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* split_like(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("split_like");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SplitLikeSchema_TtTTtI64> parser("split_like");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::SplitLike(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<TensorTuple>(), r[2].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct PairwiseDistanceSchema_TTTFDB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x1, const std::shared_ptr<one::Tensor>& x2, float p, double eps, bool keepdim);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::PairwiseDistance;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Tensor (Tensor x1, Tensor x2, Float p=2.0, Double eps=1e-6, Bool keepdim=False)";
  static FunctionDef function_def;
};

constexpr size_t PairwiseDistanceSchema_TTTFDB::max_args;
constexpr size_t PairwiseDistanceSchema_TTTFDB::max_pos_args;
constexpr char const* PairwiseDistanceSchema_TTTFDB::signature;
FunctionDef PairwiseDistanceSchema_TTTFDB::function_def = {
/*name*/"pairwise_distance",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x1", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"x2", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"p", /*default_value*/float(2.0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"eps", /*default_value*/double(1e-6), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"keepdim", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* pairwise_distance(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("pairwise_distance");
  PythonFrameGuard pf;
  static PythonArgParser<functional::PairwiseDistanceSchema_TTTFDB> parser("pairwise_distance");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::PairwiseDistance(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<float>(), r[3].As<double>(), r[4].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct CosineSimilaritySchema_TTTI32D {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& y, int32_t dim, double eps);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::CosineSimilarity;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor y, Int32 dim=1, Double eps=1e-8)";
  static FunctionDef function_def;
};

constexpr size_t CosineSimilaritySchema_TTTI32D::max_args;
constexpr size_t CosineSimilaritySchema_TTTI32D::max_pos_args;
constexpr char const* CosineSimilaritySchema_TTTI32D::signature;
FunctionDef CosineSimilaritySchema_TTTI32D::function_def = {
/*name*/"cosine_similarity",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"y", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/int32_t(1), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"eps", /*default_value*/double(1e-8), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* cosine_similarity(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("cosine_similarity");
  PythonFrameGuard pf;
  static PythonArgParser<functional::CosineSimilaritySchema_TTTI32D> parser("cosine_similarity");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::CosineSimilarity(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<int32_t>(), r[3].As<double>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct NormalizeSchema_TTFI32FB {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, float p, int32_t dim, float eps, bool use_l2_norm_kernel);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Normalize;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Tensor (Tensor input, Float p=2.0, Int32 dim=1, Float eps=1e-12, Bool use_l2_norm_kernel=True)";
  static FunctionDef function_def;
};

constexpr size_t NormalizeSchema_TTFI32FB::max_args;
constexpr size_t NormalizeSchema_TTFI32FB::max_pos_args;
constexpr char const* NormalizeSchema_TTFI32FB::signature;
FunctionDef NormalizeSchema_TTFI32FB::function_def = {
/*name*/"normalize",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"p", /*default_value*/float(2.0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*default_value*/int32_t(1), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"eps", /*default_value*/float(1e-12), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"use_l2_norm_kernel", /*default_value*/bool(true), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* normalize(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("normalize");
  PythonFrameGuard pf;
  static PythonArgParser<functional::NormalizeSchema_TTFI32FB> parser("normalize");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Normalize(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<float>(), r[2].As<int32_t>(), r[3].As<float>(), r[4].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct FusedSelfAttentionSchema_TtTI64F {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& hidden_states, int64_t head_size, float alpha);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::FusedSelfAttention;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "TensorTuple (Tensor hidden_states, Int64 head_size=8, Float alpha=1.0)";
  static FunctionDef function_def;
};

constexpr size_t FusedSelfAttentionSchema_TtTI64F::max_args;
constexpr size_t FusedSelfAttentionSchema_TtTI64F::max_pos_args;
constexpr char const* FusedSelfAttentionSchema_TtTI64F::signature;
FunctionDef FusedSelfAttentionSchema_TtTI64F::function_def = {
/*name*/"fused_self_attention",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"hidden_states", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"head_size", /*default_value*/int64_t(8), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"alpha", /*default_value*/float(1.0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* fused_self_attention(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("fused_self_attention");
  PythonFrameGuard pf;
  static PythonArgParser<functional::FusedSelfAttentionSchema_TtTI64F> parser("fused_self_attention");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::FusedSelfAttention(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>(), r[2].As<float>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct FusedScaleTrilSchema_TTI64ScSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, int64_t diagonal, const Scalar& fill_value, const Scalar& scale);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::FusedScaleTril;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor x, Int64 diagonal=0, Scalar fill_value=0, Scalar scale=1)";
  static FunctionDef function_def;
};

constexpr size_t FusedScaleTrilSchema_TTI64ScSc::max_args;
constexpr size_t FusedScaleTrilSchema_TTI64ScSc::max_pos_args;
constexpr char const* FusedScaleTrilSchema_TTI64ScSc::signature;
FunctionDef FusedScaleTrilSchema_TTI64ScSc::function_def = {
/*name*/"fused_scale_tril",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"diagonal", /*default_value*/int64_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"fill_value", /*default_value*/Scalar(0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"scale", /*default_value*/Scalar(1), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* fused_scale_tril(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("fused_scale_tril");
  PythonFrameGuard pf;
  static PythonArgParser<functional::FusedScaleTrilSchema_TTI64ScSc> parser("fused_scale_tril");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::FusedScaleTril(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>(), r[2].As<Scalar>(), r[3].As<Scalar>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct FusedBiasAddGeluSchema_TTTI32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& a, const std::shared_ptr<one::Tensor>& b, int32_t axis);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::FusedBiasAddGelu;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor a, Tensor b, *, Int32 axis)";
  static FunctionDef function_def;
};

constexpr size_t FusedBiasAddGeluSchema_TTTI32::max_args;
constexpr size_t FusedBiasAddGeluSchema_TTTI32::max_pos_args;
constexpr char const* FusedBiasAddGeluSchema_TTTI32::signature;
FunctionDef FusedBiasAddGeluSchema_TTTI32::function_def = {
/*name*/"fused_bias_add_gelu",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"a", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"b", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"axis", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* fused_bias_add_gelu(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("fused_bias_add_gelu");
  PythonFrameGuard pf;
  static PythonArgParser<functional::FusedBiasAddGeluSchema_TTTI32> parser("fused_bias_add_gelu");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::FusedBiasAddGelu(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct FusedBiasAddDropoutSchema_TTTFI32G {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& a, const std::shared_ptr<one::Tensor>& b, float p, int32_t axis, const Optional<one::Generator>& generator);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::FusedBiasAddDropout;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor a, Tensor b, *, Float p=0.5, Int32 axis, Generator generator=None)";
  static FunctionDef function_def;
};

constexpr size_t FusedBiasAddDropoutSchema_TTTFI32G::max_args;
constexpr size_t FusedBiasAddDropoutSchema_TTTFI32G::max_pos_args;
constexpr char const* FusedBiasAddDropoutSchema_TTTFI32G::signature;
FunctionDef FusedBiasAddDropoutSchema_TTTFI32G::function_def = {
/*name*/"fused_bias_add_dropout",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"a", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"b", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"p", /*default_value*/float(0.5), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"axis", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

PyObject* fused_bias_add_dropout(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("fused_bias_add_dropout");
  PythonFrameGuard pf;
  static PythonArgParser<functional::FusedBiasAddDropoutSchema_TTTFI32G> parser("fused_bias_add_dropout");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::FusedBiasAddDropout(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<float>(), r[3].As<int32_t>(), r[4].As<Optional<one::Generator>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct FusedScaleMaskSoftmaxSchema_TTTFF {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& mask, float fill_value, float scale);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::FusedScaleMaskSoftmax;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor mask, *, Float fill_value=0.0, Float scale=1.0)";
  static FunctionDef function_def;
};

constexpr size_t FusedScaleMaskSoftmaxSchema_TTTFF::max_args;
constexpr size_t FusedScaleMaskSoftmaxSchema_TTTFF::max_pos_args;
constexpr char const* FusedScaleMaskSoftmaxSchema_TTTFF::signature;
FunctionDef FusedScaleMaskSoftmaxSchema_TTTFF::function_def = {
/*name*/"fused_scale_mask_softmax",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"mask", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"fill_value", /*default_value*/float(0.0), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"scale", /*default_value*/float(1.0), /*size*/0, /*keyword_only*/true, /*optional*/false)
}
};

PyObject* fused_scale_mask_softmax(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("fused_scale_mask_softmax");
  PythonFrameGuard pf;
  static PythonArgParser<functional::FusedScaleMaskSoftmaxSchema_TTTFF> parser("fused_scale_mask_softmax");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::FusedScaleMaskSoftmax(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<float>(), r[3].As<float>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct FusedScaleMaskSoftmaxDropoutSchema_TtTTFFFBG {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& mask, float fill_value, float scale, float p, bool training, const Optional<one::Generator>& generator);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::FusedScaleMaskSoftmaxDropout;
  static constexpr size_t max_args = 7;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "TensorTuple (Tensor x, Tensor mask, *, Float fill_value=0.0, Float scale=1.0, Float p=0.5, Bool training=True, Generator generator=None)";
  static FunctionDef function_def;
};

constexpr size_t FusedScaleMaskSoftmaxDropoutSchema_TtTTFFFBG::max_args;
constexpr size_t FusedScaleMaskSoftmaxDropoutSchema_TtTTFFFBG::max_pos_args;
constexpr char const* FusedScaleMaskSoftmaxDropoutSchema_TtTTFFFBG::signature;
FunctionDef FusedScaleMaskSoftmaxDropoutSchema_TtTTFFFBG::function_def = {
/*name*/"fused_scale_mask_softmax_dropout",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"mask", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"fill_value", /*default_value*/float(0.0), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"scale", /*default_value*/float(1.0), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"p", /*default_value*/float(0.5), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"training", /*default_value*/bool(true), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

PyObject* fused_scale_mask_softmax_dropout(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("fused_scale_mask_softmax_dropout");
  PythonFrameGuard pf;
  static PythonArgParser<functional::FusedScaleMaskSoftmaxDropoutSchema_TtTTFFFBG> parser("fused_scale_mask_softmax_dropout");
  ParsedArgs<7> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::FusedScaleMaskSoftmaxDropout(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<float>(), r[3].As<float>(), r[4].As<float>(), r[5].As<bool>(), r[6].As<Optional<one::Generator>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct FusedScaleTrilSoftmaxMaskScaleSchema_TtTFI64FFG {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& a, float p, int64_t diagonal, float tril_scale_value, float tril_fill_value, const Optional<one::Generator>& generator);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::FusedScaleTrilSoftmaxMaskScale;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "TensorTuple (Tensor a, *, Float p=0.5, Int64 diagonal, Float tril_scale_value, Float tril_fill_value=0.0, Generator generator=None)";
  static FunctionDef function_def;
};

constexpr size_t FusedScaleTrilSoftmaxMaskScaleSchema_TtTFI64FFG::max_args;
constexpr size_t FusedScaleTrilSoftmaxMaskScaleSchema_TtTFI64FFG::max_pos_args;
constexpr char const* FusedScaleTrilSoftmaxMaskScaleSchema_TtTFI64FFG::signature;
FunctionDef FusedScaleTrilSoftmaxMaskScaleSchema_TtTFI64FFG::function_def = {
/*name*/"fused_scale_tril_softmax_mask_scale",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"a", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"p", /*default_value*/float(0.5), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"diagonal", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"tril_scale_value", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"tril_fill_value", /*default_value*/float(0.0), /*size*/0, /*keyword_only*/true, /*optional*/false),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

PyObject* fused_scale_tril_softmax_mask_scale(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("fused_scale_tril_softmax_mask_scale");
  PythonFrameGuard pf;
  static PythonArgParser<functional::FusedScaleTrilSoftmaxMaskScaleSchema_TtTFI64FFG> parser("fused_scale_tril_softmax_mask_scale");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::FusedScaleTrilSoftmaxMaskScale(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<float>(), r[2].As<int64_t>(), r[3].As<float>(), r[4].As<float>(), r[5].As<Optional<one::Generator>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct FusedMultiHeadAttentionInferenceSchema_TTTTI64BI64I64I64I64I64I64 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& query, const std::shared_ptr<one::Tensor>& key, const std::shared_ptr<one::Tensor>& value, int64_t num_heads, bool causal, int64_t query_hidden_slice_start, int64_t query_hidden_slice_end, int64_t key_hidden_slice_start, int64_t key_hidden_slice_end, int64_t value_hidden_slice_start, int64_t value_hidden_slice_end);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::FusedMultiHeadAttentionInference;
  static constexpr size_t max_args = 11;
  static constexpr size_t max_pos_args = 11;
  static constexpr char const* signature = "Tensor (Tensor query, Tensor key, Tensor value, Int64 num_heads, Bool causal=False, Int64 query_hidden_slice_start=0, Int64 query_hidden_slice_end=-1, Int64 key_hidden_slice_start=0, Int64 key_hidden_slice_end=-1, Int64 value_hidden_slice_start=0, Int64 value_hidden_slice_end=-1)";
  static FunctionDef function_def;
};

constexpr size_t FusedMultiHeadAttentionInferenceSchema_TTTTI64BI64I64I64I64I64I64::max_args;
constexpr size_t FusedMultiHeadAttentionInferenceSchema_TTTTI64BI64I64I64I64I64I64::max_pos_args;
constexpr char const* FusedMultiHeadAttentionInferenceSchema_TTTTI64BI64I64I64I64I64I64::signature;
FunctionDef FusedMultiHeadAttentionInferenceSchema_TTTTI64BI64I64I64I64I64I64::function_def = {
/*name*/"fused_multi_head_attention_inference",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"query", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"key", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"value", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"num_heads", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"causal", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"query_hidden_slice_start", /*default_value*/int64_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"query_hidden_slice_end", /*default_value*/int64_t(-1), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"key_hidden_slice_start", /*default_value*/int64_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"key_hidden_slice_end", /*default_value*/int64_t(-1), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"value_hidden_slice_start", /*default_value*/int64_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"value_hidden_slice_end", /*default_value*/int64_t(-1), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* fused_multi_head_attention_inference(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("fused_multi_head_attention_inference");
  PythonFrameGuard pf;
  static PythonArgParser<functional::FusedMultiHeadAttentionInferenceSchema_TTTTI64BI64I64I64I64I64I64> parser("fused_multi_head_attention_inference");
  ParsedArgs<11> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::FusedMultiHeadAttentionInference(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<int64_t>(), r[4].As<bool>(), r[5].As<int64_t>(), r[6].As<int64_t>(), r[7].As<int64_t>(), r[8].As<int64_t>(), r[9].As<int64_t>(), r[10].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct SendSchema_VTI64B {
  using FType = Maybe<void> (const std::shared_ptr<one::Tensor>& input, int64_t dst, bool send_meta);
  using R = Maybe<void>;

  static constexpr FType* func = &functional::Send;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Void (Tensor input, Int64 dst, Bool send_meta=True)";
  static FunctionDef function_def;
};

constexpr size_t SendSchema_VTI64B::max_args;
constexpr size_t SendSchema_VTI64B::max_pos_args;
constexpr char const* SendSchema_VTI64B::signature;
FunctionDef SendSchema_VTI64B::function_def = {
/*name*/"send",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<void>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dst", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"send_meta", /*default_value*/bool(true), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* send(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("send");
  PythonFrameGuard pf;
  static PythonArgParser<functional::SendSchema_VTI64B> parser("send");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Send(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RecvSchema_TI64ShDtDeT {
  using FType = Maybe<one::Tensor> (int64_t src, const Optional<Shape>& shape, const Optional<Symbol<DType>>& dtype, const Optional<Symbol<Device>>& device, const Optional<one::Tensor>& out);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Recv;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Int64 src, Shape shape=None, DataType dtype=None, Device device=None, *, Tensor out=None)";
  static FunctionDef function_def;
};

constexpr size_t RecvSchema_TI64ShDtDeT::max_args;
constexpr size_t RecvSchema_TI64ShDtDeT::max_pos_args;
constexpr char const* RecvSchema_TI64ShDtDeT::signature;
FunctionDef RecvSchema_TI64ShDtDeT::function_def = {
/*name*/"recv",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"src", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"shape", /*default_value*/Optional<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"device", /*default_value*/Optional<Symbol<Device>>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"out", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

PyObject* recv(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("recv");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RecvSchema_TI64ShDtDeT> parser("recv");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Recv(r[0].As<int64_t>(), r[1].As<Optional<Shape>>(), r[2].As<Optional<Symbol<DType>>>(), r[3].As<Optional<Symbol<Device>>>(), r[4].As<Optional<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BatchGatherSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& indices);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BatchGather;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor in, Tensor indices)";
  static FunctionDef function_def;
};

constexpr size_t BatchGatherSchema_TTT::max_args;
constexpr size_t BatchGatherSchema_TTT::max_pos_args;
constexpr char const* BatchGatherSchema_TTT::signature;
FunctionDef BatchGatherSchema_TTT::function_def = {
/*name*/"batch_gather",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"in", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"indices", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* batch_gather(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("batch_gather");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BatchGatherSchema_TTT> parser("batch_gather");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BatchGather(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct CtcGreedyDecoderSchema_TtTTB {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& log_probs, const std::shared_ptr<one::Tensor>& input_lengths, bool merge_repeated);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::CtcGreedyDecoder;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "TensorTuple (Tensor log_probs, Tensor input_lengths, Bool merge_repeated=True)";
  static FunctionDef function_def;
};

constexpr size_t CtcGreedyDecoderSchema_TtTTB::max_args;
constexpr size_t CtcGreedyDecoderSchema_TtTTB::max_pos_args;
constexpr char const* CtcGreedyDecoderSchema_TtTTB::signature;
FunctionDef CtcGreedyDecoderSchema_TtTTB::function_def = {
/*name*/"ctc_greedy_decoder",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"log_probs", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"input_lengths", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"merge_repeated", /*default_value*/bool(true), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* ctc_greedy_decoder(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("ctc_greedy_decoder");
  PythonFrameGuard pf;
  static PythonArgParser<functional::CtcGreedyDecoderSchema_TtTTB> parser("ctc_greedy_decoder");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::CtcGreedyDecoder(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct NmsSchema_TTFI32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, float iou_threshold, int32_t keep_n);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Nms;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Float iou_threshold, Int32 keep_n=-1)";
  static FunctionDef function_def;
};

constexpr size_t NmsSchema_TTFI32::max_args;
constexpr size_t NmsSchema_TTFI32::max_pos_args;
constexpr char const* NmsSchema_TTFI32::signature;
FunctionDef NmsSchema_TTFI32::function_def = {
/*name*/"nms",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"iou_threshold", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"keep_n", /*default_value*/int32_t(-1), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* nms(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("nms");
  PythonFrameGuard pf;
  static PythonArgParser<functional::NmsSchema_TTFI32> parser("nms");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Nms(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<float>(), r[2].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RoiAlignSchema_TTTFI32I32I32B {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& rois, float spatial_scale, int32_t pooled_h, int32_t pooled_w, int32_t sampling_ratio, bool aligned);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::RoiAlign;
  static constexpr size_t max_args = 7;
  static constexpr size_t max_pos_args = 7;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor rois, Float spatial_scale, Int32 pooled_h, Int32 pooled_w, Int32 sampling_ratio, Bool aligned)";
  static FunctionDef function_def;
};

constexpr size_t RoiAlignSchema_TTTFI32I32I32B::max_args;
constexpr size_t RoiAlignSchema_TTTFI32I32I32B::max_pos_args;
constexpr char const* RoiAlignSchema_TTTFI32I32I32B::signature;
FunctionDef RoiAlignSchema_TTTFI32I32I32B::function_def = {
/*name*/"roi_align",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"rois", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"spatial_scale", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"pooled_h", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"pooled_w", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"sampling_ratio", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"aligned", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* roi_align(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("roi_align");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RoiAlignSchema_TTTFI32I32I32B> parser("roi_align");
  ParsedArgs<7> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::RoiAlign(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<float>(), r[3].As<int32_t>(), r[4].As<int32_t>(), r[5].As<int32_t>(), r[6].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MeshgridSchema_TtTtS {
  using FType = Maybe<one::TensorTuple> (const TensorTuple& tensors, const std::string& indexing);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::Meshgrid;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "TensorTuple (TensorTuple tensors, String indexing=\"ij\")";
  static FunctionDef function_def;
};

constexpr size_t MeshgridSchema_TtTtS::max_args;
constexpr size_t MeshgridSchema_TtTtS::max_pos_args;
constexpr char const* MeshgridSchema_TtTtS::signature;
FunctionDef MeshgridSchema_TtTtS::function_def = {
/*name*/"meshgrid",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"tensors", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"indexing", /*default_value*/std::string("ij"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* meshgrid(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("meshgrid");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MeshgridSchema_TtTtS> parser("meshgrid");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Meshgrid(r[0].As<TensorTuple>(), r[1].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct IndexSelectSchema_TTI64T {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int64_t dim, const std::shared_ptr<one::Tensor>& index);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::IndexSelect;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Int64 dim, Tensor index)";
  static FunctionDef function_def;
};

constexpr size_t IndexSelectSchema_TTI64T::max_args;
constexpr size_t IndexSelectSchema_TTI64T::max_pos_args;
constexpr char const* IndexSelectSchema_TTI64T::signature;
FunctionDef IndexSelectSchema_TTI64T::function_def = {
/*name*/"index_select",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"index", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* index_select(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("index_select");
  PythonFrameGuard pf;
  static PythonArgParser<functional::IndexSelectSchema_TTI64T> parser("index_select");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::IndexSelect(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>(), r[2].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct DecodeOneRecSchema_TTSDtShBShSh {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::string& key, const Symbol<DType>& dtype, const Shape& shape, bool is_dynamic, const Optional<Shape>& reshape, const Optional<Shape>& batch_padding);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::DecodeOneRec;
  static constexpr size_t max_args = 7;
  static constexpr size_t max_pos_args = 7;
  static constexpr char const* signature = "Tensor (Tensor input, String key, DataType dtype, Shape shape, Bool is_dynamic=False, Shape reshape=None, Shape batch_padding=None)";
  static FunctionDef function_def;
};

constexpr size_t DecodeOneRecSchema_TTSDtShBShSh::max_args;
constexpr size_t DecodeOneRecSchema_TTSDtShBShSh::max_pos_args;
constexpr char const* DecodeOneRecSchema_TTSDtShBShSh::signature;
FunctionDef DecodeOneRecSchema_TTSDtShBShSh::function_def = {
/*name*/"decode_onerec",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"key", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*value_type*/ValueTypeOf<Symbol<DType>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"shape", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"is_dynamic", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"reshape", /*default_value*/Optional<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"batch_padding", /*default_value*/Optional<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* decode_onerec(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("decode_onerec");
  PythonFrameGuard pf;
  static PythonArgParser<functional::DecodeOneRecSchema_TTSDtShBShSh> parser("decode_onerec");
  ParsedArgs<7> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::DecodeOneRec(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::string>(), r[2].As<Symbol<DType>>(), r[3].As<Shape>(), r[4].As<bool>(), r[5].As<Optional<Shape>>(), r[6].As<Optional<Shape>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct DotSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& other);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Dot;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor other)";
  static FunctionDef function_def;
};

constexpr size_t DotSchema_TTT::max_args;
constexpr size_t DotSchema_TTT::max_pos_args;
constexpr char const* DotSchema_TTT::signature;
FunctionDef DotSchema_TTT::function_def = {
/*name*/"dot",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"other", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* dot(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("dot");
  PythonFrameGuard pf;
  static PythonArgParser<functional::DotSchema_TTT> parser("dot");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Dot(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct FusedDotFeatureInteractionSchema_TTtTBI32S {
  using FType = Maybe<one::Tensor> (const TensorTuple& features, const Optional<one::Tensor>& output_concat, bool self_interaction, int32_t output_padding, const std::string& pooling);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::FusedDotFeatureInteraction;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Tensor (TensorTuple features, Tensor output_concat=None, Bool self_interaction=False, Int32 output_padding=0, String pooling=\"none\")";
  static FunctionDef function_def;
};

constexpr size_t FusedDotFeatureInteractionSchema_TTtTBI32S::max_args;
constexpr size_t FusedDotFeatureInteractionSchema_TTtTBI32S::max_pos_args;
constexpr char const* FusedDotFeatureInteractionSchema_TTtTBI32S::signature;
FunctionDef FusedDotFeatureInteractionSchema_TTtTBI32S::function_def = {
/*name*/"fused_dot_feature_interaction",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"features", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"output_concat", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"self_interaction", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"output_padding", /*default_value*/int32_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"pooling", /*default_value*/std::string("none"), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* fused_dot_feature_interaction(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("fused_dot_feature_interaction");
  PythonFrameGuard pf;
  static PythonArgParser<functional::FusedDotFeatureInteractionSchema_TTtTBI32S> parser("fused_dot_feature_interaction");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::FusedDotFeatureInteraction(r[0].As<TensorTuple>(), r[1].As<Optional<one::Tensor>>(), r[2].As<bool>(), r[3].As<int32_t>(), r[4].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct FusedCrossFeatureInteractionSchema_TTTTTS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& x_0, const std::shared_ptr<one::Tensor>& bias, const std::string& interaction_mode);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::FusedCrossFeatureInteraction;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Tensor (Tensor x, Tensor weight, Tensor x_0, Tensor bias, String interaction_mode)";
  static FunctionDef function_def;
};

constexpr size_t FusedCrossFeatureInteractionSchema_TTTTTS::max_args;
constexpr size_t FusedCrossFeatureInteractionSchema_TTTTTS::max_pos_args;
constexpr char const* FusedCrossFeatureInteractionSchema_TTTTTS::signature;
FunctionDef FusedCrossFeatureInteractionSchema_TTTTTS::function_def = {
/*name*/"fused_cross_feature_interaction",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"x_0", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bias", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"interaction_mode", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* fused_cross_feature_interaction(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("fused_cross_feature_interaction");
  PythonFrameGuard pf;
  static PythonArgParser<functional::FusedCrossFeatureInteractionSchema_TTTTTS> parser("fused_cross_feature_interaction");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::FusedCrossFeatureInteraction(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<std::shared_ptr<one::Tensor>>(), r[4].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TensorBufferToTensorSchema_TTShDt {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Shape& instance_shape, const Symbol<DType>& dtype);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::TensorBufferToTensor;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Shape instance_shape, DataType dtype)";
  static FunctionDef function_def;
};

constexpr size_t TensorBufferToTensorSchema_TTShDt::max_args;
constexpr size_t TensorBufferToTensorSchema_TTShDt::max_pos_args;
constexpr char const* TensorBufferToTensorSchema_TTShDt::signature;
FunctionDef TensorBufferToTensorSchema_TTShDt::function_def = {
/*name*/"tensor_buffer_to_tensor",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"instance_shape", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*value_type*/ValueTypeOf<Symbol<DType>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* tensor_buffer_to_tensor(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("tensor_buffer_to_tensor");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TensorBufferToTensorSchema_TTShDt> parser("tensor_buffer_to_tensor");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::TensorBufferToTensor(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Shape>(), r[2].As<Symbol<DType>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TensorToTensorBufferSchema_TTI32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int32_t instance_dims);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::TensorToTensorBuffer;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Int32 instance_dims)";
  static FunctionDef function_def;
};

constexpr size_t TensorToTensorBufferSchema_TTI32::max_args;
constexpr size_t TensorToTensorBufferSchema_TTI32::max_pos_args;
constexpr char const* TensorToTensorBufferSchema_TTI32::signature;
FunctionDef TensorToTensorBufferSchema_TTI32::function_def = {
/*name*/"tensor_to_tensor_buffer",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"instance_dims", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* tensor_to_tensor_buffer(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("tensor_to_tensor_buffer");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TensorToTensorBufferSchema_TTI32> parser("tensor_to_tensor_buffer");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::TensorToTensorBuffer(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct GenTensorBufferSchema_TShShlFlDtB {
  using FType = Maybe<one::Tensor> (const Shape& shape, const std::vector<Shape>& shape_list, const std::vector<float>& value_list, const Symbol<DType>& data_type, bool dynamic_out);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GenTensorBuffer;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Tensor (Shape shape, ShapeList shape_list, FloatList value_list, DataType data_type, Bool dynamic_out)";
  static FunctionDef function_def;
};

constexpr size_t GenTensorBufferSchema_TShShlFlDtB::max_args;
constexpr size_t GenTensorBufferSchema_TShShlFlDtB::max_pos_args;
constexpr char const* GenTensorBufferSchema_TShShlFlDtB::signature;
FunctionDef GenTensorBufferSchema_TShShlFlDtB::function_def = {
/*name*/"gen_tensor_buffer",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"shape", /*value_type*/ValueTypeOf<Shape>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"shape_list", /*value_type*/ValueTypeOf<std::vector<Shape>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"value_list", /*value_type*/ValueTypeOf<std::vector<float>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"data_type", /*value_type*/ValueTypeOf<Symbol<DType>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dynamic_out", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* gen_tensor_buffer(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("gen_tensor_buffer");
  PythonFrameGuard pf;
  static PythonArgParser<functional::GenTensorBufferSchema_TShShlFlDtB> parser("gen_tensor_buffer");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::GenTensorBuffer(r[0].As<Shape>(), r[1].As<std::vector<Shape>>(), r[2].As<std::vector<float>>(), r[3].As<Symbol<DType>>(), r[4].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TopKSchema_TTI32B {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int32_t k, bool sorted);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::TopK;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Int32 k, Bool sorted=True)";
  static FunctionDef function_def;
};

constexpr size_t TopKSchema_TTI32B::max_args;
constexpr size_t TopKSchema_TTI32B::max_pos_args;
constexpr char const* TopKSchema_TTI32B::signature;
FunctionDef TopKSchema_TTI32B::function_def = {
/*name*/"top_k",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"k", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"sorted", /*default_value*/bool(true), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* top_k(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("top_k");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TopKSchema_TTI32B> parser("top_k");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::TopK(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct InTopKSchema_TTTI32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& targets, const std::shared_ptr<one::Tensor>& predictions, int32_t k);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::InTopK;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor targets, Tensor predictions, Int32 k)";
  static FunctionDef function_def;
};

constexpr size_t InTopKSchema_TTTI32::max_args;
constexpr size_t InTopKSchema_TTTI32::max_pos_args;
constexpr char const* InTopKSchema_TTTI32::signature;
FunctionDef InTopKSchema_TTTI32::function_def = {
/*name*/"in_top_k",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"targets", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"predictions", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"k", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* in_top_k(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("in_top_k");
  PythonFrameGuard pf;
  static PythonArgParser<functional::InTopKSchema_TTTI32> parser("in_top_k");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::InTopK(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct CumsumSchema_TTI64Dt {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int64_t dim, const Optional<Symbol<DType>>& dtype);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Cumsum;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Int64 dim, *, DataType dtype=None)";
  static FunctionDef function_def;
};

constexpr size_t CumsumSchema_TTI64Dt::max_args;
constexpr size_t CumsumSchema_TTI64Dt::max_pos_args;
constexpr char const* CumsumSchema_TTI64Dt::signature;
FunctionDef CumsumSchema_TTI64Dt::function_def = {
/*name*/"cumsum",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

PyObject* cumsum(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("cumsum");
  PythonFrameGuard pf;
  static PythonArgParser<functional::CumsumSchema_TTI64Dt> parser("cumsum");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Cumsum(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>(), r[2].As<Optional<Symbol<DType>>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct CumprodSchema_TTI64Dt {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int64_t dim, const Optional<Symbol<DType>>& dtype);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Cumprod;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor input, Int64 dim, *, DataType dtype=None)";
  static FunctionDef function_def;
};

constexpr size_t CumprodSchema_TTI64Dt::max_args;
constexpr size_t CumprodSchema_TTI64Dt::max_pos_args;
constexpr char const* CumprodSchema_TTI64Dt::signature;
FunctionDef CumprodSchema_TTI64Dt::function_def = {
/*name*/"cumprod",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dim", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*default_value*/Optional<Symbol<DType>>(), /*size*/0, /*keyword_only*/true, /*optional*/true)
}
};

PyObject* cumprod(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("cumprod");
  PythonFrameGuard pf;
  static PythonArgParser<functional::CumprodSchema_TTI64Dt> parser("cumprod");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Cumprod(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>(), r[2].As<Optional<Symbol<DType>>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct OneEmbeddingIdShuffleSchema_TtTTI32S {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& ids, const Optional<one::Tensor>& table_ids, int32_t num_tables, const std::string& embedding_name);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::OneEmbeddingIdShuffle;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "TensorTuple (Tensor ids, Tensor table_ids=None, Int32 num_tables=1, String embedding_name)";
  static FunctionDef function_def;
};

constexpr size_t OneEmbeddingIdShuffleSchema_TtTTI32S::max_args;
constexpr size_t OneEmbeddingIdShuffleSchema_TtTTI32S::max_pos_args;
constexpr char const* OneEmbeddingIdShuffleSchema_TtTTI32S::signature;
FunctionDef OneEmbeddingIdShuffleSchema_TtTTI32S::function_def = {
/*name*/"one_embedding_id_shuffle",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"ids", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"table_ids", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"num_tables", /*default_value*/int32_t(1), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_name", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* one_embedding_id_shuffle(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("one_embedding_id_shuffle");
  PythonFrameGuard pf;
  static PythonArgParser<functional::OneEmbeddingIdShuffleSchema_TtTTI32S> parser("one_embedding_id_shuffle");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::OneEmbeddingIdShuffle(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<one::Tensor>>(), r[2].As<int32_t>(), r[3].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct OneEmbeddingEmbeddingShuffleSchema_TTTTTS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& cur_rank_embeddings, const std::shared_ptr<one::Tensor>& num_unique_matrix, const std::shared_ptr<one::Tensor>& cur_rank_inverse_indices, const std::shared_ptr<one::Tensor>& inverse_unique_partition_indices, const std::string& embedding_name);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::OneEmbeddingEmbeddingShuffle;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Tensor (Tensor cur_rank_embeddings, Tensor num_unique_matrix, Tensor cur_rank_inverse_indices, Tensor inverse_unique_partition_indices, String embedding_name)";
  static FunctionDef function_def;
};

constexpr size_t OneEmbeddingEmbeddingShuffleSchema_TTTTTS::max_args;
constexpr size_t OneEmbeddingEmbeddingShuffleSchema_TTTTTS::max_pos_args;
constexpr char const* OneEmbeddingEmbeddingShuffleSchema_TTTTTS::signature;
FunctionDef OneEmbeddingEmbeddingShuffleSchema_TTTTTS::function_def = {
/*name*/"one_embedding_embedding_shuffle",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"cur_rank_embeddings", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"num_unique_matrix", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"cur_rank_inverse_indices", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"inverse_unique_partition_indices", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_name", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* one_embedding_embedding_shuffle(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("one_embedding_embedding_shuffle");
  PythonFrameGuard pf;
  static PythonArgParser<functional::OneEmbeddingEmbeddingShuffleSchema_TTTTTS> parser("one_embedding_embedding_shuffle");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::OneEmbeddingEmbeddingShuffle(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<std::shared_ptr<one::Tensor>>(), r[4].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct OneEmbeddingEmbeddingGradientShuffleSchema_TTTTTS {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& embedding_grad, const std::shared_ptr<one::Tensor>& num_unique_matrix, const std::shared_ptr<one::Tensor>& cur_rank_inverse_indices, const std::shared_ptr<one::Tensor>& inverse_unique_partition_indices, const std::string& embedding_name);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::OneEmbeddingEmbeddingGradientShuffle;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Tensor (Tensor embedding_grad, Tensor num_unique_matrix, Tensor cur_rank_inverse_indices, Tensor inverse_unique_partition_indices, String embedding_name)";
  static FunctionDef function_def;
};

constexpr size_t OneEmbeddingEmbeddingGradientShuffleSchema_TTTTTS::max_args;
constexpr size_t OneEmbeddingEmbeddingGradientShuffleSchema_TTTTTS::max_pos_args;
constexpr char const* OneEmbeddingEmbeddingGradientShuffleSchema_TTTTTS::signature;
FunctionDef OneEmbeddingEmbeddingGradientShuffleSchema_TTTTTS::function_def = {
/*name*/"one_embedding_embedding_gradient_shuffle",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"embedding_grad", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"num_unique_matrix", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"cur_rank_inverse_indices", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"inverse_unique_partition_indices", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_name", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* one_embedding_embedding_gradient_shuffle(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("one_embedding_embedding_gradient_shuffle");
  PythonFrameGuard pf;
  static PythonArgParser<functional::OneEmbeddingEmbeddingGradientShuffleSchema_TTTTTS> parser("one_embedding_embedding_gradient_shuffle");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::OneEmbeddingEmbeddingGradientShuffle(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<std::shared_ptr<one::Tensor>>(), r[4].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct OneEmbeddingLookupSchema_TTTTDtDtI64I64SSSI64 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& num_unique_ids, const std::shared_ptr<one::Tensor>& unique_ids, const std::shared_ptr<one::Tensor>& table_ids, const Symbol<DType>& dtype, const Symbol<DType>& embedding_dtype, int64_t line_size, int64_t embedding_size, const std::string& embedding_name, const std::string& embedding_tables, const std::string& state_initializer, int64_t seed);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::OneEmbeddingLookup;
  static constexpr size_t max_args = 11;
  static constexpr size_t max_pos_args = 11;
  static constexpr char const* signature = "Tensor (Tensor num_unique_ids, Tensor unique_ids, Tensor table_ids, DataType dtype, DataType embedding_dtype, Int64 line_size, Int64 embedding_size, String embedding_name, String embedding_tables, String state_initializer, Int64 seed=0)";
  static FunctionDef function_def;
};

constexpr size_t OneEmbeddingLookupSchema_TTTTDtDtI64I64SSSI64::max_args;
constexpr size_t OneEmbeddingLookupSchema_TTTTDtDtI64I64SSSI64::max_pos_args;
constexpr char const* OneEmbeddingLookupSchema_TTTTDtDtI64I64SSSI64::signature;
FunctionDef OneEmbeddingLookupSchema_TTTTDtDtI64I64SSSI64::function_def = {
/*name*/"one_embedding_lookup",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"num_unique_ids", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"unique_ids", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"table_ids", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dtype", /*value_type*/ValueTypeOf<Symbol<DType>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_dtype", /*value_type*/ValueTypeOf<Symbol<DType>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"line_size", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_size", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_name", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_tables", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"state_initializer", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"seed", /*default_value*/int64_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* one_embedding_lookup(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("one_embedding_lookup");
  PythonFrameGuard pf;
  static PythonArgParser<functional::OneEmbeddingLookupSchema_TTTTDtDtI64I64SSSI64> parser("one_embedding_lookup");
  ParsedArgs<11> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::OneEmbeddingLookup(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<Symbol<DType>>(), r[4].As<Symbol<DType>>(), r[5].As<int64_t>(), r[6].As<int64_t>(), r[7].As<std::string>(), r[8].As<std::string>(), r[9].As<std::string>(), r[10].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct OneEmbeddingFusedLookupSchema_TTTTDtSI64I64BI32SI64I64 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& shadow, const std::shared_ptr<one::Tensor>& ids, const Optional<one::Tensor>& table_ids, const Symbol<DType>& dtype, const std::string& embedding_name, int64_t line_size, int64_t embedding_size, bool is_full_cache, int32_t num_tables, const std::string& embedding_tables, const Optional<int64_t>& padding_idx, int64_t seed);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::OneEmbeddingFusedLookup;
  static constexpr size_t max_args = 12;
  static constexpr size_t max_pos_args = 12;
  static constexpr char const* signature = "Tensor (Tensor shadow, Tensor ids, Tensor table_ids=None, DataType dtype, String embedding_name, Int64 line_size, Int64 embedding_size, Bool is_full_cache, Int32 num_tables, String embedding_tables, Int64 padding_idx=None, Int64 seed=0)";
  static FunctionDef function_def;
};

constexpr size_t OneEmbeddingFusedLookupSchema_TTTTDtSI64I64BI32SI64I64::max_args;
constexpr size_t OneEmbeddingFusedLookupSchema_TTTTDtSI64I64BI32SI64I64::max_pos_args;
constexpr char const* OneEmbeddingFusedLookupSchema_TTTTDtSI64I64BI32SI64I64::signature;
FunctionDef OneEmbeddingFusedLookupSchema_TTTTDtSI64I64BI32SI64I64::function_def = {
/*name*/"one_embedding_fused_lookup",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"shadow", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"ids", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"table_ids", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"dtype", /*value_type*/ValueTypeOf<Symbol<DType>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_name", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"line_size", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_size", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"is_full_cache", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"num_tables", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_tables", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"padding_idx", /*default_value*/Optional<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"seed", /*default_value*/int64_t(0), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* one_embedding_fused_lookup(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("one_embedding_fused_lookup");
  PythonFrameGuard pf;
  static PythonArgParser<functional::OneEmbeddingFusedLookupSchema_TTTTDtSI64I64BI32SI64I64> parser("one_embedding_fused_lookup");
  ParsedArgs<12> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::OneEmbeddingFusedLookup(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<Optional<one::Tensor>>(), r[3].As<Symbol<DType>>(), r[4].As<std::string>(), r[5].As<int64_t>(), r[6].As<int64_t>(), r[7].As<bool>(), r[8].As<int32_t>(), r[9].As<std::string>(), r[10].As<Optional<int64_t>>(), r[11].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct OneEmbeddingFusedLookupGradSchema_VTTSI64I64 {
  using FType = Maybe<void> (const std::shared_ptr<one::Tensor>& ids, const std::shared_ptr<one::Tensor>& embedding_grad, const std::string& embedding_name, int64_t line_size, int64_t embedding_size);
  using R = Maybe<void>;

  static constexpr FType* func = &functional::OneEmbeddingFusedLookupGrad;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Void (Tensor ids, Tensor embedding_grad, String embedding_name, Int64 line_size, Int64 embedding_size)";
  static FunctionDef function_def;
};

constexpr size_t OneEmbeddingFusedLookupGradSchema_VTTSI64I64::max_args;
constexpr size_t OneEmbeddingFusedLookupGradSchema_VTTSI64I64::max_pos_args;
constexpr char const* OneEmbeddingFusedLookupGradSchema_VTTSI64I64::signature;
FunctionDef OneEmbeddingFusedLookupGradSchema_VTTSI64I64::function_def = {
/*name*/"one_embedding_fused_lookup_grad",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<void>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"ids", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_grad", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_name", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"line_size", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_size", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* one_embedding_fused_lookup_grad(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("one_embedding_fused_lookup_grad");
  PythonFrameGuard pf;
  static PythonArgParser<functional::OneEmbeddingFusedLookupGradSchema_VTTSI64I64> parser("one_embedding_fused_lookup_grad");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::OneEmbeddingFusedLookupGrad(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::string>(), r[3].As<int64_t>(), r[4].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct OneEmbeddingUniqueKeyValuePairSchema_TtTTI32S {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& keys, const Optional<one::Tensor>& values, int32_t num_tables, const std::string& embedding_name);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::OneEmbeddingUniqueKeyValuePair;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "TensorTuple (Tensor keys, Tensor values=None, Int32 num_tables, String embedding_name)";
  static FunctionDef function_def;
};

constexpr size_t OneEmbeddingUniqueKeyValuePairSchema_TtTTI32S::max_args;
constexpr size_t OneEmbeddingUniqueKeyValuePairSchema_TtTTI32S::max_pos_args;
constexpr char const* OneEmbeddingUniqueKeyValuePairSchema_TtTTI32S::signature;
FunctionDef OneEmbeddingUniqueKeyValuePairSchema_TtTTI32S::function_def = {
/*name*/"one_embedding_unique_key_value_pair",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"keys", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"values", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"num_tables", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_name", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* one_embedding_unique_key_value_pair(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("one_embedding_unique_key_value_pair");
  PythonFrameGuard pf;
  static PythonArgParser<functional::OneEmbeddingUniqueKeyValuePairSchema_TtTTI32S> parser("one_embedding_unique_key_value_pair");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::OneEmbeddingUniqueKeyValuePair(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<one::Tensor>>(), r[2].As<int32_t>(), r[3].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct OneEmbeddingEmbeddingPutSchema_VTTTSI64 {
  using FType = Maybe<void> (const std::shared_ptr<one::Tensor>& num_unique_ids, const std::shared_ptr<one::Tensor>& unique_ids, const std::shared_ptr<one::Tensor>& unique_embeddings, const std::string& embedding_name, int64_t line_size);
  using R = Maybe<void>;

  static constexpr FType* func = &functional::OneEmbeddingEmbeddingPut;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Void (Tensor num_unique_ids, Tensor unique_ids, Tensor unique_embeddings, String embedding_name, Int64 line_size)";
  static FunctionDef function_def;
};

constexpr size_t OneEmbeddingEmbeddingPutSchema_VTTTSI64::max_args;
constexpr size_t OneEmbeddingEmbeddingPutSchema_VTTTSI64::max_pos_args;
constexpr char const* OneEmbeddingEmbeddingPutSchema_VTTTSI64::signature;
FunctionDef OneEmbeddingEmbeddingPutSchema_VTTTSI64::function_def = {
/*name*/"one_embedding_embedding_put",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<void>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"num_unique_ids", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"unique_ids", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"unique_embeddings", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_name", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"line_size", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* one_embedding_embedding_put(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("one_embedding_embedding_put");
  PythonFrameGuard pf;
  static PythonArgParser<functional::OneEmbeddingEmbeddingPutSchema_VTTTSI64> parser("one_embedding_embedding_put");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::OneEmbeddingEmbeddingPut(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<std::string>(), r[4].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct OneEmbeddingSgdUpdateSchema_TTTTTTTFDFFI64I64S {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& num_unique_ids, const std::shared_ptr<one::Tensor>& unique_embeddings, const std::shared_ptr<one::Tensor>& embedding_grad, const Optional<one::Tensor>& learning_rate, const Optional<one::Tensor>& down_scale_by_tensor, const Optional<one::Tensor>& skip_if, float learning_rate_val, double scale, float weight_decay, float momentum, int64_t line_size, int64_t embedding_size, const std::string& embedding_name);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::OneEmbeddingSgdUpdate;
  static constexpr size_t max_args = 13;
  static constexpr size_t max_pos_args = 13;
  static constexpr char const* signature = "Tensor (Tensor num_unique_ids, Tensor unique_embeddings, Tensor embedding_grad, Tensor learning_rate=None, Tensor down_scale_by_tensor=None, Tensor skip_if=None, Float learning_rate_val, Double scale, Float weight_decay, Float momentum, Int64 line_size, Int64 embedding_size, String embedding_name)";
  static FunctionDef function_def;
};

constexpr size_t OneEmbeddingSgdUpdateSchema_TTTTTTTFDFFI64I64S::max_args;
constexpr size_t OneEmbeddingSgdUpdateSchema_TTTTTTTFDFFI64I64S::max_pos_args;
constexpr char const* OneEmbeddingSgdUpdateSchema_TTTTTTTFDFFI64I64S::signature;
FunctionDef OneEmbeddingSgdUpdateSchema_TTTTTTTFDFFI64I64S::function_def = {
/*name*/"one_embedding_sgd_update",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"num_unique_ids", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"unique_embeddings", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_grad", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"learning_rate", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"down_scale_by_tensor", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"skip_if", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"learning_rate_val", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"scale", /*value_type*/ValueTypeOf<double>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight_decay", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"momentum", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"line_size", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_size", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_name", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* one_embedding_sgd_update(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("one_embedding_sgd_update");
  PythonFrameGuard pf;
  static PythonArgParser<functional::OneEmbeddingSgdUpdateSchema_TTTTTTTFDFFI64I64S> parser("one_embedding_sgd_update");
  ParsedArgs<13> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::OneEmbeddingSgdUpdate(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<Optional<one::Tensor>>(), r[4].As<Optional<one::Tensor>>(), r[5].As<Optional<one::Tensor>>(), r[6].As<float>(), r[7].As<double>(), r[8].As<float>(), r[9].As<float>(), r[10].As<int64_t>(), r[11].As<int64_t>(), r[12].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct OneEmbeddingAdamUpdateSchema_TTTTTTTTTFDFFFFFFBI64I64S {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& num_unique_ids, const std::shared_ptr<one::Tensor>& unique_embeddings, const std::shared_ptr<one::Tensor>& embedding_grad, const Optional<one::Tensor>& learning_rate, const Optional<one::Tensor>& down_scale_by_tensor, const Optional<one::Tensor>& skip_if, const Optional<one::Tensor>& bias_correction1, const Optional<one::Tensor>& bias_correction2, float learning_rate_val, double scale, float weight_decay, float beta1, float beta2, float bias_correction1_val, float bias_correction2_val, float epsilon, bool do_bias_correction, int64_t line_size, int64_t embedding_size, const std::string& embedding_name);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::OneEmbeddingAdamUpdate;
  static constexpr size_t max_args = 20;
  static constexpr size_t max_pos_args = 20;
  static constexpr char const* signature = "Tensor (Tensor num_unique_ids, Tensor unique_embeddings, Tensor embedding_grad, Tensor learning_rate=None, Tensor down_scale_by_tensor=None, Tensor skip_if=None, Tensor bias_correction1=None, Tensor bias_correction2=None, Float learning_rate_val, Double scale, Float weight_decay, Float beta1, Float beta2, Float bias_correction1_val, Float bias_correction2_val, Float epsilon, Bool do_bias_correction, Int64 line_size, Int64 embedding_size, String embedding_name)";
  static FunctionDef function_def;
};

constexpr size_t OneEmbeddingAdamUpdateSchema_TTTTTTTTTFDFFFFFFBI64I64S::max_args;
constexpr size_t OneEmbeddingAdamUpdateSchema_TTTTTTTTTFDFFFFFFBI64I64S::max_pos_args;
constexpr char const* OneEmbeddingAdamUpdateSchema_TTTTTTTTTFDFFFFFFBI64I64S::signature;
FunctionDef OneEmbeddingAdamUpdateSchema_TTTTTTTTTFDFFFFFFBI64I64S::function_def = {
/*name*/"one_embedding_adam_update",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"num_unique_ids", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"unique_embeddings", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_grad", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"learning_rate", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"down_scale_by_tensor", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"skip_if", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"bias_correction1", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"bias_correction2", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"learning_rate_val", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"scale", /*value_type*/ValueTypeOf<double>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight_decay", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"beta1", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"beta2", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bias_correction1_val", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bias_correction2_val", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"epsilon", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"do_bias_correction", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"line_size", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_size", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_name", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* one_embedding_adam_update(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("one_embedding_adam_update");
  PythonFrameGuard pf;
  static PythonArgParser<functional::OneEmbeddingAdamUpdateSchema_TTTTTTTTTFDFFFFFFBI64I64S> parser("one_embedding_adam_update");
  ParsedArgs<20> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::OneEmbeddingAdamUpdate(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<Optional<one::Tensor>>(), r[4].As<Optional<one::Tensor>>(), r[5].As<Optional<one::Tensor>>(), r[6].As<Optional<one::Tensor>>(), r[7].As<Optional<one::Tensor>>(), r[8].As<float>(), r[9].As<double>(), r[10].As<float>(), r[11].As<float>(), r[12].As<float>(), r[13].As<float>(), r[14].As<float>(), r[15].As<float>(), r[16].As<bool>(), r[17].As<int64_t>(), r[18].As<int64_t>(), r[19].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct OneEmbeddingAdagradUpdateSchema_TTTTTTTTI64FDFFFI64I64S {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& num_unique_ids, const std::shared_ptr<one::Tensor>& unique_embeddings, const std::shared_ptr<one::Tensor>& embedding_grad, const Optional<one::Tensor>& learning_rate, const Optional<one::Tensor>& down_scale_by_tensor, const Optional<one::Tensor>& skip_if, const Optional<one::Tensor>& train_step, int64_t train_step_val, float learning_rate_val, double scale, float weight_decay, float lr_decay, float epsilon, int64_t line_size, int64_t embedding_size, const std::string& embedding_name);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::OneEmbeddingAdagradUpdate;
  static constexpr size_t max_args = 16;
  static constexpr size_t max_pos_args = 16;
  static constexpr char const* signature = "Tensor (Tensor num_unique_ids, Tensor unique_embeddings, Tensor embedding_grad, Tensor learning_rate=None, Tensor down_scale_by_tensor=None, Tensor skip_if=None, Tensor train_step=None, Int64 train_step_val, Float learning_rate_val, Double scale, Float weight_decay, Float lr_decay, Float epsilon, Int64 line_size, Int64 embedding_size, String embedding_name)";
  static FunctionDef function_def;
};

constexpr size_t OneEmbeddingAdagradUpdateSchema_TTTTTTTTI64FDFFFI64I64S::max_args;
constexpr size_t OneEmbeddingAdagradUpdateSchema_TTTTTTTTI64FDFFFI64I64S::max_pos_args;
constexpr char const* OneEmbeddingAdagradUpdateSchema_TTTTTTTTI64FDFFFI64I64S::signature;
FunctionDef OneEmbeddingAdagradUpdateSchema_TTTTTTTTI64FDFFFI64I64S::function_def = {
/*name*/"one_embedding_adagrad_update",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"num_unique_ids", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"unique_embeddings", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_grad", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"learning_rate", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"down_scale_by_tensor", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"skip_if", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"train_step", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"train_step_val", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"learning_rate_val", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"scale", /*value_type*/ValueTypeOf<double>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight_decay", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"lr_decay", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"epsilon", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"line_size", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_size", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_name", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* one_embedding_adagrad_update(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("one_embedding_adagrad_update");
  PythonFrameGuard pf;
  static PythonArgParser<functional::OneEmbeddingAdagradUpdateSchema_TTTTTTTTI64FDFFFI64I64S> parser("one_embedding_adagrad_update");
  ParsedArgs<16> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::OneEmbeddingAdagradUpdate(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<Optional<one::Tensor>>(), r[4].As<Optional<one::Tensor>>(), r[5].As<Optional<one::Tensor>>(), r[6].As<Optional<one::Tensor>>(), r[7].As<int64_t>(), r[8].As<float>(), r[9].As<double>(), r[10].As<float>(), r[11].As<float>(), r[12].As<float>(), r[13].As<int64_t>(), r[14].As<int64_t>(), r[15].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct OneEmbeddingFtrlUpdateSchema_TTTTTTTFDFFFFFI64I64S {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& num_unique_ids, const std::shared_ptr<one::Tensor>& unique_embeddings, const std::shared_ptr<one::Tensor>& embedding_grad, const Optional<one::Tensor>& learning_rate, const Optional<one::Tensor>& down_scale_by_tensor, const Optional<one::Tensor>& skip_if, float learning_rate_val, double scale, float weight_decay, float lr_power, float lambda1, float lambda2, float beta, int64_t line_size, int64_t embedding_size, const std::string& embedding_name);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::OneEmbeddingFtrlUpdate;
  static constexpr size_t max_args = 16;
  static constexpr size_t max_pos_args = 16;
  static constexpr char const* signature = "Tensor (Tensor num_unique_ids, Tensor unique_embeddings, Tensor embedding_grad, Tensor learning_rate=None, Tensor down_scale_by_tensor=None, Tensor skip_if=None, Float learning_rate_val, Double scale, Float weight_decay, Float lr_power, Float lambda1, Float lambda2, Float beta, Int64 line_size, Int64 embedding_size, String embedding_name)";
  static FunctionDef function_def;
};

constexpr size_t OneEmbeddingFtrlUpdateSchema_TTTTTTTFDFFFFFI64I64S::max_args;
constexpr size_t OneEmbeddingFtrlUpdateSchema_TTTTTTTFDFFFFFI64I64S::max_pos_args;
constexpr char const* OneEmbeddingFtrlUpdateSchema_TTTTTTTFDFFFFFI64I64S::signature;
FunctionDef OneEmbeddingFtrlUpdateSchema_TTTTTTTFDFFFFFI64I64S::function_def = {
/*name*/"one_embedding_ftrl_update",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"num_unique_ids", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"unique_embeddings", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_grad", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"learning_rate", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"down_scale_by_tensor", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"skip_if", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"learning_rate_val", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"scale", /*value_type*/ValueTypeOf<double>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight_decay", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"lr_power", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"lambda1", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"lambda2", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"beta", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"line_size", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_size", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"embedding_name", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* one_embedding_ftrl_update(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("one_embedding_ftrl_update");
  PythonFrameGuard pf;
  static PythonArgParser<functional::OneEmbeddingFtrlUpdateSchema_TTTTTTTFDFFFFFI64I64S> parser("one_embedding_ftrl_update");
  ParsedArgs<16> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::OneEmbeddingFtrlUpdate(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<Optional<one::Tensor>>(), r[4].As<Optional<one::Tensor>>(), r[5].As<Optional<one::Tensor>>(), r[6].As<float>(), r[7].As<double>(), r[8].As<float>(), r[9].As<float>(), r[10].As<float>(), r[11].As<float>(), r[12].As<float>(), r[13].As<int64_t>(), r[14].As<int64_t>(), r[15].As<std::string>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct EinSumSchema_TSTt {
  using FType = Maybe<one::Tensor> (const std::string& equation, const TensorTuple& operands);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::EinSum;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (String equation, TensorTuple operands)";
  static FunctionDef function_def;
};

constexpr size_t EinSumSchema_TSTt::max_args;
constexpr size_t EinSumSchema_TSTt::max_pos_args;
constexpr char const* EinSumSchema_TSTt::signature;
FunctionDef EinSumSchema_TSTt::function_def = {
/*name*/"einsum",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"equation", /*value_type*/ValueTypeOf<std::string>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"operands", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* einsum(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("einsum");
  PythonFrameGuard pf;
  static PythonArgParser<functional::EinSumSchema_TSTt> parser("einsum");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::EinSum(r[0].As<std::string>(), r[1].As<TensorTuple>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct PixelShuffleSchema_TTI64I64 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, int64_t h_upscale_factor, int64_t w_upscale_factor);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::PixelShuffle;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Int64 h_upscale_factor, Int64 w_upscale_factor)";
  static FunctionDef function_def;
};

constexpr size_t PixelShuffleSchema_TTI64I64::max_args;
constexpr size_t PixelShuffleSchema_TTI64I64::max_pos_args;
constexpr char const* PixelShuffleSchema_TTI64I64::signature;
FunctionDef PixelShuffleSchema_TTI64I64::function_def = {
/*name*/"pixel_shuffle",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"h_upscale_factor", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"w_upscale_factor", /*value_type*/ValueTypeOf<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* pixel_shuffle(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("pixel_shuffle");
  PythonFrameGuard pf;
  static PythonArgParser<functional::PixelShuffleSchema_TTI64I64> parser("pixel_shuffle");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::PixelShuffle(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int64_t>(), r[2].As<int64_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct IsNanSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::IsNan;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input)";
  static FunctionDef function_def;
};

constexpr size_t IsNanSchema_TT::max_args;
constexpr size_t IsNanSchema_TT::max_pos_args;
constexpr char const* IsNanSchema_TT::signature;
FunctionDef IsNanSchema_TT::function_def = {
/*name*/"isnan",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* isnan(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("isnan");
  PythonFrameGuard pf;
  static PythonArgParser<functional::IsNanSchema_TT> parser("isnan");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::IsNan(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct IsInfSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::IsInf;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input)";
  static FunctionDef function_def;
};

constexpr size_t IsInfSchema_TT::max_args;
constexpr size_t IsInfSchema_TT::max_pos_args;
constexpr char const* IsInfSchema_TT::signature;
FunctionDef IsInfSchema_TT::function_def = {
/*name*/"isinf",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* isinf(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("isinf");
  PythonFrameGuard pf;
  static PythonArgParser<functional::IsInfSchema_TT> parser("isinf");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::IsInf(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct IsFiniteSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::IsFinite;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input)";
  static FunctionDef function_def;
};

constexpr size_t IsFiniteSchema_TT::max_args;
constexpr size_t IsFiniteSchema_TT::max_pos_args;
constexpr char const* IsFiniteSchema_TT::signature;
FunctionDef IsFiniteSchema_TT::function_def = {
/*name*/"isfinite",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* isfinite(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("isfinite");
  PythonFrameGuard pf;
  static PythonArgParser<functional::IsFiniteSchema_TT> parser("isfinite");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::IsFinite(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RocAucScoreSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& label, const std::shared_ptr<one::Tensor>& pred);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::RocAucScore;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor label, Tensor pred)";
  static FunctionDef function_def;
};

constexpr size_t RocAucScoreSchema_TTT::max_args;
constexpr size_t RocAucScoreSchema_TTT::max_pos_args;
constexpr char const* RocAucScoreSchema_TTT::signature;
FunctionDef RocAucScoreSchema_TTT::function_def = {
/*name*/"roc_auc_score",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"label", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"pred", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* roc_auc_score(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("roc_auc_score");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RocAucScoreSchema_TTT> parser("roc_auc_score");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::RocAucScore(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct PinMemorySchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::PinMemory;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input)";
  static FunctionDef function_def;
};

constexpr size_t PinMemorySchema_TT::max_args;
constexpr size_t PinMemorySchema_TT::max_pos_args;
constexpr char const* PinMemorySchema_TT::signature;
FunctionDef PinMemorySchema_TT::function_def = {
/*name*/"pin_memory",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* pin_memory(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("pin_memory");
  PythonFrameGuard pf;
  static PythonArgParser<functional::PinMemorySchema_TT> parser("pin_memory");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::PinMemory(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct FillTensorSchema_TTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& in, const std::shared_ptr<one::Tensor>& value);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::FillTensor;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor in, Tensor value)";
  static FunctionDef function_def;
};

constexpr size_t FillTensorSchema_TTT::max_args;
constexpr size_t FillTensorSchema_TTT::max_pos_args;
constexpr char const* FillTensorSchema_TTT::signature;
FunctionDef FillTensorSchema_TTT::function_def = {
/*name*/"fill_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"in", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"value", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct FillSchema_TTSc {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& in, const Scalar& value);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Fill;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "Tensor (Tensor in, Scalar value)";
  static FunctionDef function_def;
};

constexpr size_t FillSchema_TTSc::max_args;
constexpr size_t FillSchema_TTSc::max_pos_args;
constexpr char const* FillSchema_TTSc::signature;
FunctionDef FillSchema_TTSc::function_def = {
/*name*/"fill_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"in", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"value", /*value_type*/ValueTypeOf<Scalar>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* fill_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("fill_");
  PythonFrameGuard pf;
  static PythonArgParser<functional::FillTensorSchema_TTT, functional::FillSchema_TTSc> parser("fill_");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::FillTensor(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::Fill(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Scalar>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RnnTanhCellSchema_TTTTTTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& hx, const std::shared_ptr<one::Tensor>& w_ih, const std::shared_ptr<one::Tensor>& w_hh, const Optional<one::Tensor>& b_ih, const Optional<one::Tensor>& b_hh);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::RnnTanhCell;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 6;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih=None, Tensor b_hh=None)";
  static FunctionDef function_def;
};

constexpr size_t RnnTanhCellSchema_TTTTTTT::max_args;
constexpr size_t RnnTanhCellSchema_TTTTTTT::max_pos_args;
constexpr char const* RnnTanhCellSchema_TTTTTTT::signature;
FunctionDef RnnTanhCellSchema_TTTTTTT::function_def = {
/*name*/"rnn_tanh_cell",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"hx", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"w_ih", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"w_hh", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"b_ih", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"b_hh", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* rnn_tanh_cell(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("rnn_tanh_cell");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RnnTanhCellSchema_TTTTTTT> parser("rnn_tanh_cell");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::RnnTanhCell(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<std::shared_ptr<one::Tensor>>(), r[4].As<Optional<one::Tensor>>(), r[5].As<Optional<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RnnReluCellSchema_TTTTTTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& hx, const std::shared_ptr<one::Tensor>& w_ih, const std::shared_ptr<one::Tensor>& w_hh, const Optional<one::Tensor>& b_ih, const Optional<one::Tensor>& b_hh);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::RnnReluCell;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 6;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih=None, Tensor b_hh=None)";
  static FunctionDef function_def;
};

constexpr size_t RnnReluCellSchema_TTTTTTT::max_args;
constexpr size_t RnnReluCellSchema_TTTTTTT::max_pos_args;
constexpr char const* RnnReluCellSchema_TTTTTTT::signature;
FunctionDef RnnReluCellSchema_TTTTTTT::function_def = {
/*name*/"rnn_relu_cell",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"hx", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"w_ih", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"w_hh", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"b_ih", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"b_hh", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* rnn_relu_cell(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("rnn_relu_cell");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RnnReluCellSchema_TTTTTTT> parser("rnn_relu_cell");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::RnnReluCell(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<std::shared_ptr<one::Tensor>>(), r[4].As<Optional<one::Tensor>>(), r[5].As<Optional<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct LstmCellSchema_TtTTtTTTT {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, const TensorTuple& hx, const std::shared_ptr<one::Tensor>& w_ih, const std::shared_ptr<one::Tensor>& w_hh, const Optional<one::Tensor>& b_ih, const Optional<one::Tensor>& b_hh);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::LstmCell;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 6;
  static constexpr char const* signature = "TensorTuple (Tensor input, TensorTuple hx, Tensor w_ih, Tensor w_hh, Tensor b_ih=None, Tensor b_hh=None)";
  static FunctionDef function_def;
};

constexpr size_t LstmCellSchema_TtTTtTTTT::max_args;
constexpr size_t LstmCellSchema_TtTTtTTTT::max_pos_args;
constexpr char const* LstmCellSchema_TtTTtTTTT::signature;
FunctionDef LstmCellSchema_TtTTtTTTT::function_def = {
/*name*/"lstm_cell",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"hx", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"w_ih", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"w_hh", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"b_ih", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"b_hh", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* lstm_cell(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("lstm_cell");
  PythonFrameGuard pf;
  static PythonArgParser<functional::LstmCellSchema_TtTTtTTTT> parser("lstm_cell");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::LstmCell(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<TensorTuple>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<std::shared_ptr<one::Tensor>>(), r[4].As<Optional<one::Tensor>>(), r[5].As<Optional<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct GruCellSchema_TTTTTTT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& hx, const std::shared_ptr<one::Tensor>& w_ih, const std::shared_ptr<one::Tensor>& w_hh, const Optional<one::Tensor>& b_ih, const Optional<one::Tensor>& b_hh);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::GruCell;
  static constexpr size_t max_args = 6;
  static constexpr size_t max_pos_args = 6;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih=None, Tensor b_hh=None)";
  static FunctionDef function_def;
};

constexpr size_t GruCellSchema_TTTTTTT::max_args;
constexpr size_t GruCellSchema_TTTTTTT::max_pos_args;
constexpr char const* GruCellSchema_TTTTTTT::signature;
FunctionDef GruCellSchema_TTTTTTT::function_def = {
/*name*/"gru_cell",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"hx", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"w_ih", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"w_hh", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"b_ih", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"b_hh", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* gru_cell(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("gru_cell");
  PythonFrameGuard pf;
  static PythonArgParser<functional::GruCellSchema_TTTTTTT> parser("gru_cell");
  ParsedArgs<6> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::GruCell(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<std::shared_ptr<one::Tensor>>(), r[4].As<Optional<one::Tensor>>(), r[5].As<Optional<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RnnTanhInputSchema_TtTTTtBI32FBBB {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional, bool batch_first);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::RnnTanhInput;
  static constexpr size_t max_args = 9;
  static constexpr size_t max_pos_args = 9;
  static constexpr char const* signature = "TensorTuple (Tensor input, Tensor hx, TensorTuple params, Bool has_biases, Int32 num_layers, Float dropout, Bool train, Bool bidirectional, Bool batch_first)";
  static FunctionDef function_def;
};

constexpr size_t RnnTanhInputSchema_TtTTTtBI32FBBB::max_args;
constexpr size_t RnnTanhInputSchema_TtTTTtBI32FBBB::max_pos_args;
constexpr char const* RnnTanhInputSchema_TtTTTtBI32FBBB::signature;
FunctionDef RnnTanhInputSchema_TtTTTtBI32FBBB::function_def = {
/*name*/"rnn_tanh",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"hx", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"params", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"has_biases", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"num_layers", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dropout", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"train", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bidirectional", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"batch_first", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct RnnTanhDataSchema_TtTTTTtBI32FBB {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& data, const std::shared_ptr<one::Tensor>& batch_sizes, const std::shared_ptr<one::Tensor>& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::RnnTanhData;
  static constexpr size_t max_args = 9;
  static constexpr size_t max_pos_args = 9;
  static constexpr char const* signature = "TensorTuple (Tensor data, Tensor batch_sizes, Tensor hx, TensorTuple params, Bool has_biases, Int32 num_layers, Float dropout, Bool train, Bool bidirectional)";
  static FunctionDef function_def;
};

constexpr size_t RnnTanhDataSchema_TtTTTTtBI32FBB::max_args;
constexpr size_t RnnTanhDataSchema_TtTTTTtBI32FBB::max_pos_args;
constexpr char const* RnnTanhDataSchema_TtTTTTtBI32FBB::signature;
FunctionDef RnnTanhDataSchema_TtTTTTtBI32FBB::function_def = {
/*name*/"rnn_tanh",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"data", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"batch_sizes", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"hx", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"params", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"has_biases", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"num_layers", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dropout", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"train", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bidirectional", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* rnn_tanh(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("rnn_tanh");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RnnTanhInputSchema_TtTTTtBI32FBBB, functional::RnnTanhDataSchema_TtTTTTtBI32FBB> parser("rnn_tanh");
  ParsedArgs<9> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::RnnTanhInput(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<TensorTuple>(), r[3].As<bool>(), r[4].As<int32_t>(), r[5].As<float>(), r[6].As<bool>(), r[7].As<bool>(), r[8].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::RnnTanhData(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<TensorTuple>(), r[4].As<bool>(), r[5].As<int32_t>(), r[6].As<float>(), r[7].As<bool>(), r[8].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct RnnReluInputSchema_TtTTTtBI32FBBB {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional, bool batch_first);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::RnnReluInput;
  static constexpr size_t max_args = 9;
  static constexpr size_t max_pos_args = 9;
  static constexpr char const* signature = "TensorTuple (Tensor input, Tensor hx, TensorTuple params, Bool has_biases, Int32 num_layers, Float dropout, Bool train, Bool bidirectional, Bool batch_first)";
  static FunctionDef function_def;
};

constexpr size_t RnnReluInputSchema_TtTTTtBI32FBBB::max_args;
constexpr size_t RnnReluInputSchema_TtTTTtBI32FBBB::max_pos_args;
constexpr char const* RnnReluInputSchema_TtTTTtBI32FBBB::signature;
FunctionDef RnnReluInputSchema_TtTTTtBI32FBBB::function_def = {
/*name*/"rnn_relu",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"hx", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"params", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"has_biases", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"num_layers", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dropout", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"train", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bidirectional", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"batch_first", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct RnnReluDataSchema_TtTTTTtBI32FBB {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& data, const std::shared_ptr<one::Tensor>& batch_sizes, const std::shared_ptr<one::Tensor>& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::RnnReluData;
  static constexpr size_t max_args = 9;
  static constexpr size_t max_pos_args = 9;
  static constexpr char const* signature = "TensorTuple (Tensor data, Tensor batch_sizes, Tensor hx, TensorTuple params, Bool has_biases, Int32 num_layers, Float dropout, Bool train, Bool bidirectional)";
  static FunctionDef function_def;
};

constexpr size_t RnnReluDataSchema_TtTTTTtBI32FBB::max_args;
constexpr size_t RnnReluDataSchema_TtTTTTtBI32FBB::max_pos_args;
constexpr char const* RnnReluDataSchema_TtTTTTtBI32FBB::signature;
FunctionDef RnnReluDataSchema_TtTTTTtBI32FBB::function_def = {
/*name*/"rnn_relu",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"data", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"batch_sizes", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"hx", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"params", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"has_biases", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"num_layers", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dropout", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"train", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bidirectional", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* rnn_relu(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("rnn_relu");
  PythonFrameGuard pf;
  static PythonArgParser<functional::RnnReluInputSchema_TtTTTtBI32FBBB, functional::RnnReluDataSchema_TtTTTTtBI32FBB> parser("rnn_relu");
  ParsedArgs<9> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::RnnReluInput(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<TensorTuple>(), r[3].As<bool>(), r[4].As<int32_t>(), r[5].As<float>(), r[6].As<bool>(), r[7].As<bool>(), r[8].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::RnnReluData(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<TensorTuple>(), r[4].As<bool>(), r[5].As<int32_t>(), r[6].As<float>(), r[7].As<bool>(), r[8].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct LstmInputSchema_TtTTtTtBI32FBBB {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, const TensorTuple& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional, bool batch_first);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::LstmInput;
  static constexpr size_t max_args = 9;
  static constexpr size_t max_pos_args = 9;
  static constexpr char const* signature = "TensorTuple (Tensor input, TensorTuple hx, TensorTuple params, Bool has_biases, Int32 num_layers, Float dropout, Bool train, Bool bidirectional, Bool batch_first)";
  static FunctionDef function_def;
};

constexpr size_t LstmInputSchema_TtTTtTtBI32FBBB::max_args;
constexpr size_t LstmInputSchema_TtTTtTtBI32FBBB::max_pos_args;
constexpr char const* LstmInputSchema_TtTTtTtBI32FBBB::signature;
FunctionDef LstmInputSchema_TtTTtTtBI32FBBB::function_def = {
/*name*/"lstm",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"hx", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"params", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"has_biases", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"num_layers", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dropout", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"train", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bidirectional", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"batch_first", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct LstmDataSchema_TtTTTtTtBI32FBB {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& data, const std::shared_ptr<one::Tensor>& batch_sizes, const TensorTuple& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::LstmData;
  static constexpr size_t max_args = 9;
  static constexpr size_t max_pos_args = 9;
  static constexpr char const* signature = "TensorTuple (Tensor data, Tensor batch_sizes, TensorTuple hx, TensorTuple params, Bool has_biases, Int32 num_layers, Float dropout, Bool train, Bool bidirectional)";
  static FunctionDef function_def;
};

constexpr size_t LstmDataSchema_TtTTTtTtBI32FBB::max_args;
constexpr size_t LstmDataSchema_TtTTTtTtBI32FBB::max_pos_args;
constexpr char const* LstmDataSchema_TtTTTtTtBI32FBB::signature;
FunctionDef LstmDataSchema_TtTTTtTtBI32FBB::function_def = {
/*name*/"lstm",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"data", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"batch_sizes", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"hx", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"params", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"has_biases", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"num_layers", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dropout", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"train", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bidirectional", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* lstm(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("lstm");
  PythonFrameGuard pf;
  static PythonArgParser<functional::LstmInputSchema_TtTTtTtBI32FBBB, functional::LstmDataSchema_TtTTTtTtBI32FBB> parser("lstm");
  ParsedArgs<9> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::LstmInput(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<TensorTuple>(), r[2].As<TensorTuple>(), r[3].As<bool>(), r[4].As<int32_t>(), r[5].As<float>(), r[6].As<bool>(), r[7].As<bool>(), r[8].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::LstmData(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<TensorTuple>(), r[3].As<TensorTuple>(), r[4].As<bool>(), r[5].As<int32_t>(), r[6].As<float>(), r[7].As<bool>(), r[8].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct GruInputSchema_TtTTTtBI32FBBB {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional, bool batch_first);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::GruInput;
  static constexpr size_t max_args = 9;
  static constexpr size_t max_pos_args = 9;
  static constexpr char const* signature = "TensorTuple (Tensor input, Tensor hx, TensorTuple params, Bool has_biases, Int32 num_layers, Float dropout, Bool train, Bool bidirectional, Bool batch_first)";
  static FunctionDef function_def;
};

constexpr size_t GruInputSchema_TtTTTtBI32FBBB::max_args;
constexpr size_t GruInputSchema_TtTTTtBI32FBBB::max_pos_args;
constexpr char const* GruInputSchema_TtTTTtBI32FBBB::signature;
FunctionDef GruInputSchema_TtTTTtBI32FBBB::function_def = {
/*name*/"gru",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"hx", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"params", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"has_biases", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"num_layers", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dropout", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"train", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bidirectional", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"batch_first", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

struct GruDataSchema_TtTTTTtBI32FBB {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& data, const std::shared_ptr<one::Tensor>& batch_sizes, const std::shared_ptr<one::Tensor>& hx, const TensorTuple& params, bool has_biases, int32_t num_layers, float dropout, bool train, bool bidirectional);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::GruData;
  static constexpr size_t max_args = 9;
  static constexpr size_t max_pos_args = 9;
  static constexpr char const* signature = "TensorTuple (Tensor data, Tensor batch_sizes, Tensor hx, TensorTuple params, Bool has_biases, Int32 num_layers, Float dropout, Bool train, Bool bidirectional)";
  static FunctionDef function_def;
};

constexpr size_t GruDataSchema_TtTTTTtBI32FBB::max_args;
constexpr size_t GruDataSchema_TtTTTTtBI32FBB::max_pos_args;
constexpr char const* GruDataSchema_TtTTTTtBI32FBB::signature;
FunctionDef GruDataSchema_TtTTTTtBI32FBB::function_def = {
/*name*/"gru",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"data", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"batch_sizes", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"hx", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"params", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"has_biases", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"num_layers", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dropout", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"train", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bidirectional", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* gru(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("gru");
  PythonFrameGuard pf;
  static PythonArgParser<functional::GruInputSchema_TtTTTtBI32FBBB, functional::GruDataSchema_TtTTTTtBI32FBB> parser("gru");
  ParsedArgs<9> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::GruInput(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<TensorTuple>(), r[3].As<bool>(), r[4].As<int32_t>(), r[5].As<float>(), r[6].As<bool>(), r[7].As<bool>(), r[8].As<bool>()));
  }
  if (idx == 1) {
    return CastToPyObject(functional::GruData(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<TensorTuple>(), r[4].As<bool>(), r[5].As<int32_t>(), r[6].As<float>(), r[7].As<bool>(), r[8].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct PackPaddedSequenceSchema_TtTTB {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& lengths, bool batch_first);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::PackPaddedSequence;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "TensorTuple (Tensor input, Tensor lengths, Bool batch_first)";
  static FunctionDef function_def;
};

constexpr size_t PackPaddedSequenceSchema_TtTTB::max_args;
constexpr size_t PackPaddedSequenceSchema_TtTTB::max_pos_args;
constexpr char const* PackPaddedSequenceSchema_TtTTB::signature;
FunctionDef PackPaddedSequenceSchema_TtTTB::function_def = {
/*name*/"pack_padded_sequence",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"lengths", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"batch_first", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* pack_padded_sequence(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("pack_padded_sequence");
  PythonFrameGuard pf;
  static PythonArgParser<functional::PackPaddedSequenceSchema_TtTTB> parser("pack_padded_sequence");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::PackPaddedSequence(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MultiTensorSgdUpdateSchema_VTtTtDFF {
  using FType = Maybe<void> (const TensorTuple& model, const TensorTuple& model_diff, double scale, float weight_decay, float learning_rate_val);
  using R = Maybe<void>;

  static constexpr FType* func = &functional::MultiTensorSgdUpdate;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "Void (TensorTuple model, TensorTuple model_diff, Double scale, Float weight_decay, Float learning_rate_val)";
  static FunctionDef function_def;
};

constexpr size_t MultiTensorSgdUpdateSchema_VTtTtDFF::max_args;
constexpr size_t MultiTensorSgdUpdateSchema_VTtTtDFF::max_pos_args;
constexpr char const* MultiTensorSgdUpdateSchema_VTtTtDFF::signature;
FunctionDef MultiTensorSgdUpdateSchema_VTtTtDFF::function_def = {
/*name*/"multi_tensor_sgd_update",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<void>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"model", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"model_diff", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"scale", /*value_type*/ValueTypeOf<double>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight_decay", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"learning_rate_val", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* multi_tensor_sgd_update(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("multi_tensor_sgd_update");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MultiTensorSgdUpdateSchema_VTtTtDFF> parser("multi_tensor_sgd_update");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::MultiTensorSgdUpdate(r[0].As<TensorTuple>(), r[1].As<TensorTuple>(), r[2].As<double>(), r[3].As<float>(), r[4].As<float>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MultiTensorAdamUpdateSchema_VTtTtTtTtFFFFFFBDFF {
  using FType = Maybe<void> (const TensorTuple& model, const TensorTuple& model_diff, const TensorTuple& m, const TensorTuple& v, float learning_rate_val, float l2, float beta1, float beta2, float bias_correction1_val, float bias_correction2_val, bool do_bias_correction, double scale, float weight_decay, float epsilon);
  using R = Maybe<void>;

  static constexpr FType* func = &functional::MultiTensorAdamUpdate;
  static constexpr size_t max_args = 14;
  static constexpr size_t max_pos_args = 14;
  static constexpr char const* signature = "Void (TensorTuple model, TensorTuple model_diff, TensorTuple m, TensorTuple v, Float learning_rate_val, Float l2, Float beta1, Float beta2, Float bias_correction1_val, Float bias_correction2_val, Bool do_bias_correction, Double scale, Float weight_decay, Float epsilon)";
  static FunctionDef function_def;
};

constexpr size_t MultiTensorAdamUpdateSchema_VTtTtTtTtFFFFFFBDFF::max_args;
constexpr size_t MultiTensorAdamUpdateSchema_VTtTtTtTtFFFFFFBDFF::max_pos_args;
constexpr char const* MultiTensorAdamUpdateSchema_VTtTtTtTtFFFFFFBDFF::signature;
FunctionDef MultiTensorAdamUpdateSchema_VTtTtTtTtFFFFFFBDFF::function_def = {
/*name*/"multi_tensor_adam_update",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<void>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"model", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"model_diff", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"m", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"v", /*value_type*/ValueTypeOf<TensorTuple>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"learning_rate_val", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"l2", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"beta1", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"beta2", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bias_correction1_val", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bias_correction2_val", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"do_bias_correction", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"scale", /*value_type*/ValueTypeOf<double>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight_decay", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"epsilon", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* multi_tensor_adam_update(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("multi_tensor_adam_update");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MultiTensorAdamUpdateSchema_VTtTtTtTtFFFFFFBDFF> parser("multi_tensor_adam_update");
  ParsedArgs<14> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::MultiTensorAdamUpdate(r[0].As<TensorTuple>(), r[1].As<TensorTuple>(), r[2].As<TensorTuple>(), r[3].As<TensorTuple>(), r[4].As<float>(), r[5].As<float>(), r[6].As<float>(), r[7].As<float>(), r[8].As<float>(), r[9].As<float>(), r[10].As<bool>(), r[11].As<double>(), r[12].As<float>(), r[13].As<float>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct TruncSchema_TT {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Trunc;
  static constexpr size_t max_args = 1;
  static constexpr size_t max_pos_args = 1;
  static constexpr char const* signature = "Tensor (Tensor input)";
  static FunctionDef function_def;
};

constexpr size_t TruncSchema_TT::max_args;
constexpr size_t TruncSchema_TT::max_pos_args;
constexpr char const* TruncSchema_TT::signature;
FunctionDef TruncSchema_TT::function_def = {
/*name*/"trunc",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* trunc(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("trunc");
  PythonFrameGuard pf;
  static PythonArgParser<functional::TruncSchema_TT> parser("trunc");
  ParsedArgs<1> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Trunc(r[0].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BatchNormStatsSchema_TtTI32F {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, int32_t axis, float eps);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::BatchNormStats;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "TensorTuple (Tensor input, Int32 axis, Float eps)";
  static FunctionDef function_def;
};

constexpr size_t BatchNormStatsSchema_TtTI32F::max_args;
constexpr size_t BatchNormStatsSchema_TtTI32F::max_pos_args;
constexpr char const* BatchNormStatsSchema_TtTI32F::signature;
FunctionDef BatchNormStatsSchema_TtTI32F::function_def = {
/*name*/"batch_norm_stats",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"axis", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"eps", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* batch_norm_stats(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("batch_norm_stats");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BatchNormStatsSchema_TtTI32F> parser("batch_norm_stats");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BatchNormStats(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>(), r[2].As<float>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BatchNormGatherStatsWithCountsSchema_TtTTTTTFFT {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& invstd, const Optional<one::Tensor>& running_mean, const Optional<one::Tensor>& running_var, float momentum, float eps, const std::shared_ptr<one::Tensor>& counts);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::BatchNormGatherStatsWithCounts;
  static constexpr size_t max_args = 8;
  static constexpr size_t max_pos_args = 8;
  static constexpr char const* signature = "TensorTuple (Tensor input, Tensor mean, Tensor invstd, Tensor running_mean=None, Tensor running_var=None, Float momentum, Float eps, Tensor counts)";
  static FunctionDef function_def;
};

constexpr size_t BatchNormGatherStatsWithCountsSchema_TtTTTTTFFT::max_args;
constexpr size_t BatchNormGatherStatsWithCountsSchema_TtTTTTTFFT::max_pos_args;
constexpr char const* BatchNormGatherStatsWithCountsSchema_TtTTTTTFFT::signature;
FunctionDef BatchNormGatherStatsWithCountsSchema_TtTTTTTFFT::function_def = {
/*name*/"batch_norm_gather_stats_with_counts",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"mean", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"invstd", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"running_mean", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"running_var", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"momentum", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"eps", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"counts", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* batch_norm_gather_stats_with_counts(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("batch_norm_gather_stats_with_counts");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BatchNormGatherStatsWithCountsSchema_TtTTTTTFFT> parser("batch_norm_gather_stats_with_counts");
  ParsedArgs<8> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BatchNormGatherStatsWithCounts(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<Optional<one::Tensor>>(), r[4].As<Optional<one::Tensor>>(), r[5].As<float>(), r[6].As<float>(), r[7].As<std::shared_ptr<one::Tensor>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BatchNormElemtSchema_TTTTTTI32F {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& bias, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& invstd, int32_t axis, float eps);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BatchNormElemt;
  static constexpr size_t max_args = 7;
  static constexpr size_t max_pos_args = 7;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor weight, Tensor bias, Tensor mean, Tensor invstd, Int32 axis, Float eps)";
  static FunctionDef function_def;
};

constexpr size_t BatchNormElemtSchema_TTTTTTI32F::max_args;
constexpr size_t BatchNormElemtSchema_TTTTTTI32F::max_pos_args;
constexpr char const* BatchNormElemtSchema_TTTTTTI32F::signature;
FunctionDef BatchNormElemtSchema_TTTTTTI32F::function_def = {
/*name*/"batch_norm_elemt",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bias", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"mean", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"invstd", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"axis", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"eps", /*value_type*/ValueTypeOf<float>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* batch_norm_elemt(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("batch_norm_elemt");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BatchNormElemtSchema_TTTTTTI32F> parser("batch_norm_elemt");
  ParsedArgs<7> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BatchNormElemt(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<std::shared_ptr<one::Tensor>>(), r[4].As<std::shared_ptr<one::Tensor>>(), r[5].As<int32_t>(), r[6].As<float>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BatchNormBackwardReduceSchema_TtTTTTI32 {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& grad_out, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& invstd, int32_t axis);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::BatchNormBackwardReduce;
  static constexpr size_t max_args = 5;
  static constexpr size_t max_pos_args = 5;
  static constexpr char const* signature = "TensorTuple (Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Int32 axis)";
  static FunctionDef function_def;
};

constexpr size_t BatchNormBackwardReduceSchema_TtTTTTI32::max_args;
constexpr size_t BatchNormBackwardReduceSchema_TtTTTTI32::max_pos_args;
constexpr char const* BatchNormBackwardReduceSchema_TtTTTTI32::signature;
FunctionDef BatchNormBackwardReduceSchema_TtTTTTI32::function_def = {
/*name*/"batch_norm_backward_reduce",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"grad_out", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"mean", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"invstd", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"axis", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* batch_norm_backward_reduce(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("batch_norm_backward_reduce");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BatchNormBackwardReduceSchema_TtTTTTI32> parser("batch_norm_backward_reduce");
  ParsedArgs<5> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BatchNormBackwardReduce(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<std::shared_ptr<one::Tensor>>(), r[4].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BatchNormBackwardElemtSchema_TTTTTTTTTI32 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& grad_out, const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& mean, const std::shared_ptr<one::Tensor>& invstd, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& sum_dy, const std::shared_ptr<one::Tensor>& sum_dy_xmu, const std::shared_ptr<one::Tensor>& count, int32_t axis);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BatchNormBackwardElemt;
  static constexpr size_t max_args = 9;
  static constexpr size_t max_pos_args = 9;
  static constexpr char const* signature = "Tensor (Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor weight, Tensor sum_dy, Tensor sum_dy_xmu, Tensor count, Int32 axis)";
  static FunctionDef function_def;
};

constexpr size_t BatchNormBackwardElemtSchema_TTTTTTTTTI32::max_args;
constexpr size_t BatchNormBackwardElemtSchema_TTTTTTTTTI32::max_pos_args;
constexpr char const* BatchNormBackwardElemtSchema_TTTTTTTTTI32::signature;
FunctionDef BatchNormBackwardElemtSchema_TTTTTTTTTI32::function_def = {
/*name*/"batch_norm_backward_elemt",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"grad_out", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"mean", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"invstd", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"sum_dy", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"sum_dy_xmu", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"count", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"axis", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* batch_norm_backward_elemt(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("batch_norm_backward_elemt");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BatchNormBackwardElemtSchema_TTTTTTTTTI32> parser("batch_norm_backward_elemt");
  ParsedArgs<9> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BatchNormBackwardElemt(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<std::shared_ptr<one::Tensor>>(), r[4].As<std::shared_ptr<one::Tensor>>(), r[5].As<std::shared_ptr<one::Tensor>>(), r[6].As<std::shared_ptr<one::Tensor>>(), r[7].As<std::shared_ptr<one::Tensor>>(), r[8].As<int32_t>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AdaptiveMaxPool1DSchema_TtTI64l {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, const std::vector<int64_t>& output_size);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::AdaptiveMaxPool1D;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "TensorTuple (Tensor input, Int64List output_size)";
  static FunctionDef function_def;
};

constexpr size_t AdaptiveMaxPool1DSchema_TtTI64l::max_args;
constexpr size_t AdaptiveMaxPool1DSchema_TtTI64l::max_pos_args;
constexpr char const* AdaptiveMaxPool1DSchema_TtTI64l::signature;
FunctionDef AdaptiveMaxPool1DSchema_TtTI64l::function_def = {
/*name*/"adaptive_max_pool1d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"output_size", /*value_type*/ValueTypeOf<std::vector<int64_t>>(), /*size*/1, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* adaptive_max_pool1d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("adaptive_max_pool1d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AdaptiveMaxPool1DSchema_TtTI64l> parser("adaptive_max_pool1d");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::AdaptiveMaxPool1D(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int64_t>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AdaptiveMaxPool2DSchema_TtTI64l {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, const std::vector<int64_t>& output_size);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::AdaptiveMaxPool2D;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "TensorTuple (Tensor input, Int64List output_size)";
  static FunctionDef function_def;
};

constexpr size_t AdaptiveMaxPool2DSchema_TtTI64l::max_args;
constexpr size_t AdaptiveMaxPool2DSchema_TtTI64l::max_pos_args;
constexpr char const* AdaptiveMaxPool2DSchema_TtTI64l::signature;
FunctionDef AdaptiveMaxPool2DSchema_TtTI64l::function_def = {
/*name*/"adaptive_max_pool2d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"output_size", /*value_type*/ValueTypeOf<std::vector<int64_t>>(), /*size*/2, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* adaptive_max_pool2d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("adaptive_max_pool2d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AdaptiveMaxPool2DSchema_TtTI64l> parser("adaptive_max_pool2d");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::AdaptiveMaxPool2D(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int64_t>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct AdaptiveMaxPool3DSchema_TtTI64l {
  using FType = Maybe<one::TensorTuple> (const std::shared_ptr<one::Tensor>& input, const std::vector<int64_t>& output_size);
  using R = Maybe<one::TensorTuple>;

  static constexpr FType* func = &functional::AdaptiveMaxPool3D;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_pos_args = 2;
  static constexpr char const* signature = "TensorTuple (Tensor input, Int64List output_size)";
  static FunctionDef function_def;
};

constexpr size_t AdaptiveMaxPool3DSchema_TtTI64l::max_args;
constexpr size_t AdaptiveMaxPool3DSchema_TtTI64l::max_pos_args;
constexpr char const* AdaptiveMaxPool3DSchema_TtTI64l::signature;
FunctionDef AdaptiveMaxPool3DSchema_TtTI64l::function_def = {
/*name*/"adaptive_max_pool3d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::TensorTuple>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"output_size", /*value_type*/ValueTypeOf<std::vector<int64_t>>(), /*size*/3, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* adaptive_max_pool3d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("adaptive_max_pool3d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::AdaptiveMaxPool3DSchema_TtTI64l> parser("adaptive_max_pool3d");
  ParsedArgs<2> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::AdaptiveMaxPool3D(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::vector<int64_t>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct ExponentialSchema_TTFG {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, float lambd, const Optional<one::Generator>& generator);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Exponential;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor x, Float lambd=1.0, Generator generator=None)";
  static FunctionDef function_def;
};

constexpr size_t ExponentialSchema_TTFG::max_args;
constexpr size_t ExponentialSchema_TTFG::max_pos_args;
constexpr char const* ExponentialSchema_TTFG::signature;
FunctionDef ExponentialSchema_TTFG::function_def = {
/*name*/"exponential_",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"lambd", /*default_value*/float(1.0), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* exponential_(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("exponential_");
  PythonFrameGuard pf;
  static PythonArgParser<functional::ExponentialSchema_TTFG> parser("exponential_");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Exponential(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<float>(), r[2].As<Optional<one::Generator>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct MultinomialSchema_TTI32BG {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& x, int32_t num_samples, bool replacement, const Optional<one::Generator>& generator);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Multinomial;
  static constexpr size_t max_args = 4;
  static constexpr size_t max_pos_args = 4;
  static constexpr char const* signature = "Tensor (Tensor x, Int32 num_samples, Bool replacement=False, Generator generator=None)";
  static FunctionDef function_def;
};

constexpr size_t MultinomialSchema_TTI32BG::max_args;
constexpr size_t MultinomialSchema_TTI32BG::max_pos_args;
constexpr char const* MultinomialSchema_TTI32BG::signature;
FunctionDef MultinomialSchema_TTI32BG::function_def = {
/*name*/"multinomial",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"x", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"num_samples", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"replacement", /*default_value*/bool(false), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"generator", /*default_value*/Optional<one::Generator>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* multinomial(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("multinomial");
  PythonFrameGuard pf;
  static PythonArgParser<functional::MultinomialSchema_TTI32BG> parser("multinomial");
  ParsedArgs<4> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::Multinomial(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<int32_t>(), r[2].As<bool>(), r[3].As<Optional<one::Generator>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct DeformConv2dSchema_TTTTTTI32I32I32I32I32I32I32I32B {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const std::shared_ptr<one::Tensor>& weight, const std::shared_ptr<one::Tensor>& offset, const std::shared_ptr<one::Tensor>& mask, const Optional<one::Tensor>& bias, int32_t stride_h, int32_t stride_w, int32_t pad_h, int32_t pad_w, int32_t dilation_h, int32_t dilation_w, int32_t groups, int32_t offset_groups, bool use_mask);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::DeformConv2d;
  static constexpr size_t max_args = 14;
  static constexpr size_t max_pos_args = 14;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor weight, Tensor offset, Tensor mask, Tensor bias=None, Int32 stride_h, Int32 stride_w, Int32 pad_h, Int32 pad_w, Int32 dilation_h, Int32 dilation_w, Int32 groups, Int32 offset_groups, Bool use_mask)";
  static FunctionDef function_def;
};

constexpr size_t DeformConv2dSchema_TTTTTTI32I32I32I32I32I32I32I32B::max_args;
constexpr size_t DeformConv2dSchema_TTTTTTI32I32I32I32I32I32I32I32B::max_pos_args;
constexpr char const* DeformConv2dSchema_TTTTTTI32I32I32I32I32I32I32I32B::signature;
FunctionDef DeformConv2dSchema_TTTTTTI32I32I32I32I32I32I32I32B::function_def = {
/*name*/"deform_conv2d",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weight", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"offset", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"mask", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"bias", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"stride_h", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"stride_w", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"pad_h", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"pad_w", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dilation_h", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"dilation_w", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"groups", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"offset_groups", /*value_type*/ValueTypeOf<int32_t>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"use_mask", /*value_type*/ValueTypeOf<bool>(), /*size*/0, /*keyword_only*/false, /*optional*/false)
}
};

PyObject* deform_conv2d(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("deform_conv2d");
  PythonFrameGuard pf;
  static PythonArgParser<functional::DeformConv2dSchema_TTTTTTI32I32I32I32I32I32I32I32B> parser("deform_conv2d");
  ParsedArgs<14> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::DeformConv2d(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<std::shared_ptr<one::Tensor>>(), r[2].As<std::shared_ptr<one::Tensor>>(), r[3].As<std::shared_ptr<one::Tensor>>(), r[4].As<Optional<one::Tensor>>(), r[5].As<int32_t>(), r[6].As<int32_t>(), r[7].As<int32_t>(), r[8].As<int32_t>(), r[9].As<int32_t>(), r[10].As<int32_t>(), r[11].As<int32_t>(), r[12].As<int32_t>(), r[13].As<bool>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

struct BinCountSchema_TTTI64 {
  using FType = Maybe<one::Tensor> (const std::shared_ptr<one::Tensor>& input, const Optional<one::Tensor>& weights, const Optional<int64_t>& minlength);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::BinCount;
  static constexpr size_t max_args = 3;
  static constexpr size_t max_pos_args = 3;
  static constexpr char const* signature = "Tensor (Tensor input, Tensor weights=None, Int64 minlength=None)";
  static FunctionDef function_def;
};

constexpr size_t BinCountSchema_TTTI64::max_args;
constexpr size_t BinCountSchema_TTTI64::max_pos_args;
constexpr char const* BinCountSchema_TTTI64::signature;
FunctionDef BinCountSchema_TTTI64::function_def = {
/*name*/"bincount",
/*return_def*/ReturnDef(ValueTypeOf<Maybe<one::Tensor>>()),
/*argument_def*/{
  ArgumentDef(/*name*/"input", /*value_type*/ValueTypeOf<std::shared_ptr<one::Tensor>>(), /*size*/0, /*keyword_only*/false, /*optional*/false),
  ArgumentDef(/*name*/"weights", /*default_value*/Optional<one::Tensor>(), /*size*/0, /*keyword_only*/false, /*optional*/true),
  ArgumentDef(/*name*/"minlength", /*default_value*/Optional<int64_t>(), /*size*/0, /*keyword_only*/false, /*optional*/true)
}
};

PyObject* bincount(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_ERRORS
  OF_PROFILER_RANGE_GUARD("bincount");
  PythonFrameGuard pf;
  static PythonArgParser<functional::BinCountSchema_TTTI64> parser("bincount");
  ParsedArgs<3> r;
  int idx = parser.Parse(args, kwargs, &r);
  if (idx == 0) {
    return CastToPyObject(functional::BinCount(r[0].As<std::shared_ptr<one::Tensor>>(), r[1].As<Optional<one::Tensor>>(), r[2].As<Optional<int64_t>>()));
  }
  Py_RETURN_NONE;
  END_HANDLE_ERRORS
}

}  // namespace functional
}  // namespace one

namespace functional = one::functional;

ONEFLOW_API_PYBIND11_MODULE("_C", m) {
  static PyMethodDef functions[] = {
    {"add", (PyCFunction)functional::add, METH_VARARGS | METH_KEYWORDS, NULL},
    {"amin", (PyCFunction)functional::amin, METH_VARARGS | METH_KEYWORDS, NULL},
    {"sub", (PyCFunction)functional::sub, METH_VARARGS | METH_KEYWORDS, NULL},
    {"mul", (PyCFunction)functional::mul, METH_VARARGS | METH_KEYWORDS, NULL},
    {"mul_", (PyCFunction)functional::mul_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"addcmul", (PyCFunction)functional::addcmul, METH_VARARGS | METH_KEYWORDS, NULL},
    {"addcmul_", (PyCFunction)functional::addcmul_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"addcdiv", (PyCFunction)functional::addcdiv, METH_VARARGS | METH_KEYWORDS, NULL},
    {"addcdiv_", (PyCFunction)functional::addcdiv_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"div", (PyCFunction)functional::div, METH_VARARGS | METH_KEYWORDS, NULL},
    {"div_", (PyCFunction)functional::div_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"equal", (PyCFunction)functional::equal, METH_VARARGS | METH_KEYWORDS, NULL},
    {"not_equal", (PyCFunction)functional::not_equal, METH_VARARGS | METH_KEYWORDS, NULL},
    {"greater", (PyCFunction)functional::greater, METH_VARARGS | METH_KEYWORDS, NULL},
    {"greater_equal", (PyCFunction)functional::greater_equal, METH_VARARGS | METH_KEYWORDS, NULL},
    {"logical_and", (PyCFunction)functional::logical_and, METH_VARARGS | METH_KEYWORDS, NULL},
    {"logical_or", (PyCFunction)functional::logical_or, METH_VARARGS | METH_KEYWORDS, NULL},
    {"logical_not", (PyCFunction)functional::logical_not, METH_VARARGS | METH_KEYWORDS, NULL},
    {"logical_xor", (PyCFunction)functional::logical_xor, METH_VARARGS | METH_KEYWORDS, NULL},
    {"less", (PyCFunction)functional::less, METH_VARARGS | METH_KEYWORDS, NULL},
    {"less_equal", (PyCFunction)functional::less_equal, METH_VARARGS | METH_KEYWORDS, NULL},
    {"pow", (PyCFunction)functional::pow, METH_VARARGS | METH_KEYWORDS, NULL},
    {"searchsorted", (PyCFunction)functional::searchsorted, METH_VARARGS | METH_KEYWORDS, NULL},
    {"floor_divide", (PyCFunction)functional::floor_divide, METH_VARARGS | METH_KEYWORDS, NULL},
    {"trunc_divide", (PyCFunction)functional::trunc_divide, METH_VARARGS | METH_KEYWORDS, NULL},
    {"max", (PyCFunction)functional::max, METH_VARARGS | METH_KEYWORDS, NULL},
    {"min", (PyCFunction)functional::min, METH_VARARGS | METH_KEYWORDS, NULL},
    {"median", (PyCFunction)functional::median, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reduce_max", (PyCFunction)functional::reduce_max, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reduce_min", (PyCFunction)functional::reduce_min, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reduce_sum", (PyCFunction)functional::reduce_sum, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reduce_nansum", (PyCFunction)functional::reduce_nansum, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reduce_mean", (PyCFunction)functional::reduce_mean, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reduce_all", (PyCFunction)functional::reduce_all, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reduce_any", (PyCFunction)functional::reduce_any, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reduce_prod", (PyCFunction)functional::reduce_prod, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reduce_min_device_stage", (PyCFunction)functional::reduce_min_device_stage, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reduce_max_device_stage", (PyCFunction)functional::reduce_max_device_stage, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reduce_min_global_stage", (PyCFunction)functional::reduce_min_global_stage, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reduce_max_global_stage", (PyCFunction)functional::reduce_max_global_stage, METH_VARARGS | METH_KEYWORDS, NULL},
    {"transpose", (PyCFunction)functional::transpose, METH_VARARGS | METH_KEYWORDS, NULL},
    {"as_strided", (PyCFunction)functional::as_strided, METH_VARARGS | METH_KEYWORDS, NULL},
    {"select", (PyCFunction)functional::select, METH_VARARGS | METH_KEYWORDS, NULL},
    {"swapaxes", (PyCFunction)functional::swapaxes, METH_VARARGS | METH_KEYWORDS, NULL},
    {"swapdims", (PyCFunction)functional::swapdims, METH_VARARGS | METH_KEYWORDS, NULL},
    {"amax", (PyCFunction)functional::amax, METH_VARARGS | METH_KEYWORDS, NULL},
    {"permute", (PyCFunction)functional::permute, METH_VARARGS | METH_KEYWORDS, NULL},
    {"T", (PyCFunction)functional::T, METH_VARARGS | METH_KEYWORDS, NULL},
    {"t", (PyCFunction)functional::t, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reciprocal", (PyCFunction)functional::reciprocal, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reciprocal_no_nan", (PyCFunction)functional::reciprocal_no_nan, METH_VARARGS | METH_KEYWORDS, NULL},
    {"image_flip", (PyCFunction)functional::image_flip, METH_VARARGS | METH_KEYWORDS, NULL},
    {"sin", (PyCFunction)functional::sin, METH_VARARGS | METH_KEYWORDS, NULL},
    {"sin_", (PyCFunction)functional::sin_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"cos", (PyCFunction)functional::cos, METH_VARARGS | METH_KEYWORDS, NULL},
    {"cosh", (PyCFunction)functional::cosh, METH_VARARGS | METH_KEYWORDS, NULL},
    {"cosh_grad", (PyCFunction)functional::cosh_grad, METH_VARARGS | METH_KEYWORDS, NULL},
    {"fmod", (PyCFunction)functional::fmod, METH_VARARGS | METH_KEYWORDS, NULL},
    {"log", (PyCFunction)functional::log, METH_VARARGS | METH_KEYWORDS, NULL},
    {"log2", (PyCFunction)functional::log2, METH_VARARGS | METH_KEYWORDS, NULL},
    {"log10", (PyCFunction)functional::log10, METH_VARARGS | METH_KEYWORDS, NULL},
    {"sqrt", (PyCFunction)functional::sqrt, METH_VARARGS | METH_KEYWORDS, NULL},
    {"rsqrt", (PyCFunction)functional::rsqrt, METH_VARARGS | METH_KEYWORDS, NULL},
    {"square", (PyCFunction)functional::square, METH_VARARGS | METH_KEYWORDS, NULL},
    {"sqrt_square_sum", (PyCFunction)functional::sqrt_square_sum, METH_VARARGS | METH_KEYWORDS, NULL},
    {"std", (PyCFunction)functional::std, METH_VARARGS | METH_KEYWORDS, NULL},
    {"var", (PyCFunction)functional::var, METH_VARARGS | METH_KEYWORDS, NULL},
    {"rms_layer_norm", (PyCFunction)functional::rms_layer_norm, METH_VARARGS | METH_KEYWORDS, NULL},
    {"relu", (PyCFunction)functional::relu, METH_VARARGS | METH_KEYWORDS, NULL},
    {"hann_window", (PyCFunction)functional::hann_window, METH_VARARGS | METH_KEYWORDS, NULL},
    {"hardtanh", (PyCFunction)functional::hardtanh, METH_VARARGS | METH_KEYWORDS, NULL},
    {"tan", (PyCFunction)functional::tan, METH_VARARGS | METH_KEYWORDS, NULL},
    {"tan_grad", (PyCFunction)functional::tan_grad, METH_VARARGS | METH_KEYWORDS, NULL},
    {"tanh", (PyCFunction)functional::tanh, METH_VARARGS | METH_KEYWORDS, NULL},
    {"tanh_grad", (PyCFunction)functional::tanh_grad, METH_VARARGS | METH_KEYWORDS, NULL},
    {"threshold", (PyCFunction)functional::threshold, METH_VARARGS | METH_KEYWORDS, NULL},
    {"elu", (PyCFunction)functional::elu, METH_VARARGS | METH_KEYWORDS, NULL},
    {"celu", (PyCFunction)functional::celu, METH_VARARGS | METH_KEYWORDS, NULL},
    {"gelu", (PyCFunction)functional::gelu, METH_VARARGS | METH_KEYWORDS, NULL},
    {"gelu_with_approximate", (PyCFunction)functional::gelu_with_approximate, METH_VARARGS | METH_KEYWORDS, NULL},
    {"glu", (PyCFunction)functional::glu, METH_VARARGS | METH_KEYWORDS, NULL},
    {"sigmoid", (PyCFunction)functional::sigmoid, METH_VARARGS | METH_KEYWORDS, NULL},
    {"sigmoid_grad", (PyCFunction)functional::sigmoid_grad, METH_VARARGS | METH_KEYWORDS, NULL},
    {"hardsigmoid", (PyCFunction)functional::hardsigmoid, METH_VARARGS | METH_KEYWORDS, NULL},
    {"hardshrink", (PyCFunction)functional::hardshrink, METH_VARARGS | METH_KEYWORDS, NULL},
    {"softmax", (PyCFunction)functional::softmax, METH_VARARGS | METH_KEYWORDS, NULL},
    {"log_softmax", (PyCFunction)functional::log_softmax, METH_VARARGS | METH_KEYWORDS, NULL},
    {"hardswish", (PyCFunction)functional::hardswish, METH_VARARGS | METH_KEYWORDS, NULL},
    {"leaky_relu", (PyCFunction)functional::leaky_relu, METH_VARARGS | METH_KEYWORDS, NULL},
    {"normal", (PyCFunction)functional::normal, METH_VARARGS | METH_KEYWORDS, NULL},
    {"normalization", (PyCFunction)functional::normalization, METH_VARARGS | METH_KEYWORDS, NULL},
    {"normalization_add_relu", (PyCFunction)functional::normalization_add_relu, METH_VARARGS | METH_KEYWORDS, NULL},
    {"eye", (PyCFunction)functional::eye, METH_VARARGS | METH_KEYWORDS, NULL},
    {"eye_", (PyCFunction)functional::eye_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"erfinv", (PyCFunction)functional::erfinv, METH_VARARGS | METH_KEYWORDS, NULL},
    {"erfinv_", (PyCFunction)functional::erfinv_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"arange", (PyCFunction)functional::arange, METH_VARARGS | METH_KEYWORDS, NULL},
    {"global_arange", (PyCFunction)functional::global_arange, METH_VARARGS | METH_KEYWORDS, NULL},
    {"flatten", (PyCFunction)functional::flatten, METH_VARARGS | METH_KEYWORDS, NULL},
    {"argmax", (PyCFunction)functional::argmax, METH_VARARGS | METH_KEYWORDS, NULL},
    {"argmin", (PyCFunction)functional::argmin, METH_VARARGS | METH_KEYWORDS, NULL},
    {"argwhere", (PyCFunction)functional::argwhere, METH_VARARGS | METH_KEYWORDS, NULL},
    {"nonzero", (PyCFunction)functional::nonzero, METH_VARARGS | METH_KEYWORDS, NULL},
    {"broadcast_like", (PyCFunction)functional::broadcast_like, METH_VARARGS | METH_KEYWORDS, NULL},
    {"cast", (PyCFunction)functional::cast, METH_VARARGS | METH_KEYWORDS, NULL},
    {"constant", (PyCFunction)functional::constant, METH_VARARGS | METH_KEYWORDS, NULL},
    {"global_constant", (PyCFunction)functional::global_constant, METH_VARARGS | METH_KEYWORDS, NULL},
    {"empty", (PyCFunction)functional::empty, METH_VARARGS | METH_KEYWORDS, NULL},
    {"global_empty", (PyCFunction)functional::global_empty, METH_VARARGS | METH_KEYWORDS, NULL},
    {"bernoulli", (PyCFunction)functional::bernoulli, METH_VARARGS | METH_KEYWORDS, NULL},
    {"bernoulli_", (PyCFunction)functional::bernoulli_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"concat", (PyCFunction)functional::concat, METH_VARARGS | METH_KEYWORDS, NULL},
    {"bias_add", (PyCFunction)functional::bias_add, METH_VARARGS | METH_KEYWORDS, NULL},
    {"conv1d", (PyCFunction)functional::conv1d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"conv2d", (PyCFunction)functional::conv2d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"conv3d", (PyCFunction)functional::conv3d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"fake_quantization", (PyCFunction)functional::fake_quantization, METH_VARARGS | METH_KEYWORDS, NULL},
    {"quantization", (PyCFunction)functional::quantization, METH_VARARGS | METH_KEYWORDS, NULL},
    {"min_max_observer", (PyCFunction)functional::min_max_observer, METH_VARARGS | METH_KEYWORDS, NULL},
    {"moving_average_min_max_observer", (PyCFunction)functional::moving_average_min_max_observer, METH_VARARGS | METH_KEYWORDS, NULL},
    {"deconv1d", (PyCFunction)functional::deconv1d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"deconv2d", (PyCFunction)functional::deconv2d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"deconv3d", (PyCFunction)functional::deconv3d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"expand", (PyCFunction)functional::expand, METH_VARARGS | METH_KEYWORDS, NULL},
    {"repeat", (PyCFunction)functional::repeat, METH_VARARGS | METH_KEYWORDS, NULL},
    {"repeat_interleave", (PyCFunction)functional::repeat_interleave, METH_VARARGS | METH_KEYWORDS, NULL},
    {"tile", (PyCFunction)functional::tile, METH_VARARGS | METH_KEYWORDS, NULL},
    {"roll", (PyCFunction)functional::roll, METH_VARARGS | METH_KEYWORDS, NULL},
    {"expand_dims", (PyCFunction)functional::expand_dims, METH_VARARGS | METH_KEYWORDS, NULL},
    {"unsqueeze", (PyCFunction)functional::unsqueeze, METH_VARARGS | METH_KEYWORDS, NULL},
    {"squeeze", (PyCFunction)functional::squeeze, METH_VARARGS | METH_KEYWORDS, NULL},
    {"exp", (PyCFunction)functional::exp, METH_VARARGS | METH_KEYWORDS, NULL},
    {"gather", (PyCFunction)functional::gather, METH_VARARGS | METH_KEYWORDS, NULL},
    {"dim_gather", (PyCFunction)functional::dim_gather, METH_VARARGS | METH_KEYWORDS, NULL},
    {"embedding_renorm_", (PyCFunction)functional::embedding_renorm_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"embedding", (PyCFunction)functional::embedding, METH_VARARGS | METH_KEYWORDS, NULL},
    {"arg_sort", (PyCFunction)functional::arg_sort, METH_VARARGS | METH_KEYWORDS, NULL},
    {"gather_nd", (PyCFunction)functional::gather_nd, METH_VARARGS | METH_KEYWORDS, NULL},
    {"scatternd", (PyCFunction)functional::scatternd, METH_VARARGS | METH_KEYWORDS, NULL},
    {"tensor_scatter_nd_update", (PyCFunction)functional::tensor_scatter_nd_update, METH_VARARGS | METH_KEYWORDS, NULL},
    {"scatterndlike", (PyCFunction)functional::scatterndlike, METH_VARARGS | METH_KEYWORDS, NULL},
    {"matmul", (PyCFunction)functional::matmul, METH_VARARGS | METH_KEYWORDS, NULL},
    {"mm", (PyCFunction)functional::mm, METH_VARARGS | METH_KEYWORDS, NULL},
    {"fused_mlp", (PyCFunction)functional::fused_mlp, METH_VARARGS | METH_KEYWORDS, NULL},
    {"fused_matmul_bias_add_relu_dropout", (PyCFunction)functional::fused_matmul_bias_add_relu_dropout, METH_VARARGS | METH_KEYWORDS, NULL},
    {"batch_matmul", (PyCFunction)functional::batch_matmul, METH_VARARGS | METH_KEYWORDS, NULL},
    {"matrix_vector_product", (PyCFunction)functional::matrix_vector_product, METH_VARARGS | METH_KEYWORDS, NULL},
    {"tensordot", (PyCFunction)functional::tensordot, METH_VARARGS | METH_KEYWORDS, NULL},
    {"l1_loss", (PyCFunction)functional::l1_loss, METH_VARARGS | METH_KEYWORDS, NULL},
    {"mse_loss", (PyCFunction)functional::mse_loss, METH_VARARGS | METH_KEYWORDS, NULL},
    {"kl_div_loss", (PyCFunction)functional::kl_div_loss, METH_VARARGS | METH_KEYWORDS, NULL},
    {"nll_loss", (PyCFunction)functional::nll_loss, METH_VARARGS | METH_KEYWORDS, NULL},
    {"binary_cross_entropy_loss", (PyCFunction)functional::binary_cross_entropy_loss, METH_VARARGS | METH_KEYWORDS, NULL},
    {"binary_cross_entropy_with_logits_loss", (PyCFunction)functional::binary_cross_entropy_with_logits_loss, METH_VARARGS | METH_KEYWORDS, NULL},
    {"binary_cross_entropy_with_logits_loss_grad", (PyCFunction)functional::binary_cross_entropy_with_logits_loss_grad, METH_VARARGS | METH_KEYWORDS, NULL},
    {"sparse_cross_entropy", (PyCFunction)functional::sparse_cross_entropy, METH_VARARGS | METH_KEYWORDS, NULL},
    {"distributed_sparse_cross_entropy", (PyCFunction)functional::distributed_sparse_cross_entropy, METH_VARARGS | METH_KEYWORDS, NULL},
    {"cross_entropy", (PyCFunction)functional::cross_entropy, METH_VARARGS | METH_KEYWORDS, NULL},
    {"sparse_softmax_cross_entropy", (PyCFunction)functional::sparse_softmax_cross_entropy, METH_VARARGS | METH_KEYWORDS, NULL},
    {"softmax_cross_entropy", (PyCFunction)functional::softmax_cross_entropy, METH_VARARGS | METH_KEYWORDS, NULL},
    {"softmax_cross_entropy_grad", (PyCFunction)functional::softmax_cross_entropy_grad, METH_VARARGS | METH_KEYWORDS, NULL},
    {"smooth_l1_loss", (PyCFunction)functional::smooth_l1_loss, METH_VARARGS | METH_KEYWORDS, NULL},
    {"combined_margin_loss", (PyCFunction)functional::combined_margin_loss, METH_VARARGS | METH_KEYWORDS, NULL},
    {"triplet_margin_loss", (PyCFunction)functional::triplet_margin_loss, METH_VARARGS | METH_KEYWORDS, NULL},
    {"margin_ranking_loss", (PyCFunction)functional::margin_ranking_loss, METH_VARARGS | METH_KEYWORDS, NULL},
    {"ctc_loss", (PyCFunction)functional::ctc_loss, METH_VARARGS | METH_KEYWORDS, NULL},
    {"affine_grid", (PyCFunction)functional::affine_grid, METH_VARARGS | METH_KEYWORDS, NULL},
    {"grid_sample", (PyCFunction)functional::grid_sample, METH_VARARGS | METH_KEYWORDS, NULL},
    {"where", (PyCFunction)functional::where, METH_VARARGS | METH_KEYWORDS, NULL},
    {"masked_fill", (PyCFunction)functional::masked_fill, METH_VARARGS | METH_KEYWORDS, NULL},
    {"masked_fill_", (PyCFunction)functional::masked_fill_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"movedim", (PyCFunction)functional::movedim, METH_VARARGS | METH_KEYWORDS, NULL},
    {"tensor_split", (PyCFunction)functional::tensor_split, METH_VARARGS | METH_KEYWORDS, NULL},
    {"hsplit", (PyCFunction)functional::hsplit, METH_VARARGS | METH_KEYWORDS, NULL},
    {"vsplit", (PyCFunction)functional::vsplit, METH_VARARGS | METH_KEYWORDS, NULL},
    {"negative", (PyCFunction)functional::negative, METH_VARARGS | METH_KEYWORDS, NULL},
    {"layer_norm_affine", (PyCFunction)functional::layer_norm_affine, METH_VARARGS | METH_KEYWORDS, NULL},
    {"layer_norm", (PyCFunction)functional::layer_norm, METH_VARARGS | METH_KEYWORDS, NULL},
    {"group_norm", (PyCFunction)functional::group_norm, METH_VARARGS | METH_KEYWORDS, NULL},
    {"avg_pool2d_nhwc", (PyCFunction)functional::avg_pool2d_nhwc, METH_VARARGS | METH_KEYWORDS, NULL},
    {"adaptive_avg_pool1d", (PyCFunction)functional::adaptive_avg_pool1d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"adaptive_avg_pool2d", (PyCFunction)functional::adaptive_avg_pool2d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"adaptive_avg_pool3d", (PyCFunction)functional::adaptive_avg_pool3d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"max_pool1d", (PyCFunction)functional::max_pool1d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"max_pool2d", (PyCFunction)functional::max_pool2d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"max_pool3d", (PyCFunction)functional::max_pool3d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"prelu", (PyCFunction)functional::prelu, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reshape", (PyCFunction)functional::reshape, METH_VARARGS | METH_KEYWORDS, NULL},
    {"view", (PyCFunction)functional::view, METH_VARARGS | METH_KEYWORDS, NULL},
    {"contiguous", (PyCFunction)functional::contiguous, METH_VARARGS | METH_KEYWORDS, NULL},
    {"contiguous_", (PyCFunction)functional::contiguous_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"slice_view_1d_contiguous", (PyCFunction)functional::slice_view_1d_contiguous, METH_VARARGS | METH_KEYWORDS, NULL},
    {"narrow", (PyCFunction)functional::narrow, METH_VARARGS | METH_KEYWORDS, NULL},
    {"slice", (PyCFunction)functional::slice, METH_VARARGS | METH_KEYWORDS, NULL},
    {"slice_update", (PyCFunction)functional::slice_update, METH_VARARGS | METH_KEYWORDS, NULL},
    {"copy", (PyCFunction)functional::copy, METH_VARARGS | METH_KEYWORDS, NULL},
    {"to", (PyCFunction)functional::to, METH_VARARGS | METH_KEYWORDS, NULL},
    {"flip", (PyCFunction)functional::flip, METH_VARARGS | METH_KEYWORDS, NULL},
    {"upsample", (PyCFunction)functional::upsample, METH_VARARGS | METH_KEYWORDS, NULL},
    {"upsample_linear_1d", (PyCFunction)functional::upsample_linear_1d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"upsample_nearest_1d", (PyCFunction)functional::upsample_nearest_1d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"upsample_nearest_2d", (PyCFunction)functional::upsample_nearest_2d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"upsample_bilinear_2d", (PyCFunction)functional::upsample_bilinear_2d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"upsample_bicubic_2d", (PyCFunction)functional::upsample_bicubic_2d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"upsample_nearest_3d", (PyCFunction)functional::upsample_nearest_3d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"upsample_trilinear_3d", (PyCFunction)functional::upsample_trilinear_3d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"abs", (PyCFunction)functional::abs, METH_VARARGS | METH_KEYWORDS, NULL},
    {"acos", (PyCFunction)functional::acos, METH_VARARGS | METH_KEYWORDS, NULL},
    {"acosh", (PyCFunction)functional::acosh, METH_VARARGS | METH_KEYWORDS, NULL},
    {"asin", (PyCFunction)functional::asin, METH_VARARGS | METH_KEYWORDS, NULL},
    {"asinh", (PyCFunction)functional::asinh, METH_VARARGS | METH_KEYWORDS, NULL},
    {"atan", (PyCFunction)functional::atan, METH_VARARGS | METH_KEYWORDS, NULL},
    {"atan2", (PyCFunction)functional::atan2, METH_VARARGS | METH_KEYWORDS, NULL},
    {"atanh", (PyCFunction)functional::atanh, METH_VARARGS | METH_KEYWORDS, NULL},
    {"ceil", (PyCFunction)functional::ceil, METH_VARARGS | METH_KEYWORDS, NULL},
    {"erf", (PyCFunction)functional::erf, METH_VARARGS | METH_KEYWORDS, NULL},
    {"erfc", (PyCFunction)functional::erfc, METH_VARARGS | METH_KEYWORDS, NULL},
    {"expm1", (PyCFunction)functional::expm1, METH_VARARGS | METH_KEYWORDS, NULL},
    {"floor", (PyCFunction)functional::floor, METH_VARARGS | METH_KEYWORDS, NULL},
    {"floor_", (PyCFunction)functional::floor_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"lgamma", (PyCFunction)functional::lgamma, METH_VARARGS | METH_KEYWORDS, NULL},
    {"log1p", (PyCFunction)functional::log1p, METH_VARARGS | METH_KEYWORDS, NULL},
    {"logsigmoid", (PyCFunction)functional::logsigmoid, METH_VARARGS | METH_KEYWORDS, NULL},
    {"rint", (PyCFunction)functional::rint, METH_VARARGS | METH_KEYWORDS, NULL},
    {"round", (PyCFunction)functional::round, METH_VARARGS | METH_KEYWORDS, NULL},
    {"sign", (PyCFunction)functional::sign, METH_VARARGS | METH_KEYWORDS, NULL},
    {"sinh", (PyCFunction)functional::sinh, METH_VARARGS | METH_KEYWORDS, NULL},
    {"softplus", (PyCFunction)functional::softplus, METH_VARARGS | METH_KEYWORDS, NULL},
    {"softshrink", (PyCFunction)functional::softshrink, METH_VARARGS | METH_KEYWORDS, NULL},
    {"one_hot", (PyCFunction)functional::one_hot, METH_VARARGS | METH_KEYWORDS, NULL},
    {"unsorted_segment_sum", (PyCFunction)functional::unsorted_segment_sum, METH_VARARGS | METH_KEYWORDS, NULL},
    {"tril", (PyCFunction)functional::tril, METH_VARARGS | METH_KEYWORDS, NULL},
    {"triu", (PyCFunction)functional::triu, METH_VARARGS | METH_KEYWORDS, NULL},
    {"triu_", (PyCFunction)functional::triu_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"clamp", (PyCFunction)functional::clamp, METH_VARARGS | METH_KEYWORDS, NULL},
    {"clamp_", (PyCFunction)functional::clamp_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"clamp_min", (PyCFunction)functional::clamp_min, METH_VARARGS | METH_KEYWORDS, NULL},
    {"clamp_min_", (PyCFunction)functional::clamp_min_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"clamp_max", (PyCFunction)functional::clamp_max, METH_VARARGS | METH_KEYWORDS, NULL},
    {"clamp_max_", (PyCFunction)functional::clamp_max_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"clip", (PyCFunction)functional::clip, METH_VARARGS | METH_KEYWORDS, NULL},
    {"clip_", (PyCFunction)functional::clip_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"vector_norm", (PyCFunction)functional::vector_norm, METH_VARARGS | METH_KEYWORDS, NULL},
    {"matrix_norm", (PyCFunction)functional::matrix_norm, METH_VARARGS | METH_KEYWORDS, NULL},
    {"norm", (PyCFunction)functional::norm, METH_VARARGS | METH_KEYWORDS, NULL},
    {"inv", (PyCFunction)functional::inv, METH_VARARGS | METH_KEYWORDS, NULL},
    {"linalg_cross", (PyCFunction)functional::linalg_cross, METH_VARARGS | METH_KEYWORDS, NULL},
    {"dropout", (PyCFunction)functional::dropout, METH_VARARGS | METH_KEYWORDS, NULL},
    {"dropout1d", (PyCFunction)functional::dropout1d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"dropout2d", (PyCFunction)functional::dropout2d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"dropout3d", (PyCFunction)functional::dropout3d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"pad", (PyCFunction)functional::pad, METH_VARARGS | METH_KEYWORDS, NULL},
    {"silu", (PyCFunction)functional::silu, METH_VARARGS | METH_KEYWORDS, NULL},
    {"mish", (PyCFunction)functional::mish, METH_VARARGS | METH_KEYWORDS, NULL},
    {"selu", (PyCFunction)functional::selu, METH_VARARGS | METH_KEYWORDS, NULL},
    {"softsign", (PyCFunction)functional::softsign, METH_VARARGS | METH_KEYWORDS, NULL},
    {"diag", (PyCFunction)functional::diag, METH_VARARGS | METH_KEYWORDS, NULL},
    {"diagonal", (PyCFunction)functional::diagonal, METH_VARARGS | METH_KEYWORDS, NULL},
    {"scatter", (PyCFunction)functional::scatter, METH_VARARGS | METH_KEYWORDS, NULL},
    {"scatter_add", (PyCFunction)functional::scatter_add, METH_VARARGS | METH_KEYWORDS, NULL},
    {"tensor_setitem", (PyCFunction)functional::tensor_setitem, METH_VARARGS | METH_KEYWORDS, NULL},
    {"avg_pool1d", (PyCFunction)functional::avg_pool1d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"avg_pool2d", (PyCFunction)functional::avg_pool2d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"avg_pool3d", (PyCFunction)functional::avg_pool3d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"minimum", (PyCFunction)functional::minimum, METH_VARARGS | METH_KEYWORDS, NULL},
    {"maximum", (PyCFunction)functional::maximum, METH_VARARGS | METH_KEYWORDS, NULL},
    {"stack", (PyCFunction)functional::stack, METH_VARARGS | METH_KEYWORDS, NULL},
    {"atleast_1d", (PyCFunction)functional::atleast_1d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"atleast_2d", (PyCFunction)functional::atleast_2d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"atleast_3d", (PyCFunction)functional::atleast_3d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"hstack", (PyCFunction)functional::hstack, METH_VARARGS | METH_KEYWORDS, NULL},
    {"vstack", (PyCFunction)functional::vstack, METH_VARARGS | METH_KEYWORDS, NULL},
    {"dstack", (PyCFunction)functional::dstack, METH_VARARGS | METH_KEYWORDS, NULL},
    {"column_stack", (PyCFunction)functional::column_stack, METH_VARARGS | METH_KEYWORDS, NULL},
    {"row_stack", (PyCFunction)functional::row_stack, METH_VARARGS | METH_KEYWORDS, NULL},
    {"to_global", (PyCFunction)functional::to_global, METH_VARARGS | METH_KEYWORDS, NULL},
    {"to_local", (PyCFunction)functional::to_local, METH_VARARGS | METH_KEYWORDS, NULL},
    {"stream_touch", (PyCFunction)functional::stream_touch, METH_VARARGS | METH_KEYWORDS, NULL},
    {"broadcast", (PyCFunction)functional::broadcast, METH_VARARGS | METH_KEYWORDS, NULL},
    {"local_all_reduce", (PyCFunction)functional::local_all_reduce, METH_VARARGS | METH_KEYWORDS, NULL},
    {"local_reduce", (PyCFunction)functional::local_reduce, METH_VARARGS | METH_KEYWORDS, NULL},
    {"select_top_n", (PyCFunction)functional::select_top_n, METH_VARARGS | METH_KEYWORDS, NULL},
    {"identity", (PyCFunction)functional::identity, METH_VARARGS | METH_KEYWORDS, NULL},
    {"amp_white_identity", (PyCFunction)functional::amp_white_identity, METH_VARARGS | METH_KEYWORDS, NULL},
    {"amp_black_identity", (PyCFunction)functional::amp_black_identity, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reshape_like", (PyCFunction)functional::reshape_like, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reduce_sum_like", (PyCFunction)functional::reduce_sum_like, METH_VARARGS | METH_KEYWORDS, NULL},
    {"rand", (PyCFunction)functional::rand, METH_VARARGS | METH_KEYWORDS, NULL},
    {"randn", (PyCFunction)functional::randn, METH_VARARGS | METH_KEYWORDS, NULL},
    {"randn_like", (PyCFunction)functional::randn_like, METH_VARARGS | METH_KEYWORDS, NULL},
    {"randint", (PyCFunction)functional::randint, METH_VARARGS | METH_KEYWORDS, NULL},
    {"randint_like", (PyCFunction)functional::randint_like, METH_VARARGS | METH_KEYWORDS, NULL},
    {"randperm", (PyCFunction)functional::randperm, METH_VARARGS | METH_KEYWORDS, NULL},
    {"unfold_tensor", (PyCFunction)functional::unfold_tensor, METH_VARARGS | METH_KEYWORDS, NULL},
    {"unfold", (PyCFunction)functional::unfold, METH_VARARGS | METH_KEYWORDS, NULL},
    {"fold", (PyCFunction)functional::fold, METH_VARARGS | METH_KEYWORDS, NULL},
    {"split", (PyCFunction)functional::split, METH_VARARGS | METH_KEYWORDS, NULL},
    {"unbind", (PyCFunction)functional::unbind, METH_VARARGS | METH_KEYWORDS, NULL},
    {"chunk", (PyCFunction)functional::chunk, METH_VARARGS | METH_KEYWORDS, NULL},
    {"split_like", (PyCFunction)functional::split_like, METH_VARARGS | METH_KEYWORDS, NULL},
    {"pairwise_distance", (PyCFunction)functional::pairwise_distance, METH_VARARGS | METH_KEYWORDS, NULL},
    {"cosine_similarity", (PyCFunction)functional::cosine_similarity, METH_VARARGS | METH_KEYWORDS, NULL},
    {"normalize", (PyCFunction)functional::normalize, METH_VARARGS | METH_KEYWORDS, NULL},
    {"fused_self_attention", (PyCFunction)functional::fused_self_attention, METH_VARARGS | METH_KEYWORDS, NULL},
    {"fused_scale_tril", (PyCFunction)functional::fused_scale_tril, METH_VARARGS | METH_KEYWORDS, NULL},
    {"fused_bias_add_gelu", (PyCFunction)functional::fused_bias_add_gelu, METH_VARARGS | METH_KEYWORDS, NULL},
    {"fused_bias_add_dropout", (PyCFunction)functional::fused_bias_add_dropout, METH_VARARGS | METH_KEYWORDS, NULL},
    {"fused_scale_mask_softmax", (PyCFunction)functional::fused_scale_mask_softmax, METH_VARARGS | METH_KEYWORDS, NULL},
    {"fused_scale_mask_softmax_dropout", (PyCFunction)functional::fused_scale_mask_softmax_dropout, METH_VARARGS | METH_KEYWORDS, NULL},
    {"fused_scale_tril_softmax_mask_scale", (PyCFunction)functional::fused_scale_tril_softmax_mask_scale, METH_VARARGS | METH_KEYWORDS, NULL},
    {"fused_multi_head_attention_inference", (PyCFunction)functional::fused_multi_head_attention_inference, METH_VARARGS | METH_KEYWORDS, NULL},
    {"send", (PyCFunction)functional::send, METH_VARARGS | METH_KEYWORDS, NULL},
    {"recv", (PyCFunction)functional::recv, METH_VARARGS | METH_KEYWORDS, NULL},
    {"batch_gather", (PyCFunction)functional::batch_gather, METH_VARARGS | METH_KEYWORDS, NULL},
    {"ctc_greedy_decoder", (PyCFunction)functional::ctc_greedy_decoder, METH_VARARGS | METH_KEYWORDS, NULL},
    {"nms", (PyCFunction)functional::nms, METH_VARARGS | METH_KEYWORDS, NULL},
    {"roi_align", (PyCFunction)functional::roi_align, METH_VARARGS | METH_KEYWORDS, NULL},
    {"meshgrid", (PyCFunction)functional::meshgrid, METH_VARARGS | METH_KEYWORDS, NULL},
    {"index_select", (PyCFunction)functional::index_select, METH_VARARGS | METH_KEYWORDS, NULL},
    {"decode_onerec", (PyCFunction)functional::decode_onerec, METH_VARARGS | METH_KEYWORDS, NULL},
    {"dot", (PyCFunction)functional::dot, METH_VARARGS | METH_KEYWORDS, NULL},
    {"fused_dot_feature_interaction", (PyCFunction)functional::fused_dot_feature_interaction, METH_VARARGS | METH_KEYWORDS, NULL},
    {"fused_cross_feature_interaction", (PyCFunction)functional::fused_cross_feature_interaction, METH_VARARGS | METH_KEYWORDS, NULL},
    {"tensor_buffer_to_tensor", (PyCFunction)functional::tensor_buffer_to_tensor, METH_VARARGS | METH_KEYWORDS, NULL},
    {"tensor_to_tensor_buffer", (PyCFunction)functional::tensor_to_tensor_buffer, METH_VARARGS | METH_KEYWORDS, NULL},
    {"gen_tensor_buffer", (PyCFunction)functional::gen_tensor_buffer, METH_VARARGS | METH_KEYWORDS, NULL},
    {"top_k", (PyCFunction)functional::top_k, METH_VARARGS | METH_KEYWORDS, NULL},
    {"in_top_k", (PyCFunction)functional::in_top_k, METH_VARARGS | METH_KEYWORDS, NULL},
    {"cumsum", (PyCFunction)functional::cumsum, METH_VARARGS | METH_KEYWORDS, NULL},
    {"cumprod", (PyCFunction)functional::cumprod, METH_VARARGS | METH_KEYWORDS, NULL},
    {"one_embedding_id_shuffle", (PyCFunction)functional::one_embedding_id_shuffle, METH_VARARGS | METH_KEYWORDS, NULL},
    {"one_embedding_embedding_shuffle", (PyCFunction)functional::one_embedding_embedding_shuffle, METH_VARARGS | METH_KEYWORDS, NULL},
    {"one_embedding_embedding_gradient_shuffle", (PyCFunction)functional::one_embedding_embedding_gradient_shuffle, METH_VARARGS | METH_KEYWORDS, NULL},
    {"one_embedding_lookup", (PyCFunction)functional::one_embedding_lookup, METH_VARARGS | METH_KEYWORDS, NULL},
    {"one_embedding_fused_lookup", (PyCFunction)functional::one_embedding_fused_lookup, METH_VARARGS | METH_KEYWORDS, NULL},
    {"one_embedding_fused_lookup_grad", (PyCFunction)functional::one_embedding_fused_lookup_grad, METH_VARARGS | METH_KEYWORDS, NULL},
    {"one_embedding_unique_key_value_pair", (PyCFunction)functional::one_embedding_unique_key_value_pair, METH_VARARGS | METH_KEYWORDS, NULL},
    {"one_embedding_embedding_put", (PyCFunction)functional::one_embedding_embedding_put, METH_VARARGS | METH_KEYWORDS, NULL},
    {"one_embedding_sgd_update", (PyCFunction)functional::one_embedding_sgd_update, METH_VARARGS | METH_KEYWORDS, NULL},
    {"one_embedding_adam_update", (PyCFunction)functional::one_embedding_adam_update, METH_VARARGS | METH_KEYWORDS, NULL},
    {"one_embedding_adagrad_update", (PyCFunction)functional::one_embedding_adagrad_update, METH_VARARGS | METH_KEYWORDS, NULL},
    {"one_embedding_ftrl_update", (PyCFunction)functional::one_embedding_ftrl_update, METH_VARARGS | METH_KEYWORDS, NULL},
    {"einsum", (PyCFunction)functional::einsum, METH_VARARGS | METH_KEYWORDS, NULL},
    {"pixel_shuffle", (PyCFunction)functional::pixel_shuffle, METH_VARARGS | METH_KEYWORDS, NULL},
    {"isnan", (PyCFunction)functional::isnan, METH_VARARGS | METH_KEYWORDS, NULL},
    {"isinf", (PyCFunction)functional::isinf, METH_VARARGS | METH_KEYWORDS, NULL},
    {"isfinite", (PyCFunction)functional::isfinite, METH_VARARGS | METH_KEYWORDS, NULL},
    {"roc_auc_score", (PyCFunction)functional::roc_auc_score, METH_VARARGS | METH_KEYWORDS, NULL},
    {"pin_memory", (PyCFunction)functional::pin_memory, METH_VARARGS | METH_KEYWORDS, NULL},
    {"fill_", (PyCFunction)functional::fill_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"rnn_tanh_cell", (PyCFunction)functional::rnn_tanh_cell, METH_VARARGS | METH_KEYWORDS, NULL},
    {"rnn_relu_cell", (PyCFunction)functional::rnn_relu_cell, METH_VARARGS | METH_KEYWORDS, NULL},
    {"lstm_cell", (PyCFunction)functional::lstm_cell, METH_VARARGS | METH_KEYWORDS, NULL},
    {"gru_cell", (PyCFunction)functional::gru_cell, METH_VARARGS | METH_KEYWORDS, NULL},
    {"rnn_tanh", (PyCFunction)functional::rnn_tanh, METH_VARARGS | METH_KEYWORDS, NULL},
    {"rnn_relu", (PyCFunction)functional::rnn_relu, METH_VARARGS | METH_KEYWORDS, NULL},
    {"lstm", (PyCFunction)functional::lstm, METH_VARARGS | METH_KEYWORDS, NULL},
    {"gru", (PyCFunction)functional::gru, METH_VARARGS | METH_KEYWORDS, NULL},
    {"pack_padded_sequence", (PyCFunction)functional::pack_padded_sequence, METH_VARARGS | METH_KEYWORDS, NULL},
    {"multi_tensor_sgd_update", (PyCFunction)functional::multi_tensor_sgd_update, METH_VARARGS | METH_KEYWORDS, NULL},
    {"multi_tensor_adam_update", (PyCFunction)functional::multi_tensor_adam_update, METH_VARARGS | METH_KEYWORDS, NULL},
    {"trunc", (PyCFunction)functional::trunc, METH_VARARGS | METH_KEYWORDS, NULL},
    {"batch_norm_stats", (PyCFunction)functional::batch_norm_stats, METH_VARARGS | METH_KEYWORDS, NULL},
    {"batch_norm_gather_stats_with_counts", (PyCFunction)functional::batch_norm_gather_stats_with_counts, METH_VARARGS | METH_KEYWORDS, NULL},
    {"batch_norm_elemt", (PyCFunction)functional::batch_norm_elemt, METH_VARARGS | METH_KEYWORDS, NULL},
    {"batch_norm_backward_reduce", (PyCFunction)functional::batch_norm_backward_reduce, METH_VARARGS | METH_KEYWORDS, NULL},
    {"batch_norm_backward_elemt", (PyCFunction)functional::batch_norm_backward_elemt, METH_VARARGS | METH_KEYWORDS, NULL},
    {"adaptive_max_pool1d", (PyCFunction)functional::adaptive_max_pool1d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"adaptive_max_pool2d", (PyCFunction)functional::adaptive_max_pool2d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"adaptive_max_pool3d", (PyCFunction)functional::adaptive_max_pool3d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"exponential_", (PyCFunction)functional::exponential_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"multinomial", (PyCFunction)functional::multinomial, METH_VARARGS | METH_KEYWORDS, NULL},
    {"deform_conv2d", (PyCFunction)functional::deform_conv2d, METH_VARARGS | METH_KEYWORDS, NULL},
    {"bincount", (PyCFunction)functional::bincount, METH_VARARGS | METH_KEYWORDS, NULL},

    {NULL, NULL, 0, NULL}
  };

  PyObject* module = m.ptr();
  if (module) {
    PyModule_AddFunctions(module, functions);
  }
}

}  // namespace oneflow
