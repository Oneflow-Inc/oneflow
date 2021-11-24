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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INTERP_CTX_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INTERP_CTX_H_

#include <string>
#include <vector>

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

class OpInterpCtx {
 public:
  template<typename T>
  Maybe<const T&> GetAttr(const char* attr_name) const {
    return *reinterpret_cast<const T*>(JUST(GetAttr(attr_name)));
  }

  virtual Maybe<const void*> GetAttr(const char* attr_name) const = 0;

 public:
  Optional<Symbol<Device>> device;               // for local op
  Optional<Symbol<ParallelDesc>> parallel_desc;  // for consistent op
  Optional<Symbol<cfg::NdSbp>> nd_sbp;           // for consistent op
  Optional<user_op::OpKernelState> state;
};

class DefaultOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<const void*> GetAttr(const char* attr_name) const override {
    return Error::RuntimeError() << "Should not access attribute for `DefaultOpInterpCtx`.";
  }
};

class ConstantOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<const void*> GetAttr(const char* attr_name) const override {
    if (!strcmp(attr_name, "shape")) {
      return (const void*)&shape;
    } else if (!strcmp(attr_name, "dtype")) {
      return (const void*)&dtype;
    } else if (!strcmp(attr_name, "is_floating_value")) {
      return (const void*)&is_floating_value;
    } else if (!strcmp(attr_name, "integer_value")) {
      return (const void*)&integer_value;
    } else if (!strcmp(attr_name, "floating_value")) {
      return (const void*)&floating_value;
    } else if (!strcmp(attr_name, "nd_sbp")) {
      return (const void*)&nd_sbp;
    } else {
      return Error::RuntimeError() << "Constant op has no attribute named " << attr_name;
    }
  }

 public:
  Shape shape;
  DataType dtype;
  bool is_floating_value;
  int64_t integer_value;
  double floating_value;
  std::vector<std::string> nd_sbp;
};

class ReshapeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<const void*> GetAttr(const char* attr_name) const override {
    if (!strcmp(attr_name, "shape")) {
      return (const void*)&shape;
    } else {
      return Error::RuntimeError() << "Reshape op has no attribute named " << attr_name;
    }
  }

 public:
  Shape shape;
};

class CastOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<const void*> GetAttr(const char* attr_name) const override {
    if (!strcmp(attr_name, "dtype")) {
      return (const void*)&dtype;
    } else {
      return Error::RuntimeError() << "Cast op has no attribute named " << attr_name;
    }
  }

 public:
  DataType dtype;
};

class ScalarLogicalOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<const void*> GetAttr(const char* attr_name) const override {
    if (!strcmp(attr_name, "float_operand")) {
      return (const void*)&float_operand;
    } else if (!strcmp(attr_name, "int_operand")) {
      return (const void*)&int_operand;
    } else if (!strcmp(attr_name, "has_float_operand")) {
      return (const void*)&has_float_operand;
    } else if (!strcmp(attr_name, "has_int_operand")) {
      return (const void*)&has_int_operand;
    } else {
      return Error::RuntimeError() << "ScalarLogical op has no attribute named " << attr_name;
    }
  }

 public:
  double float_operand;
  int64_t int_operand;
  bool has_float_operand;
  bool has_int_operand;
};

class ArgWhereOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<const void*> GetAttr(const char* attr_name) const override {
    if (!strcmp(attr_name, "dtype")) {
      return (const void*)&dtype;
    } else {
      return Error::RuntimeError() << "ArgWhere op has no attribute named " << attr_name;
    }
  }

 public:
  DataType dtype;
};

class SliceOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<const void*> GetAttr(const char* attr_name) const override {
    if (!strcmp(attr_name, "start")) {
      return (const void*)&start;
    } else if (!strcmp(attr_name, "stop")) {
      return (const void*)&stop;
    } else if (!strcmp(attr_name, "step")) {
      return (const void*)&step;
    } else {
      return Error::RuntimeError() << "Slice op has no attribute named " << attr_name;
    }
  }

 public:
  std::vector<int64_t> start;
  std::vector<int64_t> stop;
  std::vector<int64_t> step;
};

class FlattenOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<const void*> GetAttr(const char* attr_name) const override {
    if (!strcmp(attr_name, "start_dim")) {
      return (const void*)&start_dim;
    } else if (!strcmp(attr_name, "end_dim")) {
      return (const void*)&end_dim;
    } else {
      return Error::RuntimeError() << "Flatten op has no attribute named " << attr_name;
    }
  }

 public:
  int32_t start_dim;
  int32_t end_dim;
};

class ReduceOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<const void*> GetAttr(const char* attr_name) const override {
    if (!strcmp(attr_name, "axis")) {
      return (const void*)&axis;
    } else if (!strcmp(attr_name, "keepdims")) {
      return (const void*)&keepdims;
    } else {
      return Error::RuntimeError() << "Reduce op has no attribute named " << attr_name;
    }
  }

 public:
  std::vector<int32_t> axis;
  bool keepdims;
};

class ReduceMinOpInterpCtx : public ReduceOpInterpCtx {};
class ReduceMaxOpInterpCtx : public ReduceOpInterpCtx {};
class ReduceSumOpInterpCtx : public ReduceOpInterpCtx {};

class ConcatOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<const void*> GetAttr(const char* attr_name) const override {
    if (!strcmp(attr_name, "axis")) {
      return (const void*)&axis;
    } else if (!strcmp(attr_name, "max_dim_size")) {
      return (const void*)&max_dim_size;
    } else {
      return Error::RuntimeError() << "Concat op has no attribute named " << attr_name;
    }
  }

 public:
  int64_t axis;
  int64_t max_dim_size;
};

class ExpandDimsOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<const void*> GetAttr(const char* attr_name) const override {
    if (!strcmp(attr_name, "axis")) {
      return (const void*)&axis;
    } else {
      return Error::RuntimeError() << "ExpandDims op has no attribute named " << attr_name;
    }
  }

 public:
  int32_t axis;
};

class MatMulOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<const void*> GetAttr(const char* attr_name) const override {
    if (!strcmp(attr_name, "transpose_a")) {
      return (const void*)&transpose_a;
    } else if (!strcmp(attr_name, "transpose_b")) {
      return (const void*)&transpose_b;
    } else if (!strcmp(attr_name, "alpha")) {
      return (const void*)&alpha;
    } else {
      return Error::RuntimeError() << "MatMul op has no attribute named " << attr_name;
    }
  }

 public:
  bool transpose_a;
  bool transpose_b;
  double alpha;
};

class BatchMatMulOpInterpCtx : public MatMulOpInterpCtx {};

class BernoulliOpInterpCtx : public OpInterpCtx {
 public:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERP_CTX_H_
