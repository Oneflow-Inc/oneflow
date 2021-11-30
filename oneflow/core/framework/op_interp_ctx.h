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
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"

namespace oneflow {

namespace user_op {
class OpKernelState;
}  // namespace user_op

class OpInterpCtx {
 public:
  template<typename T>
  Maybe<const T&> GetAttr(const char* attr_name) const {
    return *reinterpret_cast<const T*>(JUST(GetAttr(attr_name)));
  }

  virtual Maybe<const void*> GetAttr(const char* attr_name) const = 0;

  size_t hash_value() const {
    // TODO(hjchen2)
    return 0;
  }

 public:
  Optional<Symbol<Device>> device;               // for local op
  Optional<Symbol<ParallelDesc>> parallel_desc;  // for consistent op
  Optional<Symbol<cfg::NdSbp>> nd_sbp;           // for consistent op
  Optional<user_op::OpKernelState> state;
};

class FakeOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<const void*> GetAttr(const char* attr_name) const override {
    return Error::RuntimeError() << "Should not access attribute for `FakeOpInterpCtx`.";
  }
};

#ifndef DEFINE_OP_INTERP_CTX_CLASS
#define DEFINE_OP_INTERP_CTX_CLASS
#include "oneflow/core/framework/op_interp_ctx_generated.h"
#endif  // DEFINE_OP_INTERP_CTX_CLASS
#undef DEFINE_OP_INTERP_CTX_CLASS

class CastToConsistentOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<const void*> GetAttr(const char* attr_name) const override {
    if (!strcmp(attr_name, "shape")) {
      return (const void*)&shape;
    } else if (!strcmp(attr_name, "dtype")) {
      return (const void*)&dtype;
    } else {
      return Error::RuntimeError() << "CastToConsistent op has no attribute named " << attr_name;
    }
  }

 public:
  Shape shape;
  DataType dtype;
};

class SelectTopNOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<const void*> GetAttr(const char* attr_name) const override {
    if (!strcmp(attr_name, "top_n")) {
      return (const void*)&top_n;
    } else {
      return Error::RuntimeError() << "SelectTopN op has no attribute named " << attr_name;
    }
  }

 public:
  int32_t top_n;
};

class FeedInputOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<const void*> GetAttr(const char* attr_name) const override {
    return Error::RuntimeError() << "FeedInput op has no attribute named " << attr_name;
  }
};

class FetchOutputOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<const void*> GetAttr(const char* attr_name) const override {
    return Error::RuntimeError() << "FetchOutput op has no attribute named " << attr_name;
  }
};

class FeedVariableOpInterpCtx : public OpInterpCtx {
 public:
  Maybe<const void*> GetAttr(const char* attr_name) const override {
    if (!strcmp(attr_name, "l2")) {
      return (const void*)&l2;
    } else {
      return Error::RuntimeError() << "FeedVariable op has no attribute named " << attr_name;
    }
  }

 public:
  double l2;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERP_CTX_H_
