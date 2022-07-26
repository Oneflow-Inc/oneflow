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
#include "oneflow/core/common/cached_functor_ptr.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/framework/nd_sbp.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class ImageFlipFuntor {
 public:
  ImageFlipFuntor() {
    op_ = CHECK_JUST(
        one::OpBuilder("image_flip").Input("in").Input("flip_code").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& flip_code) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, flip_code});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DecodeOneRecFunctor {
 public:
  DecodeOneRecFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("onerec_decoder").Input("in").Output("out").Build());
  }

  struct DecodeOneRec {
    Maybe<AttrMap> operator()(const std::string& key, const Symbol<DType>& dtype,
                              const Shape& shape, const bool is_dynamic, bool reshape_has_value,
                              const Shape& reshape_value, bool batch_padding_has_value,
                              const Shape& batch_padding_value) {
      MutableAttrMap attrs;
      bool has_reshape = false;
      bool has_batch_padding = false;

      if (reshape_has_value) {
        has_reshape = true;
        JUST(attrs.SetAttr<Shape>("reshape", reshape_value));
      } else {
        has_reshape = false;
        JUST(attrs.SetAttr<Shape>("reshape", shape));
      }

      if (batch_padding_has_value) {
        has_batch_padding = true;
        JUST(attrs.SetAttr<Shape>("batch_padding", batch_padding_value));
      } else {
        has_batch_padding = false;
        JUST(attrs.SetAttr<Shape>("batch_padding", shape));
      }
      JUST(attrs.SetAttr<std::string>("key", key));
      JUST(attrs.SetAttr<DataType>("data_type", dtype->data_type()));
      JUST(attrs.SetAttr<Shape>("static_shape", shape));
      JUST(attrs.SetAttr<bool>("is_dynamic", is_dynamic));
      JUST(attrs.SetAttr<bool>("has_reshape", has_reshape));
      JUST(attrs.SetAttr<bool>("has_batch_padding", has_batch_padding));
      return AttrMap(attrs);
    }
  };

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const std::string& key,
                           const Symbol<DType>& dtype, const Shape& shape, const bool is_dynamic,
                           const Optional<Shape>& reshape,
                           const Optional<Shape>& batch_padding) const {
    constexpr auto* GetAttrs = CACHED_FUNCTOR_PTR(DecodeOneRec);
    const auto attrs = *JUST(GetAttrs(key, dtype, shape, is_dynamic, reshape.has_value(),
                                      reshape.value_or(Shape()), batch_padding.has_value(),
                                      batch_padding.value_or(Shape())));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::ImageFlipFuntor>("ImageFlip");
  m.add_functor<impl::DecodeOneRecFunctor>("DecodeOneRec");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
