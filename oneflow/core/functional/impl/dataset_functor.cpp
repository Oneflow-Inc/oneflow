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
#include "oneflow/core/framework/mutable_attr_map.h"
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

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const std::string& key,
                           const Symbol<DType>& dtype, const Shape& shape, const bool is_dynamic,
                           const Optional<Shape>& reshape,
                           const Optional<Shape>& batch_padding) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("reshape", "batch_padding", "key", "data_type",
                                                 "static_shape", "is_dynamic", "has_reshape",
                                                 "has_batch_padding");
    bool has_reshape = reshape.has_value();
    bool has_batch_padding = batch_padding.has_value();
    attrs.SetAllAttrs(has_reshape ? *JUST(reshape) : shape,
                      has_batch_padding ? *JUST(batch_padding) : shape, key, dtype->data_type(),
                      shape, is_dynamic, has_reshape, has_batch_padding);
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
