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
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
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
    MutableAttrMap attrs;
    bool has_reshape = false;
    bool has_batch_padding = false;

    if (reshape.has_value()) {
      has_reshape = true;
      JUST(attrs.SetAttr<Shape>("reshape", *JUST(reshape)));
    } else {
      has_reshape = false;
      JUST(attrs.SetAttr<Shape>("reshape", shape));
    }

    if (batch_padding.has_value()) {
      has_batch_padding = true;
      JUST(attrs.SetAttr<Shape>("batch_padding", *JUST(batch_padding)));
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

    return OpInterpUtil::Dispatch<Tensor>(*op_, {input}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ReadOneRecFunctor {
 public:
  ReadOneRecFunctor() { op_ = CHECK_JUST(one::OpBuilder("OneRecReader").Output("out").Build()); }

  Maybe<Tensor> operator()(const std::vector<std::string>& files, const int32_t batch_size,
                           const bool random_shuffle, const std::string& shuffle_mode,
                           const int32_t shuffle_buffer_size, const bool shuffle_after_epoch,
                           const bool verify_example,
                           const Optional<Symbol<ParallelDesc>>& placement,
                           const Optional<std::vector<Symbol<cfg::SbpParallel>>>& sbp) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<std::string>>("files", files));
    JUST(attrs.SetAttr<int32_t>("batch_size", batch_size));
    JUST(attrs.SetAttr<bool>("random_shuffle", random_shuffle));
    JUST(attrs.SetAttr<std::string>("shuffle_mode", shuffle_mode));
    JUST(attrs.SetAttr<int32_t>("shuffle_buffer_size", shuffle_buffer_size));
    JUST(attrs.SetAttr<bool>("shuffle_after_epoch", shuffle_after_epoch));
    JUST(attrs.SetAttr<bool>("verify_example", verify_example));

    if (placement.has_value()) {
      CHECK_OR_RETURN(sbp.has_value());
      const auto& nd_sbp = JUST(GetNdSbp(*JUST(sbp)));
      return OpInterpUtil::Dispatch<Tensor>(
          *op_, {}, one::OpExprInterpContext(attrs, JUST(placement), nd_sbp));
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_, {}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::ImageFlipFuntor>("ImageFlip");
  m.add_functor<impl::DecodeOneRecFunctor>("DecodeOneRec");
  m.add_functor<impl::ReadOneRecFunctor>("ReadOneRec");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
