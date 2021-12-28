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
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_generated.h"
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
    bool has_reshape = false;
    bool has_batch_padding = false;

    auto ctx = std::make_shared<schema::OnerecDecoderOp>();
    if (reshape.has_value()) {
      has_reshape = true;
      ctx->set_reshape(*JUST(reshape));
    } else {
      has_reshape = false;
      ctx->set_reshape(shape);
    }

    if (batch_padding.has_value()) {
      has_batch_padding = true;
      ctx->set_batch_padding(*JUST(batch_padding));
    } else {
      has_batch_padding = false;
      ctx->set_batch_padding(shape);
    }
    ctx->set_key(key);
    ctx->set_data_type(dtype->data_type());
    ctx->set_static_shape(shape);
    ctx->set_is_dynamic(is_dynamic);
    ctx->set_has_reshape(has_reshape);
    ctx->set_has_batch_padding(has_batch_padding);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input}, ctx);
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
    auto ctx = std::make_shared<schema::OneRecReaderOp>();
    ctx->set_files(files);
    ctx->set_batch_size(batch_size);
    ctx->set_random_shuffle(random_shuffle);
    ctx->set_shuffle_mode(shuffle_mode);
    ctx->set_shuffle_buffer_size(shuffle_buffer_size);
    ctx->set_shuffle_after_epoch(shuffle_after_epoch);
    ctx->set_verify_example(verify_example);
    if (placement.has_value()) {
      JUST(CheckDeviceIdsIsValid(JUST(placement)));
      CHECK_OR_RETURN(sbp.has_value())
          << "placement is not None, but sbp is None. It's not allowed.";
      return OpInterpUtil::Dispatch<Tensor>(
          *op_, {}, OpExprInterpContext(ctx, JUST(placement), JUST(GetNdSbp(*JUST(sbp)))));
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {}, ctx);
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
