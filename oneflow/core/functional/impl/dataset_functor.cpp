#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/scalar.h"

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
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in,
                           const std::shared_ptr<one::Tensor>& flip_code) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {in, flip_code});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) { m.add_functor<impl::ImageFlipFuntor>("ImageFlip"); };

}  // namespace functional
}  // namespace one
}  // namespace oneflow
