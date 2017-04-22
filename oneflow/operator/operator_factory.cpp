#include "operator/operator_factory.h"
#include "glog/logging.h"
#include "operator/convolution_op.h"
#include "operator/innerproduct_op.h"
#include "operator/data_loader_op.h"
#include "operator/multinomial_logistic_loss_op.h"
#include "operator/relu_op.h"
#include "operator/softmax_op.h"
#include "operator/pooling_op.h"

namespace oneflow {

std::shared_ptr<Operator> OpFactory::ConstructOp(
    const OperatorConf& op_conf) const {
  static HashMap<int, std::function<Operator*()>>
  op_type2new_op_func = {
    {OperatorConf::kConvolutionConf, []() { return new ConvolutionOp; } },
    {OperatorConf::kInnerproductConf, []() { return new ConvolutionOp; } },
    {OperatorConf::kDataLoaderConf, []() { return new ConvolutionOp; } },
    {OperatorConf::kPoolingConf, []() { return new ConvolutionOp; } },
    {OperatorConf::kReluConf, []() { return new ConvolutionOp; } },
    {OperatorConf::kSoftmaxConf, []() { return new ConvolutionOp; } },
    {OperatorConf::kMultinomialLogisticLossConf, []() { return new ConvolutionOp; } },
    {OperatorConf::kCopyConf, []() { return new ConvolutionOp; } },
    {OperatorConf::kCloneConf, []() { return new ConvolutionOp; } },
    {OperatorConf::kBoxingConf, []() { return new ConvolutionOp; } },
    {OperatorConf::kModelUpdateConf, []() { return new ConvolutionOp; } },
    {OperatorConf::kModelLoadConf, []() { return new ConvolutionOp; } },
    {OperatorConf::kModelSaveConf, []() { return new ConvolutionOp; } },
    {OperatorConf::kConcatConf, []() { return new ConvolutionOp; } },
  };
  std::shared_ptr<Operator> ret;
  ret.reset(op_type2new_op_func.at(op_conf.specified_type_case())());
  ret->Init(op_conf);
  return ret;
}

} // namespace oneflow
