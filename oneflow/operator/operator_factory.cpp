#include "operator/operator_factory.h"
#include "glog/logging.h"
#include "operator/convolution_op.h"
#include "operator/innerproduct_op.h"
#include "operator/loader_op.h"
#include "operator/multinomial_logistic_loss_op.h"
#include "operator/relu_op.h"
#include "operator/softmax_op.h"
#include "operator/pooling_op.h"

namespace oneflow {

// It is ugly now, maybe we can find one more elegant implemention ?
std::unique_ptr<Operator> OperatorFactory::ConstructOp(
    const OperatorConf& op_conf) const {
  std::unique_ptr<Operator> ret;
  switch (op_conf.specified_type_case()) {
    case OperatorConf::kConvolutionOpConf: {
      ret.reset(new ConvolutionOp);
      break;
    }
    case OperatorConf::kInnerProductOpConf: {
      ret.reset(new InnerProductOp);
      break;
    }
    case OperatorConf::kLoaderOpConf: {
      ret.reset(new LoaderOp);
      break;
    }
    case OperatorConf::kPoolingOpConf: {
      ret.reset(new PoolingOp);
      break;
    }
    case OperatorConf::kReluOpConf: {
      ret.reset(new ReluOp);
      break;
    }
    case OperatorConf::kSoftmaxOpConf: {
      ret.reset(new SoftmaxOp);
      break;
    }
    case OperatorConf::kMultinomialLogisticLossOpConf: {
      ret.reset(new MultinomialLogisticLossOp);
      break;
    }
    default: {
      LOG(FATAL) << "unknow op";
      break;
    }
  }
  ret->Init(op_conf);
  return ret;
}

} // namespace oneflow
