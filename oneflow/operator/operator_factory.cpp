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

// It is ugly now, maybe we can find one more elegant implemention ?
std::shared_ptr<Operator> OperatorFactory::ConstructOp(
    const OperatorConf& op_conf) const {
  std::shared_ptr<Operator> ret;
  switch (op_conf.specified_type_case()) {
    case OperatorConf::kConvolutionConf: {
      ret.reset(new ConvolutionOp);
      break;
    }
    case OperatorConf::kInnerproductConf: {
      ret.reset(new InnerProductOp);
      break;
    }
    case OperatorConf::kDataLoaderConf: {
      ret.reset(new DataLoaderOp);
      break;
    }
    case OperatorConf::kPoolingConf: {
      ret.reset(new PoolingOp);
      break;
    }
    case OperatorConf::kReluConf: {
      ret.reset(new ReluOp);
      break;
    }
    case OperatorConf::kSoftmaxConf: {
      ret.reset(new SoftmaxOp);
      break;
    }
    case OperatorConf::kMultinomialLogisticLossConf: {
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
