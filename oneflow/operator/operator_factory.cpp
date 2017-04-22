#include "operator/operator_factory.h"
#include <functional>
#include "glog/logging.h"
#include "operator/convolution_op.h"
#include "operator/innerproduct_op.h"
#include "operator/data_loader_op.h"
#include "operator/multinomial_logistic_loss_op.h"
#include "operator/relu_op.h"
#include "operator/softmax_op.h"
#include "operator/pooling_op.h"
#include "operator/copy_op.h"
#include "operator/clone_op.h"
#include "operator/boxing_op.h"
#include "operator/model_load_op.h"
#include "operator/model_save_op.h"
#include "operator/model_update_op.h"
#include "operator/concat_op.h"

namespace oneflow {

std::shared_ptr<Operator> OpFactory::ConstructOp(
    const OperatorConf& op_conf) const {
  static HashMap<int, std::function<Operator*()>>
  op_type2new_op_func = {
    {OperatorConf::kConvolutionConf, []() { return new ConvolutionOp; }},
    {OperatorConf::kInnerproductConf, []() { return new InnerProductOp; }},
    {OperatorConf::kDataLoaderConf, []() { return new DataLoaderOp; }},
    {OperatorConf::kPoolingConf, []() { return new PoolingOp; }},
    {OperatorConf::kReluConf, []() { return new ReluOp; }},
    {OperatorConf::kSoftmaxConf, []() { return new SoftmaxOp; }},
    {OperatorConf::kMultinomialLogisticLossConf, []() { return new MultinomialLogisticLossOp; }},
    {OperatorConf::kCopyConf, []() { return new CopyOp; }},
    {OperatorConf::kCloneConf, []() { return new CloneOp; }},
    {OperatorConf::kBoxingConf, []() { return new BoxingOp; }},
    {OperatorConf::kModelUpdateConf, []() { return new ModelUpdateOp; }},
    {OperatorConf::kModelLoadConf, []() { return new ModelLoadOp; }},
    {OperatorConf::kModelSaveConf, []() { return new ModelSaveOp; }},
    {OperatorConf::kConcatConf, []() { return new ConcatOp; }},
  };
  std::shared_ptr<Operator> ret;
  ret.reset(op_type2new_op_func.at(op_conf.specified_type_case())());
  ret->Init(op_conf);
  return ret;
}

} // namespace oneflow
