#ifndef OPERATOR_MULTINOMIAL_LOGISTIC_LOSS_OP_H_
#define OPERATOR_MULTINOMIAL_LOGISTIC_LOSS_OP_H_

#include "operator/operator.h"

namespace oneflow {

// MLLoss = MultinomialLogisticLoss
class MLLossDataBlobDescSet final : public DataBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(MLLossDataBlobDescSet);
  MLLossDataBlobDescSet() = default;
  ~MLLossDataBlobDescSet() = default;

  void Init();

 private:
  BlobDescriptor* data_;
  BlobDescriptor* data_diff_;
  BlobDescriptor* label_;
  BlobDescriptor* label_diff_;
  BlobDescriptor* loss_;
  BlobDescriptor* loss_buffer_;

};

class MLLossModelBlobDescSet final : public ModelBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(MLLossModelBlobDescSet);
  MLLossModelBlobDescSet() = default;
  ~MLLossModelBlobDescSet() = default;
  
  void Init() {
    ModelBlobDescSet::Init();
  }

 private:

};

class MultinomialLogisticLossOp : public Operator {
 public:
  DISALLOW_COPY_AND_MOVE(MultinomialLogisticLossOp);
  MultinomialLogisticLossOp() = default;
  ~MultinomialLogisticLossOp() = default;

  void Init(const OperatorConf& op_conf) override;
  bool IsElemWise() const override { return false; }

 private:

};

} // namespace oneflow

#endif // OPERATOR_MULTINOMIAL_LOGISTIC_LOSS_OP_H_
