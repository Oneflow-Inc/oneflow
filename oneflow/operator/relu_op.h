#ifndef ONEFLOW_OPERATOR_RELU_OP_H_
#define ONEFLOW_OPERATOR_RELU_OP_H_

#include "operator/operator.h"

namespace oneflow {

class ReluDataBlobDescSet final : public DataBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(ReluDataBlobDescSet);
  ReluDataBlobDescSet() = default;
  ~ReluDataBlobDescSet() = default;

  void Init();

 private:
  BlobDescriptor* in_;
  BlobDescriptor* in_diff_;
  BlobDescriptor* out_;
  BlobDescriptor* out_diff_;

};

class ReluModelBlobDescSet final : public ModelBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(ReluModelBlobDescSet);
  ReluModelBlobDescSet() = default;
  ~ReluModelBlobDescSet() = default;

  void Init() {
    ModelBlobDescSet::Init();
  }

 private:
};

class ReluOp final : public Operator {
 public:
  DISALLOW_COPY_AND_MOVE(ReluOp);
  ReluOp() = default;
  ~ReluOp() = default;

  void Init(const OperatorConf& op_conf) override;
  bool IsElemWise() const override { return true; }

 private:
  ReluOpConf op_conf_;

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_RELU_OP_H_
