#ifndef ONEFLOW_OPERATOR_SOFTMAX_OP_H_
#define ONEFLOW_OPERATOR_SOFTMAX_OP_H_

#include "operator/operator.h"

namespace oneflow {

class SoftmaxDataBlobDescSet final : public DataBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(SoftmaxDataBlobDescSet);
  SoftmaxDataBlobDescSet() = default;
  ~SoftmaxDataBlobDescSet() = default;

  void Init(const std::string& op_name);

 private:
  BlobDescriptor* in_;
  BlobDescriptor* in_diff_;
  BlobDescriptor* out_;
  BlobDescriptor* out_diff_;

};

class SoftmaxModelBlobDescSet final : public ModelBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(SoftmaxModelBlobDescSet);
  SoftmaxModelBlobDescSet() = default;
  ~SoftmaxModelBlobDescSet() = default;

  void Init(const std::string& op_name) {
    ModelBlobDescSet::Init();
  }

 private:
};

class SoftmaxOp : public Operator {
 public:
  DISALLOW_COPY_AND_MOVE(SoftmaxOp);
  SoftmaxOp() = default;
  ~SoftmaxOp() = default;

  void Init(const OperatorConf& op_conf) override;
  bool IsElemWise() const override { return false; }

 private:
  SoftmaxOpConf op_conf_;

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_SOFTMAX_OP_H_
