#ifndef OPERATOR_CONVOLUTION_OP_H_
#define OPERATOR_CONVOLUTION_OP_H_

#include "operator/operator.h"

namespace oneflow {

class ConvolutionDataBlobDescSet final : public DataBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(ConvolutionDataBlobDescSet);
  ConvolutionDataBlobDescSet() = default;
  ~ConvolutionDataBlobDescSet() = default;

  void Init(const std::string& op_name);

 private:
  BlobDescriptor* in_;
  BlobDescriptor* in_diff_;
  BlobDescriptor* out_;
  BlobDescriptor* out_diff_;
  BlobDescriptor* col_buf_;

};

class ConvolutionModelBlobDescSet final : public ModelBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(ConvolutionModelBlobDescSet);
  ConvolutionModelBlobDescSet() = default;
  ~ConvolutionModelBlobDescSet() = default;

  void Init(const std::string& op_name);

 private:
  BlobDescriptor* weight_;
  BlobDescriptor* weight_diff_;
  BlobDescriptor* bias_;
  BlobDescriptor* bias_diff_;
  BlobDescriptor* bias_multiplier_;
};

class ConvolutionOp final : public Operator {
 public:
  DISALLOW_COPY_AND_MOVE(ConvolutionOp);
  ConvolutionOp() = default;
  ~ConvolutionOp() = default;

  void Init(const OperatorConf& op_conf) override;
  bool IsElemWise() const override { return false; }

 private:

};

} // namespace oneflow

#endif // OPERATOR_CONVOLUTION_OP_H_
