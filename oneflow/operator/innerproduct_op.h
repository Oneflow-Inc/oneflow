#ifndef OPERATOR_INNERPRODUCT_OP_H_
#define OPERATOR_INNERPRODUCT_OP_H_

#include "operator/operator.h"

namespace oneflow {

class InnerProductDataBlobDescSet final : public DataBlobDescSet {
 public:
   DISALLOW_COPY_AND_MOVE(InnerProductDataBlobDescSet);
   InnerProductDataBlobDescSet() = default;
   ~InnerProductDataBlobDescSet() = default;

   void Init(const std::string& op_name);

 private:
   BlobDescriptor* in_;
   BlobDescriptor* in_diff_;
   BlobDescriptor* out_;
   BlobDescriptor* out_diff_;
};

class InnerProductModelBlobDescSet final : public ModelBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(InnerProductModelBlobDescSet);
  InnerProductModelBlobDescSet() = default;
  ~InnerProductModelBlobDescSet() = default;
  
  void Init(const std::string& op_name);

 private:
  BlobDescriptor* weight_;
  BlobDescriptor* weight_diff_;
  BlobDescriptor* bias_;
  BlobDescriptor* bias_diff_;
  BlobDescriptor* bias_multiplier_;

};

class InnerProductOp final : public Operator {
 public:
  DISALLOW_COPY_AND_MOVE(InnerProductOp);
  InnerProductOp() = default;
  ~InnerProductOp() = default;

  void Init(const OperatorConf& op_conf) override;
  bool IsElemWise() const override { return false; }

 private:
  InnerProductOpConf op_conf_;

};

} // namespace oneflow

#endif // OPERATOR_INNERPRODUCT_OP_H_
