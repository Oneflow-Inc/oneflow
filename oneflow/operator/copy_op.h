#ifndef ONEFLOW_OPERATOR_COPY_OP_H_
#define ONEFLOW_OPERATOR_COPY_OP_H_

#include "operator/operator.h"

namespace oneflow {

class CopyDataBlobDescSet final : public DataBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(CopyDataBlobDescSet);
  CopyDataBlobDescSet() = default;
  ~CopyDataBlobDescSet() = default;

  void Init(const google::protobuf::RepeatedPtrField<std::string>& lbns);

 private:
  std::vector<BlobDescriptor*> input_blobs_;
  std::vector<BlobDescriptor*> output_blobs_;

};

class CopyModelBlobDescSet final : public ModelBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(CopyModelBlobDescSet);
  CopyModelBlobDescSet() = default;
  ~CopyModelBlobDescSet() = default;

  void Init() {
    ModelBlobDescSet::Init();
  }

 private:

};

class CopyOp final : public Operator {
 public:
  DISALLOW_COPY_AND_MOVE(CopyOp);
  CopyOp() = default;
  ~CopyOp() = default;

  void Init(const OperatorConf& op_conf) override;

  std::string ibn2lbn(const std::string& input_blob_name) const override;
  std::string obn2lbn(const std::string& output_blob_name) const override;

  bool IsElemWise() const override { return false; }

 private:

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_COPY_OP_H_
