#ifndef ONEFLOW_OPERATOR_POOLING_OP_H_
#define ONEFLOW_OPERATOR_POOLING_OP_H_

#include "operator/operator.h"

namespace oneflow {

class PoolingDataBlobDescSet final : public DataBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(PoolingDataBlobDescSet);
  PoolingDataBlobDescSet() = default;
  ~PoolingDataBlobDescSet() = default;

  void Init() {
    DataBlobDescSet::Init();
    RegisterInputBlobPptr("in", &in_);
    RegisterInputDiffBlobPptr("in_diff", &in_diff_);
    RegisterOutputBlobPptr("out", &out_);
    RegisterOutputDiffBlobPptr("out_diff", &out_diff_);
    RegisterDataTmpBlobPptr("idx", &idx_);
  }

 private:
  BlobDescriptor* in_;
  BlobDescriptor* in_diff_;
  BlobDescriptor* out_;
  BlobDescriptor* out_diff_;
  BlobDescriptor* idx_;

};

class PoolingModelBlobDescSet final : public ModelBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(PoolingModelBlobDescSet);
  PoolingModelBlobDescSet() = default;
  ~PoolingModelBlobDescSet() = default;

  void Init() {
    ModelBlobDescSet::Init();
  }

 private:
};

class PoolingOp final : public Operator {
 public:
  DISALLOW_COPY_AND_MOVE(PoolingOp);
  PoolingOp() = default;
  ~PoolingOp() = default;

  void Init(const OperatorConf& op_conf) override;
  bool IsElemWise() const override { return false; }

 private:

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_POOLING_OP_H_
