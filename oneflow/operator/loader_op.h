#ifndef OPERATOR_LOADER_OP_H_
#define OPERATOR_LOADER_OP_H_

#include "operator/operator.h"

namespace oneflow {

class LoaderDataBlobDescSet final : public DataBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(LoaderDataBlobDescSet);
  LoaderDataBlobDescSet() = default;
  ~LoaderDataBlobDescSet() = default;

  void Init() {
    DataBlobDescSet::Init();
    RegisterOutputBlobPptr("data", &data_);
    RegisterOutputBlobPptr("label", &label_);
  }

 private:
  BlobDescriptor* data_;
  BlobDescriptor* label_;

};

class LoaderModelBlobDescSet final : public ModelBlobDescSet {
 public:
  DISALLOW_COPY_AND_MOVE(LoaderModelBlobDescSet);
  LoaderModelBlobDescSet() = default;
  ~LoaderModelBlobDescSet() = default;

  void Init() {
    ModelBlobDescSet::Init();
  }

 private:
};

class LoaderOp final : public Operator {
 public:
  DISALLOW_COPY_AND_MOVE(LoaderOp);
  LoaderOp() = default;
  ~LoaderOp() = default;
  
  void Init(const OperatorConf& op_conf) override;
  bool IsElemWise() const override { return false; }

 private:
  LoaderOpConf op_conf_;

};

} // namespace oneflow

#endif // OPERATOR_LOADER_OP_H_
