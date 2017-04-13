#ifndef OPERATOR_DATA_LOADER_OP_H_
#define OPERATOR_DATA_LOADER_OP_H_

#include "operator/operator.h"

namespace oneflow {

class DataLoaderOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataLoaderOp);
  DataLoaderOp() = default;
  ~DataLoaderOp() = default;
  
  void Init(const OperatorConf& op_conf) override;

  void InferBlobDesc4ObAndDtbFromIb() const override { TODO(); }

 private:

};

} // namespace oneflow

#endif // OPERATOR_DATA_LOADER_OP_H_
