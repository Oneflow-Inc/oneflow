#ifndef ONEFLOW_CUSTOMIZED_DATA_OFRECORD_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_OFRECORD_DATASET_H_

#include "oneflow/customized/data/dataset.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

class OFRecordDataSet final : public DataSet<TensorBuffer> {
 public:
  OFRecordDataSet(user_op::KernelInitContext* ctx) { TODO(); }
  ~OFRecordDataSet() = default;

  void Next(TensorBuffer& tensor) override {}

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_OFRECORD_DATASET_H_
