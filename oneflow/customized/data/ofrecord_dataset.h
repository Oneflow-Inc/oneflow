#ifndef ONEFLOW_CUSTOMIZED_DATA_OFRECORD_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_OFRECORD_DATASET_H_

#include "oneflow/customized/data/dataset.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

class OFRecordDataset final : public Dataset<TensorBuffer> {
 public:
  OFRecordDataset(user_op::KernelInitContext* ctx) { TODO(); }
  ~OFRecordDataset() = default;

  void Next(TensorBuffer& tensor) override {}

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_OFRECORD_DATASET_H_
