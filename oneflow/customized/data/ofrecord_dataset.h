#ifndef ONEFLOW_CUSTOMIZED_DATA_OFRECORD_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_OFRECORD_DATASET_H_

#include "oneflow/customized/data/dataset.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

class OFRecordDataset final : public Dataset<TensorBuffer> {
 public:
  using LoadTargetPtr = std::shared_ptr<TensorBuffer>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  OFRecordDataset(user_op::KernelInitContext* ctx) {
    int32_t batch_size = ctx->GetAttr<int32_t>("batch_size");
    int64_t total_empty_size = 2 * 2 * batch_size;  // maybe 2 * batch_size
    int64_t tensor_init_bytes = ctx->GetAttr<int64_t>("tensor_init_bytes");
  }
  ~OFRecordDataset() = default;

  LoadTargetPtrList Next() override { TODO(); }

 private:
  std::unique_ptr<EmptyTensorManager<TensorBuffer>> empty_tensor_mgr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_OFRECORD_DATASET_H_
