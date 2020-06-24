#ifndef ONEFLOW_CUSTOMIZED_DATA_ONEREC_PARSER_H_
#define ONEFLOW_CUSTOMIZED_DATA_ONEREC_PARSER_H_

#include "oneflow/customized/data/parser.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {
namespace data {

class OneRecParser final : public Parser<TensorBuffer> {
 public:
  using LoadTargetPtr = std::shared_ptr<TensorBuffer>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  OneRecParser() = default;
  ~OneRecParser() = default;

  void Parse(std::shared_ptr<LoadTargetPtrList> batch_data,
             user_op::KernelComputeContext* ctx) override {
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    FOR_RANGE(int32_t, i, 0, batch_data->size()) {
      TensorBuffer* out = out_tensor->mut_dptr<TensorBuffer>() + i;
      TensorBuffer* tensor = batch_data->at(i).get();
      out->Resize(tensor->shape(), tensor->data_type());
      std::memcpy(out->mut_data(), tensor->data(), out->nbytes());
    }
  }
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_ONEREC_PARSER_H_
