#ifndef ONEFLOW_CUSTOMIZED_DATA_OFRECORD_PARSER_H_
#define ONEFLOW_CUSTOMIZED_DATA_OFRECORD_PARSER_H_

#include "oneflow/customized/data/parser.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {
namespace data {

class OFRecordParser final : public Parser<TensorBuffer> {
 public:
  using LoadTargetPtr = std::shared_ptr<TensorBuffer>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  OFRecordParser() = default;
  ~OFRecordParser() = default;

  void Parse(std::shared_ptr<LoadTargetPtrList> batch_data,
             user_op::KernelComputeContext* ctx) override {
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    OFRecord* dptr = out_tensor->mut_dptr<OFRecord>();
    MultiThreadLoop(batch_data->size(), [&](size_t i) {
      TensorBuffer* buffer = batch_data->at(i).get();
      CHECK(dptr[i].ParseFromArray(buffer->data<char>(), buffer->shape().elem_cnt()));
    });
    if (batch_data->size() != out_tensor->shape().elem_cnt()) {
      CHECK_EQ(out_tensor->mut_shape()->NumAxes(), 1);
      out_tensor->mut_shape()->Set(0, batch_data->size());
    }
  }
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_OFRECORD_PARSER_H_
