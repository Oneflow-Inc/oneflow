#ifndef ONEFLOW_CUSTOMIZED_DATA_OFRECORD_DATA_READER_H_
#define ONEFLOW_CUSTOMIZED_DATA_OFRECORD_DATA_READER_H_

#include "oneflow/customized/data/data_reader.h"
#include "oneflow/customized/data/ofrecord_dataset.h"
#include "oneflow/customized/data/ofrecord_parser.h"
#include "oneflow/customized/data/random_shuffle_dataset.h"

namespace oneflow {

class OFRecordDataReader final : public DataReader<TensorBuffer> {
 public:
  OFRecordDataReader(user_op::KernelInitContext* ctx) : DataReader<TensorBuffer>(ctx) {
    loader_.reset(new OFRecordDataSet(ctx));
    parser_.reset(new OFRecordParser());
    if (ctx->GetAttr<bool>("random_shuffle")) {
      // std::unique_ptr<RandomShuffleDataSet<TensorBuffer>> random_shuffle(
      //    new RandomShuffleDataSet<TensorBuffer>(ctx, std::move(loader_)));
      loader_.reset(new RandomShuffleDataSet<TensorBuffer>(ctx, std::move(loader_)));
    } else {
      TODO();
    }
  }
  ~OFRecordDataReader() = default;

 protected:
  using DataReader<TensorBuffer>::loader_;
  using DataReader<TensorBuffer>::parser_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_OFRECORD_DATA_READER_H_
