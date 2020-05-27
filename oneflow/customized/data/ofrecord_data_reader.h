#ifndef ONEFLOW_CUSTOMIZED_DATA_OFRECORD_DATA_READER_H_
#define ONEFLOW_CUSTOMIZED_DATA_OFRECORD_DATA_READER_H_

#include "oneflow/customized/data/data_reader.h"
#include "oneflow/customized/data/ofrecord_dataset.h"
#include "oneflow/customized/data/ofrecord_parser.h"
#include "oneflow/customized/data/random_shuffle_dataset.h"
#include "oneflow/customized/data/batch_dataset.h"
#include <iostream>

namespace oneflow {
namespace data {

class OFRecordDataReader final : public DataReader<TensorBuffer> {
 public:
  OFRecordDataReader(user_op::KernelInitContext* ctx) : DataReader<TensorBuffer>(ctx) {
    loader_.reset(new OFRecordDataset(ctx));
    parser_.reset(new OFRecordParser());
    if (ctx->Attr<bool>("random_shuffle")) {
      loader_.reset(new RandomShuffleDataset<TensorBuffer>(ctx, std::move(loader_)));
    }
    int32_t batch_size = ctx->TensorDesc4ArgNameAndIndex("out", 0)->shape().elem_cnt();
    loader_.reset(new BatchDataset<TensorBuffer>(batch_size, std::move(loader_)));
    StartLoadThread();
  }
  ~OFRecordDataReader() = default;

 protected:
  using DataReader<TensorBuffer>::loader_;
  using DataReader<TensorBuffer>::parser_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_OFRECORD_DATA_READER_H_
