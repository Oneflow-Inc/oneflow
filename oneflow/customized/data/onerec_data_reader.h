#ifndef ONEFLOW_CUSTOMIZED_DATA_ONEREC_DATA_READER_H_
#define ONEFLOW_CUSTOMIZED_DATA_ONEREC_DATA_READER_H_

#include "oneflow/customized/data/data_reader.h"
#include "oneflow/customized/data/onerec_dataset.h"
#include "oneflow/customized/data/onerec_parser.h"
#include "oneflow/customized/data/random_shuffle_dataset.h"
#include "oneflow/customized/data/batch_dataset.h"
#include <iostream>

namespace oneflow {
namespace data {

class OneRecDataReader final : public DataReader<TensorBuffer> {
 public:
  OneRecDataReader(user_op::KernelInitContext* ctx) : DataReader<TensorBuffer>(ctx) {
    loader_.reset(new OneRecDataset(ctx));
    parser_.reset(new OneRecParser());
    if (ctx->Attr<bool>("random_shuffle")) {
      loader_.reset(new RandomShuffleDataset<TensorBuffer>(ctx, std::move(loader_)));
    }
    int32_t batch_size = ctx->TensorDesc4ArgNameAndIndex("out", 0)->shape().elem_cnt();
    loader_.reset(new BatchDataset<TensorBuffer>(batch_size, std::move(loader_)));
    StartLoadThread();
  }
  ~OneRecDataReader() = default;

 protected:
  using DataReader<TensorBuffer>::loader_;
  using DataReader<TensorBuffer>::parser_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_ONEREC_DATA_READER_H_
