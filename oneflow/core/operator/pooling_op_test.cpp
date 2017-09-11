#include "oneflow/core/operator/pooling_op.h"

namespace oneflow {

template<typename T>
std::shared_ptr<Operator> GetTestPoolingOp() {
  JobConf job_conf;
  job_conf.set_default_data_type(GetDataType<T>::val);
  JobDesc::Singleton()->InitFromJobConf(job_conf);
  OperatorConf op_conf;
  op_conf.set_name("pooling_test");
  op_conf.mutable_pooling_conf()->set_in("pooling_in");
  op_conf.mutable_pooling_conf()->set_out("pooling_out");
  op_conf.mutable_pooling_conf()->set_pool(PoolingOpConf_PoolMethod_kMax);
  op_conf.mutable_pooling_conf()->set_pad_h(1);
  op_conf.mutable_pooling_conf()->set_pad_w(1);
  op_conf.mutable_pooling_conf()->set_kernel_size_h(2);
  op_conf.mutable_pooling_conf()->set_kernel_size_w(2);
  op_conf.mutable_pooling_conf()->set_stride_h(2);
  op_conf.mutable_pooling_conf()->set_stride_w(2);
  return ConstructOp(op_conf);
}

template<typename T>
void TestPoolingOp(ParallelPolicy policy, bool has_data_id) {
  auto pooling_op = GetTestPoolingOp<T>();
  HashMap<std::string, BlobDesc*> bn2blob_desc_map{
      {"in", new BlobDesc(Shape({100, 64, 11, 11}), GetDataType<T>::val,
                          has_data_id)},
      {"out", new BlobDesc},
      {"idx", new BlobDesc}};
  auto Bn2BlobDescFunc = [&bn2blob_desc_map](const std::string& bn) {
    return bn2blob_desc_map.at(bn);
  };
  pooling_op->InferBlobDesc4FwBlobs(Bn2BlobDescFunc, policy, 0, 1);
  ASSERT_EQ(*Bn2BlobDescFunc("out"),
            BlobDesc(Shape({100, 64, 6, 6}), GetDataType<T>::val, has_data_id));
  ASSERT_EQ(*Bn2BlobDescFunc("idx"),
            BlobDesc(Shape({100, 64, 6, 6}), DataType::kUInt32, false));
}

TEST(PoolingOp, pooling) {
#define MAKE_ENTRY(data_type, policy, has_data_id) \
  TestPoolingOp<OF_PP_PAIR_FIRST(data_type)>(policy, has_data_id);
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ,
                                   PARALLEL_POLICY_SEQ, BOOL_SEQ)
}

}  // namespace oneflow
