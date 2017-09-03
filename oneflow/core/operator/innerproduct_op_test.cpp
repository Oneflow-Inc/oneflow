#include "oneflow/core/operator/innerproduct_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

template<typename T>
void TestInnerProductOp(ParallelPolicy policy, bool has_bias_term,
                        bool has_data_id) {
  int out_num = 40;
  JobConf job_conf;
  job_conf.set_default_data_type(GetDataType<T>::val);
  JobDesc::Singleton()->InitFromJobConf(job_conf);

  OperatorConf op_conf;
  op_conf.set_name("innerproduct_test");
  op_conf.mutable_innerproduct_conf()->mutable_in()->set_name("ip_in");
  op_conf.mutable_innerproduct_conf()->mutable_in()->set_data_type(
      GetDataType<T>::val);
  op_conf.mutable_innerproduct_conf()->mutable_out()->set_name("ip_out");
  op_conf.mutable_innerproduct_conf()->mutable_out()->set_data_type(
      GetDataType<T>::val);
  op_conf.mutable_innerproduct_conf()->set_has_bias_term(has_bias_term);
  op_conf.mutable_innerproduct_conf()->set_out_num(out_num);
  auto ip_op = ConstructOp(op_conf);

  HashMap<std::string, BlobDesc*> bn2blob_desc = {
      {"in", new BlobDesc(Shape({1000, 3, 256, 256}), GetDataType<T>::val,
                          has_data_id)},
      {"out", new BlobDesc},
      {"weight", new BlobDesc},
  };
  if (has_bias_term) {
    bn2blob_desc["bias"] = new BlobDesc;
    bn2blob_desc["bias_multiplier"] = new BlobDesc;
  }
  auto bn2blob_desc_func = [&](const std::string& bn) {
    return bn2blob_desc.at(bn);
  };

  ip_op->InferBlobDesc4FwBlobs(bn2blob_desc_func, policy, 3, 10);

  if (policy == kModelParallel) {
    BalancedSplitter splitter(out_num, 10);
    out_num = splitter.At(3).size();
  }

  ASSERT_TRUE(
      *bn2blob_desc.at("out")
      == BlobDesc(Shape({1000, out_num}), GetDataType<T>::val, has_data_id));

  ASSERT_TRUE(
      *bn2blob_desc.at("weight")
      == BlobDesc(Shape({out_num, 3 * 256 * 256}), GetDataType<T>::val, false));
  if (has_bias_term) {
    ASSERT_TRUE(*bn2blob_desc.at("bias")
                == BlobDesc(Shape({1, out_num}), GetDataType<T>::val, false));
    ASSERT_TRUE(*bn2blob_desc.at("bias_multiplier")
                == BlobDesc(Shape({1000, 1}), GetDataType<T>::val, false));
  }
}

}  // namespace

TEST(InnerProductOp, innerproduct) {
#define MAKE_ENTRY(data_type, policy, has_bias, has_data_id)        \
  TestInnerProductOp<OF_PP_PAIR_FIRST(data_type)>(policy, has_bias, \
                                                  has_data_id);
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ,
                                   PARALLEL_POLICY_SEQ, BOOL_SEQ, BOOL_SEQ)
}

}  // namespace oneflow
