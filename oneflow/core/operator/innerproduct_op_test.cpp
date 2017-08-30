#include "oneflow/core/operator/innerproduct_op.h"
#include <string>
#include <vector>
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

  std::vector<int64_t> shape_vec = {1000, 3, 256, 256};
  BlobDesc* blob_desc =
      new BlobDesc(Shape(shape_vec), GetDataType<T>::val, has_data_id);
  HashMap<std::string, BlobDesc*> bn2blobdesc_ptr = {
      {ip_op->SoleIbn(), blob_desc},
      {ip_op->SoleObn(), new BlobDesc},
      {ip_op->model_bns().at(0), new BlobDesc},
  };
  if (has_bias_term) {
    bn2blobdesc_ptr[ip_op->model_bns().at(1)] = new BlobDesc;
    bn2blobdesc_ptr[ip_op->model_tmp_bns().at(0)] = new BlobDesc;
  }
  auto fp = [&bn2blobdesc_ptr](const std::string& bn) {
    return bn2blobdesc_ptr.at(bn);
  };

  ip_op->InferBlobDesc4FwBlobs(fp, policy, 3, 10);

  if (policy == kModelParallel) {
    BalancedSplitter splitter(40, 10);
    out_num = splitter.At(3).size();
  }

  // check out blob
  BlobDesc out_blob_desc =
      BlobDesc(Shape({1000, out_num}), GetDataType<T>::val, has_data_id);
  CHECK(*bn2blobdesc_ptr.at(ip_op->SoleObn()) == out_blob_desc);

  // check weight blob
  BlobDesc weight_blob_desc =
      BlobDesc(Shape({out_num, 3 * 256 * 256}), GetDataType<T>::val, false);
  CHECK(*bn2blobdesc_ptr.at(ip_op->model_bns().at(0)) == weight_blob_desc);
  if (has_bias_term) {
    BlobDesc bias_blob_desc =
        BlobDesc(Shape({1, out_num}), GetDataType<T>::val, false);
    CHECK(*bn2blobdesc_ptr.at(ip_op->model_bns().at(1)) == bias_blob_desc);
    BlobDesc bias_tmp_blob_desc =
        BlobDesc(Shape({1000, 1}), GetDataType<T>::val, false);
    CHECK(*bn2blobdesc_ptr.at(ip_op->model_tmp_bns().at(0))
          == bias_tmp_blob_desc);
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
