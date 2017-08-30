#include "oneflow/core/operator/innerproduct_op.h"
#include <string>
#include <vector>
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

template<typename T>
void CheckBlobDesc(BlobDesc blob_desc, Shape shape, bool has_data_id) {
  CHECK_EQ(blob_desc.shape(), shape);
  CHECK_EQ(blob_desc.data_type(), GetDataType<T>::val);
  CHECK_EQ(blob_desc.has_data_id(), has_data_id);
}

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
  BlobDesc* blob_desc = new BlobDesc;
  blob_desc->mut_shape() = Shape(shape_vec);
  blob_desc->set_data_type(GetDataType<T>::val);
  blob_desc->set_has_data_id(has_data_id);
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
  CheckBlobDesc<T>(*bn2blobdesc_ptr.at(ip_op->SoleObn()),
                   Shape({1000, out_num}), has_data_id);

  // check weight blob
  CheckBlobDesc<T>(*bn2blobdesc_ptr.at(ip_op->model_bns().at(0)),
                   Shape({out_num, 3 * 256 * 256}), false);
  if (has_bias_term) {
    CheckBlobDesc<T>(*bn2blobdesc_ptr.at(ip_op->model_bns().at(1)),
                     Shape({1, out_num}), false);
    CheckBlobDesc<T>(*bn2blobdesc_ptr.at(ip_op->model_tmp_bns().at(0)),
                     Shape({1000, 1}), false);
  }
}

}  // namespace

TEST(InnerProductOp, modelparallel_innerproduct_op_test) {
  for (bool with_bias : {false, true}) {
    for (bool with_data_id : {false, true}) {
      TestInnerProductOp<float>(kModelParallel, with_bias, with_data_id);
      TestInnerProductOp<double>(kModelParallel, with_bias, with_data_id);
    }
  }
}

TEST(InnerProductOp, dataparallel_innerproduct_op_test) {
  for (bool with_bias : {false, true}) {
    for (bool with_data_id : {false, true}) {
      TestInnerProductOp<float>(kDataParallel, with_bias, with_data_id);
      TestInnerProductOp<double>(kDataParallel, with_bias, with_data_id);
    }
  }
}

}  // namespace oneflow
