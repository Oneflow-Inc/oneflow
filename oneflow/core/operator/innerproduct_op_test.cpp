#include "oneflow/core/operator/innerproduct_op.h"
#include <string>
#include <vector>
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

template<typename T>
void InitOpConf(OperatorConf* op_conf, std::string name, bool has_bias_term) {
  op_conf->set_name(name);
  op_conf->mutable_innerproduct_conf()->mutable_in()->set_name("ip_in");
  op_conf->mutable_innerproduct_conf()->mutable_in()->set_data_type(
      GetDataType<T>::val);
  op_conf->mutable_innerproduct_conf()->mutable_out()->set_name("ip_out");
  op_conf->mutable_innerproduct_conf()->mutable_out()->set_data_type(
      GetDataType<T>::val);
  op_conf->mutable_innerproduct_conf()->set_has_bias_term(has_bias_term);
  op_conf->mutable_innerproduct_conf()->set_out_num(40);
}

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
  // op_conf.mutable_innerproduct_conf()->mutable_out()->set_data_type(kChar);
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

  Shape* out_shape_ptr = &bn2blobdesc_ptr.at(ip_op->SoleObn())->mut_shape();
  CHECK_EQ(*out_shape_ptr, Shape({1000, out_num}));
  CheckBlobDesc<T>(*bn2blobdesc_ptr.at(ip_op->SoleObn()),
                   Shape({1000, out_num}), has_data_id);
  CheckBlobDesc<T>(*bn2blobdesc_ptr.at(ip_op->model_bns().at(0)),
                   Shape({out_num, 3 * 256 * 256}), false);
  Shape* weight_shape_ptr =
      &bn2blobdesc_ptr.at(ip_op->model_bns().at(0))->mut_shape();
  CHECK_EQ(*weight_shape_ptr, Shape({out_num, 3 * 256 * 256}));
  if (has_bias_term) {
    CheckBlobDesc<T>(*bn2blobdesc_ptr.at(ip_op->model_bns().at(1)),
                     Shape({1, out_num}), false);
    CheckBlobDesc<T>(*bn2blobdesc_ptr.at(ip_op->model_tmp_bns().at(0)),
                     Shape({1000, 1}), false);
    Shape* bias_shape_ptr =
        &bn2blobdesc_ptr.at(ip_op->model_bns().at(1))->mut_shape();
    CHECK_EQ(*bias_shape_ptr, Shape({1, out_num}));
    Shape* bias_multiplier_shape_ptr =
        &bn2blobdesc_ptr.at(ip_op->model_tmp_bns().at(0))->mut_shape();
    CHECK_EQ(*bias_multiplier_shape_ptr, Shape({1000, 1}));
  }
}

template<typename T>
void TestModelParallelInnerProductOp(bool has_bias_term) {
  JobConf job_conf;
  job_conf.set_default_data_type(GetDataType<T>::val);
  JobDesc::Singleton()->InitFromJobConf(job_conf);
  OperatorConf op_conf;
  InitOpConf<T>(&op_conf, "modelparallel_ip_test", has_bias_term);
  auto ip_op = ConstructOp(op_conf);
  std::vector<int64_t> shape_vec = {1000, 3, 256, 256};
  BlobDesc* blob_desc = new BlobDesc;
  blob_desc->mut_shape() = Shape(shape_vec);
  blob_desc->set_data_type(GetDataType<T>::val);
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

  ip_op->InferBlobDesc4FwBlobs(fp, kModelParallel, 3, 10);

  BalancedSplitter splitter(40, 10);
  int out_num = splitter.At(3).size();

  Shape out_shape_ptr = bn2blobdesc_ptr.at(ip_op->SoleObn())->mut_shape();
  CHECK_EQ(out_shape_ptr, Shape({1000, out_num}));
  Shape* weight_shape_ptr =
      &bn2blobdesc_ptr.at(ip_op->model_bns().at(0))->mut_shape();
  CHECK_EQ(*weight_shape_ptr, Shape({out_num, 3 * 256 * 256}));
  if (has_bias_term) {
    Shape* bias_shape_ptr =
        &bn2blobdesc_ptr.at(ip_op->model_bns().at(1))->mut_shape();
    CHECK_EQ(*bias_shape_ptr, Shape({1, out_num}));
    Shape* bias_multiplier_shape_ptr =
        &bn2blobdesc_ptr.at(ip_op->model_tmp_bns().at(0))->mut_shape();
    CHECK_EQ(*bias_multiplier_shape_ptr, Shape({1000, 1}));
  }
}

template<typename T>
void TestDataParallelInnerProductOp(bool has_bias_term) {
  JobConf job_conf;
  job_conf.set_default_data_type(GetDataType<T>::val);
  JobDesc::Singleton()->InitFromJobConf(job_conf);
  OperatorConf op_conf;
  InitOpConf<T>(&op_conf, "dataparallel_ip_test", has_bias_term);
  auto ip_op = ConstructOp(op_conf);

  std::vector<int64_t> shape_vec = {1000, 3, 256, 256};
  BlobDesc* blob_desc = new BlobDesc;
  blob_desc->mut_shape() = Shape(shape_vec);
  blob_desc->set_data_type(GetDataType<T>::val);
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

  ip_op->InferBlobDesc4FwBlobs(fp, kDataParallel, 3, 10);

  Shape* out_shape_ptr = &bn2blobdesc_ptr.at(ip_op->SoleObn())->mut_shape();
  CHECK_EQ(*out_shape_ptr, Shape({1000, 40}));
  Shape* weight_shape_ptr =
      &bn2blobdesc_ptr.at(ip_op->model_bns().at(0))->mut_shape();
  CHECK_EQ(*weight_shape_ptr, Shape({40, 3 * 256 * 256}));
  if (has_bias_term) {
    Shape* bias_shape_ptr =
        &bn2blobdesc_ptr.at(ip_op->model_bns().at(1))->mut_shape();
    CHECK_EQ(*bias_shape_ptr, Shape({1, 40}));
    Shape* bias_multiplier_shape_ptr =
        &bn2blobdesc_ptr.at(ip_op->model_tmp_bns().at(0))->mut_shape();
    CHECK_EQ(*bias_multiplier_shape_ptr, Shape({1000, 1}));
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

TEST(InnerProductOp, modelparallel_innerproduct_with_bias) {
  TestModelParallelInnerProductOp<float>(true);
  TestModelParallelInnerProductOp<double>(true);
}

TEST(InnerProductOp, modelparallel_innerproduct_without_bias) {
  TestModelParallelInnerProductOp<float>(false);
  TestModelParallelInnerProductOp<double>(false);
}

TEST(InnerProductOp, dataparallel_innerproduct_with_bias) {
  TestDataParallelInnerProductOp<float>(true);
  TestDataParallelInnerProductOp<double>(true);
}

TEST(InnerProductOp, dataparallel_innerproduct_without_bias) {
  TestDataParallelInnerProductOp<float>(false);
  TestDataParallelInnerProductOp<double>(false);
}

}  // namespace oneflow
