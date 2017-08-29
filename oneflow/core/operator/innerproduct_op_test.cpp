#include "oneflow/core/operator/innerproduct_op.h"
#include <string>
#include <vector>
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

void InitOpConf(OperatorConf* op_conf, std::string name, bool has_bias_term) {
  op_conf->set_name(name);
  op_conf->mutable_innerproduct_conf()->mutable_in()->set_name("ip_in");
  op_conf->mutable_innerproduct_conf()->mutable_out()->set_name("ip_out");
  op_conf->mutable_innerproduct_conf()->set_has_bias_term(has_bias_term);
  op_conf->mutable_innerproduct_conf()->set_out_num(40);
}
void TestModelParallelInnerProductOp(bool has_bias_term) {
  OperatorConf op_conf;
  InitOpConf(&op_conf, "modelparallel_ip_test", has_bias_term);
  auto ip_op = ConstructOp(op_conf);
  std::vector<int64_t> shape_vec = {1000, 3, 256, 256};
  BlobDesc* blob_desc = new BlobDesc;
  blob_desc->mut_shape() = Shape(shape_vec);
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

  Shape* out_shape_ptr = &bn2blobdesc_ptr.at(ip_op->SoleObn())->mut_shape();
  CHECK_EQ(*out_shape_ptr, Shape({1000, out_num}));
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

void TestDataParallelInnerProductOp(bool has_bias_term) {
  OperatorConf op_conf;
  InitOpConf(&op_conf, "dataparallel_ip_test", has_bias_term);
  auto ip_op = ConstructOp(op_conf);

  std::vector<int64_t> shape_vec = {1000, 3, 256, 256};
  BlobDesc* blob_desc = new BlobDesc;
  blob_desc->mut_shape() = Shape(shape_vec);
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

TEST(InnerProductOp, modelparallel_innerproduct_with_bias) {
  TestModelParallelInnerProductOp(true);
}

TEST(InnerProductOp, modelparallel_innerproduct_without_bias) {
  TestModelParallelInnerProductOp(false);
}

TEST(InnerProductOp, dataparallel_innerproduct_with_bias) {
  TestDataParallelInnerProductOp(true);
}

TEST(InnerProductOp, dataparallel_innerproduct_without_bias) {
  TestDataParallelInnerProductOp(false);
}

}  // namespace oneflow
