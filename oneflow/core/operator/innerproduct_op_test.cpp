#include "oneflow/core/operator/innerproduct_op.h"
#include <string>
#include <vector>
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

TEST(InnerProductOp, modelparallel_innerproduct_with_bias) {
  OperatorConf op_conf;
  bool has_bias_term = true;
  op_conf.set_name("modelparallel_ip_test");
  op_conf.mutable_innerproduct_conf()->set_in("ip_in");
  op_conf.mutable_innerproduct_conf()->set_out("ip_out");
  op_conf.mutable_innerproduct_conf()->set_has_bias_term(has_bias_term);
  op_conf.mutable_innerproduct_conf()->set_out_num(40);
  auto ip_op = OpMgr::Singleton().ConstructOp(op_conf);
  std::vector<int64_t> shape_vec = {1000, 3, 256, 256};
  HashMap<std::string, Shape*> bn2shape_ptr = {
      {ip_op->SoleIbn(), new Shape(shape_vec)},
      {ip_op->SoleObn(), new Shape},
      {ip_op->model_bns().at(0), new Shape},
  };
  if (has_bias_term) {
    bn2shape_ptr[ip_op->model_bns().at(1)] = new Shape;
    bn2shape_ptr[ip_op->model_tmp_bns().at(0)] = new Shape;
  }
  auto fp = [&bn2shape_ptr](const std::string& bn) {
    return bn2shape_ptr.at(bn);
  };

  ip_op->InferShape4FwBlobs(fp, kModelParallel, 3, 10);

  BalancedSplitter splitter(40, 10);
  int out_num = splitter.At(3).size();

  Shape* out_shape_ptr = bn2shape_ptr.at(ip_op->SoleObn());
  CHECK_EQ(*out_shape_ptr, Shape({1000, out_num}));
  Shape* weight_shape_ptr = bn2shape_ptr.at(ip_op->model_bns().at(0));
  CHECK_EQ(*weight_shape_ptr, Shape({out_num, 3 * 256 * 256}));
  if (has_bias_term) {
    Shape* bias_shape_ptr = bn2shape_ptr.at(ip_op->model_bns().at(1));
    CHECK_EQ(*bias_shape_ptr, Shape({out_num}));
    Shape* bias_multiplier_shape_ptr =
        bn2shape_ptr.at(ip_op->model_tmp_bns().at(0));
    CHECK_EQ(*bias_multiplier_shape_ptr, Shape({1000, 1}));
  }
}

TEST(InnerProductOp, modelparallel_innerproduct_without_bias) {
  OperatorConf op_conf;
  bool has_bias_term = false;
  op_conf.set_name("modelparallel_ip_test");
  op_conf.mutable_innerproduct_conf()->set_in("ip_in");
  op_conf.mutable_innerproduct_conf()->set_out("ip_out");
  op_conf.mutable_innerproduct_conf()->set_has_bias_term(has_bias_term);
  op_conf.mutable_innerproduct_conf()->set_out_num(40);
  auto ip_op = OpMgr::Singleton().ConstructOp(op_conf);
  std::vector<int64_t> shape_vec = {1000, 3, 256, 256};
  HashMap<std::string, Shape*> bn2shape_ptr = {
      {ip_op->SoleIbn(), new Shape(shape_vec)},
      {ip_op->SoleObn(), new Shape},
      {ip_op->model_bns().at(0), new Shape},
  };
  if (has_bias_term) {
    bn2shape_ptr[ip_op->model_bns().at(1)] = new Shape;
    bn2shape_ptr[ip_op->model_tmp_bns().at(0)] = new Shape;
  }
  auto fp = [&bn2shape_ptr](const std::string& bn) {
    return bn2shape_ptr.at(bn);
  };

  ip_op->InferShape4FwBlobs(fp, kModelParallel, 3, 10);

  BalancedSplitter splitter(40, 10);
  int out_num = splitter.At(3).size();

  Shape* out_shape_ptr = bn2shape_ptr.at(ip_op->SoleObn());
  CHECK_EQ(*out_shape_ptr, Shape({1000, out_num}));
  Shape* weight_shape_ptr = bn2shape_ptr.at(ip_op->model_bns().at(0));
  CHECK_EQ(*weight_shape_ptr, Shape({out_num, 3 * 256 * 256}));
  if (has_bias_term) {
    Shape* bias_shape_ptr = bn2shape_ptr.at(ip_op->model_bns().at(1));
    CHECK_EQ(*bias_shape_ptr, Shape({out_num}));
    Shape* bias_multiplier_shape_ptr =
        bn2shape_ptr.at(ip_op->model_tmp_bns().at(0));
    CHECK_EQ(*bias_multiplier_shape_ptr, Shape({1000, 1}));
  }
}

TEST(InnerProductOp, dataparallel_innerproduct_with_bias) {
  OperatorConf op_conf;
  bool has_bias_term = true;
  op_conf.set_name("dataparallel_ip_test");
  op_conf.mutable_innerproduct_conf()->set_in("ip_in");
  op_conf.mutable_innerproduct_conf()->set_out("ip_out");
  op_conf.mutable_innerproduct_conf()->set_has_bias_term(has_bias_term);
  op_conf.mutable_innerproduct_conf()->set_out_num(40);
  auto ip_op = OpMgr::Singleton().ConstructOp(op_conf);

  std::vector<int64_t> shape_vec = {1000, 3, 256, 256};
  HashMap<std::string, Shape*> bn2shape_ptr = {
      {ip_op->SoleIbn(), new Shape(shape_vec)},
      {ip_op->SoleObn(), new Shape},
      {ip_op->model_bns().at(0), new Shape},
  };
  if (has_bias_term) {
    bn2shape_ptr[ip_op->model_bns().at(1)] = new Shape;
    bn2shape_ptr[ip_op->model_tmp_bns().at(0)] = new Shape;
  }
  auto fp = [&bn2shape_ptr](const std::string& bn) {
    return bn2shape_ptr.at(bn);
  };

  ip_op->InferShape4FwBlobs(fp, kDataParallel, 3, 10);

  Shape* out_shape_ptr = bn2shape_ptr.at(ip_op->SoleObn());
  CHECK_EQ(*out_shape_ptr, Shape({1000, 40}));
  Shape* weight_shape_ptr = bn2shape_ptr.at(ip_op->model_bns().at(0));
  CHECK_EQ(*weight_shape_ptr, Shape({40, 3 * 256 * 256}));
  if (has_bias_term) {
    Shape* bias_shape_ptr = bn2shape_ptr.at(ip_op->model_bns().at(1));
    CHECK_EQ(*bias_shape_ptr, Shape({40}));
    Shape* bias_multiplier_shape_ptr =
        bn2shape_ptr.at(ip_op->model_tmp_bns().at(0));
    CHECK_EQ(*bias_multiplier_shape_ptr, Shape({1000, 1}));
  }
}

TEST(InnerProductOp, dataparallel_innerproduct_without_bias) {
  OperatorConf op_conf;
  bool has_bias_term = false;
  op_conf.set_name("dataparallel_ip_test");
  op_conf.mutable_innerproduct_conf()->set_in("ip_in");
  op_conf.mutable_innerproduct_conf()->set_out("ip_out");
  op_conf.mutable_innerproduct_conf()->set_has_bias_term(has_bias_term);
  op_conf.mutable_innerproduct_conf()->set_out_num(40);
  auto ip_op = OpMgr::Singleton().ConstructOp(op_conf);

  std::vector<int64_t> shape_vec = {1000, 3, 256, 256};
  HashMap<std::string, Shape*> bn2shape_ptr = {
      {ip_op->SoleIbn(), new Shape(shape_vec)},
      {ip_op->SoleObn(), new Shape},
      {ip_op->model_bns().at(0), new Shape},
  };
  if (has_bias_term) {
    bn2shape_ptr[ip_op->model_bns().at(1)] = new Shape;
    bn2shape_ptr[ip_op->model_tmp_bns().at(0)] = new Shape;
  }
  auto fp = [&bn2shape_ptr](const std::string& bn) {
    return bn2shape_ptr.at(bn);
  };

  ip_op->InferShape4FwBlobs(fp, kDataParallel, 3, 10);

  Shape* out_shape_ptr = bn2shape_ptr.at(ip_op->SoleObn());
  CHECK_EQ(*out_shape_ptr, Shape({1000, 40}));
  Shape* weight_shape_ptr = bn2shape_ptr.at(ip_op->model_bns().at(0));
  CHECK_EQ(*weight_shape_ptr, Shape({40, 3 * 256 * 256}));
  if (has_bias_term) {
    Shape* bias_shape_ptr = bn2shape_ptr.at(ip_op->model_bns().at(1));
    CHECK_EQ(*bias_shape_ptr, Shape({40}));
    Shape* bias_multiplier_shape_ptr =
        bn2shape_ptr.at(ip_op->model_tmp_bns().at(0));
    CHECK_EQ(*bias_multiplier_shape_ptr, Shape({1000, 1}));
  }
}

}  // namespace oneflow
