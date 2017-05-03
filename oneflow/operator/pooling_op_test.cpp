#include "operator/pooling_op.h"
#include <vector>
#include "gtest/gtest.h"
#include "operator/operator_manager.h"
#include "operator/op_util.h"

namespace oneflow {

TEST(PoolingOp, pool_100x64x11x11) {
  // create pooling_op with input shape 100x64x11x11
  // PoolMethod = MAX
  // pad_h = pad_w = 1
  // kernel_h = hernel_w = 3
  // stride_h = stride_w = 2
  OperatorConf op_conf;
  op_conf.set_name("pooling_test");
  PoolingOpConf* pooling_conf = op_conf.mutable_pooling_conf();
  pooling_conf->set_in("pooling_in");
  pooling_conf->set_out("pooling_out");
  pooling_conf->set_pool(PoolingOpConf::MAX);
  pooling_conf->set_pad(1);
  pooling_conf->set_kernel_size(2);
  pooling_conf->set_stride(2);
  auto pooling_op = OpMgr::Singleton().ConstructOp(op_conf);
  std::vector<int64_t> input_shape_vec = {100, 64, 11, 11};
  HashMap<std::string, Shape*> bn2ShapePtr{
      {pooling_op->SoleIbn(), new Shape(input_shape_vec)},
      {pooling_op->SoleObn(), new Shape},
      {*(pooling_op->data_tmp_bns().begin()), new Shape}};
  auto fp = [&bn2ShapePtr](const std::string& bn) {
    return bn2ShapePtr.at(bn);
  };
  // do infer shape
  pooling_op->InferShape4FwBlobs(fp, kDataParallel, 0, 1);
  // test
  Shape* input_shape_ptr = bn2ShapePtr.at(pooling_op->SoleIbn());
  Shape* output_shape_ptr = bn2ShapePtr.at(pooling_op->SoleObn());
  Shape* data_tmp_shape_ptr = bn2ShapePtr.at(
      (*pooling_op->data_tmp_bns().begin()));
  // n * c * h_o * w_o
  // where h_o = (h_i + 2 * pad_h - kernel_h) / stride_h + 1 and w_o likewise.
  // n = 100
  // c = 64
  // h_o = (11 + 2 * 1 - 3) / 2 + 1 = 6
  // w_o = (11 + 2 * 1 - 3) / 2 + 1 = 6
  std::vector<int64_t> output_shape_vec = {100, 64, 6, 6};
  ASSERT_EQ(*output_shape_ptr, Shape(output_shape_vec));
  ASSERT_EQ(*data_tmp_shape_ptr, *output_shape_ptr);
  ASSERT_NE(data_tmp_shape_ptr, output_shape_ptr);
}

}  // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
