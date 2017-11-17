#include "oneflow/core/kernel/momentum_model_update_kernel.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename T>
Kernel* BuildMomentumMdUpdateKernel(float learning_rate, float beta) {
  OperatorConf op_conf;
  op_conf.set_name("momentum_model_update_test");
  MomentumModelUpdateOpConf* momentum_md_update_conf =
      op_conf.mutable_momentum_mdupdt_conf();
  momentum_md_update_conf->set_learning_rate(learning_rate);
  momentum_md_update_conf->set_beta(beta);
  auto momentum_md_update_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  momentum_md_update_op->ToProto(&op_proto);
  auto momentum_md_update_kernel = new MomentumMdUpdateKernel<device_type, T>();
  momentum_md_update_kernel->InitFromOpProto(op_proto);
  return momentum_md_update_kernel;
}

void InitJobDesc(int32_t piece_size, int32_t num_of_pieces_in_batch) {
  JobConf job_conf;
  job_conf.set_piece_size(piece_size);
  auto train_conf = job_conf.mutable_train_conf();
  train_conf->set_num_of_pieces_in_batch(num_of_pieces_in_batch);
  JobDesc::Singleton()->InitFromJobConf(job_conf);
}

template<DeviceType device_type, typename T>
std::function<Blob*(const std::string&)> BuildBnInOp2Blob() {
  using KTC = KTCommon<device_type, T>;

  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 3, 2}), GetDataType<T>::val, false);

  auto bn2blob = new HashMap<std::string, Blob*>;
  (*bn2blob)["model"] = KTC::CreateBlobWithSameVal(blob_desc, 2);
  (*bn2blob)["momentum"] = KTC::CreateBlobWithSameVal(blob_desc, 4);
  (*bn2blob)["model_diffs"] = KTC::CreateBlobWithSameVal(blob_desc, 4);
  (*bn2blob)["model_expected"] = KTC::CreateBlobWithSameVal(blob_desc, 3);
  (*bn2blob)["momentum_expected"] = KTC::CreateBlobWithSameVal(blob_desc, 1);
  return [bn2blob](const std::string& bn) { return bn2blob->at(bn); };
}

template<DeviceType device_type, typename T>
void TestMomentumMdUpdateKernel() {
  using KTC = KTCommon<device_type, T>;
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);

  const float learning_rate = {0.5f};
  const float beta = {0.5f};
  auto BnInOp2Blob = BuildBnInOp2Blob<device_type, T>();
  auto momentum_md_update_kernel =
      BuildMomentumMdUpdateKernel<device_type, T>(learning_rate, beta);
  int32_t piece_size = 1;
  int32_t num_of_pieces_in_batch = 2;
  InitJobDesc(piece_size, num_of_pieces_in_batch);

  momentum_md_update_kernel->Forward(ctx, BnInOp2Blob);
  SyncStream<device_type>(&ctx);

  KTC::CheckResult(BnInOp2Blob, "momentum", "momentum_expected");
  KTC::CheckResult(BnInOp2Blob, "model", "model_expected");
}

}  // namespace

}  // namespace test

TEST(MomentumMdUpdateKernel, model_update) {
#define MAKE_ENTRY(device_type, type_pair) \
  test::TestMomentumMdUpdateKernel<device_type, OF_PP_PAIR_FIRST(type_pair)>();
  OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, OF_PP_INTERNAL_SEQ_PRODUCT(
                                       DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ))
}

}  // namespace oneflow
