#include "oneflow/core/kernel/opkernel_test_common.h"
#include "oneflow/core/kernel/momentum_model_update_kernel.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename T>
MdUpdateKernel<device_type>* BuildMomentumMdUpdateKernel(float learning_rate,
                                                         float beta) {
  OperatorConf op_conf;
  op_conf.set_name("momentum_model_update_test");
  MomentumModelUpdateOpConf* momentum_md_update_conf =
      op_conf.mutable_momentum_mdupdt_conf();
  momentum_md_update_conf->set_learning_rate(learning_rate);
  momentum_md_update_conf->set_beta(beta);
  auto momentum_md_update_op = ConstructOp(op_conf);
  auto bn2blob_desc_func = ConstructBn2BlobDescFunc(momentum_md_update_op);
  KernelConf kernel_conf;
  momentum_md_update_op->GenKernelConf(bn2blob_desc_func, false, nullptr,
                                       &kernel_conf);
  auto momentum_md_update_kernel = new MomentumMdUpdateKernel<device_type, T>();
  momentum_md_update_kernel->Init(nullptr, kernel_conf);
  return momentum_md_update_kernel;
}

template<typename T>
void InitJobDesc(int32_t piece_size, int32_t num_of_pieces_in_batch,
                 int32_t data_part_num) {
  JobConf job_conf;
  job_conf.set_default_data_type(GetDataType<T>::val);
  job_conf.set_single_piece_size(piece_size);
  job_conf.set_data_part_num(data_part_num);
  auto train_conf = job_conf.mutable_train_conf();
  train_conf->set_num_of_pieces_in_batch(num_of_pieces_in_batch);
  JobDesc::NewSingleton();
  JobDesc::Singleton()->job_conf_ = job_conf;
}

template<DeviceType device_type, typename T>
std::function<Blob*(const std::string&)> BuildBnInOp2Blob() {
  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 3, 2}), GetDataType<T>::val, false);

  RandomValConf random_val_conf = {{"model", blob_desc}};
  SameValConf same_val_conf = {
      {"pre_model", std::make_tuple(2, blob_desc)},
      {"momentum", std::make_tuple(4, blob_desc)},
      {"model_diff_acc", std::make_tuple(4, blob_desc)},
      {"model_expected", std::make_tuple(3, blob_desc)},
      {"momentum_expected", std::make_tuple(1, blob_desc)}};
  return KTCommon<device_type, T>::ConstructBnInOp2BlobFunc(
      random_val_conf, same_val_conf, SpecifiedValConf<T>());
}

template<DeviceType device_type, typename T>
void TestMomentumMdUpdateKernel() {
  using KTC = KTCommon<device_type, T>;
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);

  int32_t piece_size = 1;
  int32_t num_of_pieces_in_batch = 2;
  int32_t data_part_num = 1;
  InitJobDesc<T>(piece_size, num_of_pieces_in_batch, data_part_num);
  const float learning_rate = {0.5f};
  const float beta = {0.5f};
  auto BnInOp2Blob = BuildBnInOp2Blob<device_type, T>();
  auto momentum_mdupdt_kernel =
      BuildMomentumMdUpdateKernel<device_type, T>(learning_rate, beta);

  momentum_mdupdt_kernel->UpdateModel(ctx.device_ctx, BnInOp2Blob("pre_model"),
                                      0, BnInOp2Blob);
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
