#include "oneflow/core/kernel/rmsprop_model_update_kernel.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
void RmspropModelUpdateTestCase(OpKernelTestCase* mdupdt_test_case,
                              const std::string& job_type,
                              const std::string& fw_or_bw) {
  mdupdt_test_case->set_is_train(job_type == "train");
  mdupdt_test_case->set_is_forward(fw_or_bw == "forward");
  NormalModelUpdateOpConf* normal_mdupdt_conf =
      mdupdt_test_case->mut_op_conf()->mutable_normal_mdupdt_conf();
  normal_mdupdt_conf->set_in_num(1);
  NormalModelUpdateOpUserConf* user_conf =
      normal_mdupdt_conf->mutable_user_conf();
  user_conf->set_learning_rate(2.0f);
  RMSPropModelUpdateConf* rmsprop_conf = user_conf->mutable_rmsprop_conf();
  rmsprop_conf->set_decay_rate(1.0f / 2);
  rmsprop_conf->set_epsilon(3.0f);

  mdupdt_test_case->InitJobConf([](JobConf* job_conf) {
    job_conf->set_single_piece_size(1);
    job_conf->set_data_part_num(1);
    job_conf->mutable_train_conf()->set_num_of_pieces_in_batch(2);
  });

  DataType data_type = GetDataType<T>::value;
  Shape shape({1, 3, 2});
  BlobDesc* blob_desc = new BlobDesc(shape, data_type, false, false, 1);
  
  mdupdt_test_case->template InitBlob<T>("model", blob_desc, std::vector<T>(shape.elem_cnt(), 2));
  mdupdt_test_case->template InitBlob<T>("mean_square", blob_desc, std::vector<T>(shape.elem_cnt(), 0));
  mdupdt_test_case->template InitBlob<T>("packed_blob", blob_desc,
                                         std::vector<T>(shape.elem_cnt(), 2));
  mdupdt_test_case->template ForwardCheckBlob<T>(
      "model", blob_desc, std::vector<T>(shape.elem_cnt(), 1));

  auto* other = new std::tuple<int64_t, const Blob*>(
      -1, mdupdt_test_case->mut_bn_in_op2blob()->at("packed_blob"));
  mdupdt_test_case->mut_kernel_ctx()->other = other;
}

namespace {

template<DeviceType device_type, typename T>
Kernel* BuildRMSPropMdUpdateKernel(float learning_rate, float decay_rate,
                                   float epsilon) {
  OperatorConf op_conf;
  op_conf.set_name("rmsprop_model_update_test");
  RMSPropModelUpdateOpConf* rmsprop_md_update_conf =
      op_conf.mutable_rmsprop_mdupdt_conf();
  rmsprop_md_update_conf->set_learning_rate(learning_rate);
  rmsprop_md_update_conf->set_decay_rate(decay_rate);
  rmsprop_md_update_conf->set_epsilon(epsilon);
  auto rmsprop_md_update_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  rmsprop_md_update_op->ToProto(&op_proto);
  auto rmsprop_md_update_kernel = new RMSPropMdUpdateKernel<device_type, T>();
  rmsprop_md_update_kernel->InitFromOpProto(op_proto);
  return rmsprop_md_update_kernel;
}

void InitJobDesc(int32_t piece_size, int32_t num_of_pieces_in_batch) {
  JobConf job_conf;
  job_conf.set_piece_size(piece_size);
  auto train_conf = job_conf.mutable_train_conf();
  train_conf->set_num_of_pieces_in_batch(num_of_pieces_in_batch);
  Global<JobDesc>::Get()->InitFromJobConf(job_conf);
}

template<DeviceType device_type, typename T>
std::function<Blob*(const std::string&)> BuildBnInOp2Blob(
    std::vector<int64_t>& dim_vec) {
  using KTC = KTCommon<device_type, T>;

  BlobDesc* blob_desc =
      new BlobDesc(Shape(dim_vec), GetDataType<T>::value, false);

  auto bn2blob = new HashMap<std::string, Blob*>;
  (*bn2blob)["model"] = KTC::CreateBlobWithSameVal(blob_desc, 2);
  (*bn2blob)["mean_square"] = KTC::CreateBlobWithSameVal(blob_desc, 0);
  (*bn2blob)["model_diffs"] = KTC::CreateBlobWithSameVal(blob_desc, 2);
  (*bn2blob)["model_expected"] = KTC::CreateBlobWithSameVal(blob_desc, 0);
  (*bn2blob)["mean_square_expected"] = KTC::CreateBlobWithSameVal(blob_desc, 1);
  return [bn2blob](const std::string& bn) { return bn2blob->at(bn); };
}

template<DeviceType device_type, typename T>
void TestRMSPropMdUpdateKernel() {
  using KTC = KTCommon<device_type, T>;
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);
  ctx.other = new int64_t(1);

  std::vector<int64_t> dim_vec = {1, 3, 2};
  const float learning_rate = {2.0f};
  const float decay_rate = 1.0f / 2;
  const float epsilon = 3.0f;
  auto BnInOp2Blob = BuildBnInOp2Blob<device_type, T>(dim_vec);
  auto rmsprop_md_update_kernel = BuildRMSPropMdUpdateKernel<device_type, T>(
      learning_rate, decay_rate, epsilon);
  int32_t piece_size = 1;
  int32_t num_of_pieces_in_batch = 2;
  InitJobDesc(piece_size, num_of_pieces_in_batch);

  rmsprop_md_update_kernel->Forward(ctx, BnInOp2Blob);
  ctx.other = new int64_t(2);
  rmsprop_md_update_kernel->Forward(ctx, BnInOp2Blob);
  SyncStream<device_type>(&ctx);

  KTC::CheckResult(BnInOp2Blob, "mean_square", "mean_square_expected");
  KTC::CheckResult(BnInOp2Blob, "model", "model_expected");
}

}  // namespace

}  // namespace test

TEST(RMSPropMdUpdateKernel, model_update) {
#define MAKE_ENTRY(device_type, type_pair) \
  test::TestRMSPropMdUpdateKernel<device_type, OF_PP_PAIR_FIRST(type_pair)>();
  OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, OF_PP_INTERNAL_SEQ_PRODUCT(
                                       DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ))
}

}  // namespace oneflow
