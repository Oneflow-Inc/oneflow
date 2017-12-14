#include "oneflow/core/job/mock_job_desc.h"
#include "oneflow/core/kernel/softmax_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

#include "oneflow/core/operator/softmax_op.h"
#include "oneflow/core/operator/op_test_util.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename T>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobMap() {
  using KTC = KTCommon<device_type, T>;

  DataType data_type = GetDataType<T>::val;
  auto bn2blob = new HashMap<std::string, Blob*>;
  (*bn2blob)["in"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({2, 4}), data_type, false), {1, 2, 3, 4, 0, 0, 0, 0});
  (*bn2blob)["out"] = KTC::CreateBlobWithRandomVal(
      new BlobDesc(Shape({2, 4}), data_type, false));
  (*bn2blob)["tmp"] =
      KTC::CreateBlobWithRandomVal(new BlobDesc(Shape({2}), data_type, false));
  (*bn2blob)["in_diff"] = KTC::CreateBlobWithRandomVal(
      new BlobDesc(Shape({2, 4}), data_type, false));
  (*bn2blob)["out_diff"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({2, 4}), data_type, false),
      {0.2f, 1, 2, 3, -4, 3, -2, 1});
  (*bn2blob)["expected_out"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({2, 4}), data_type, false),
      {0.0320586f, 0.0871443f, 0.2368828f, 0.6439143f, 0.25f, 0.25f, 0.25f,
       0.25f});
  (*bn2blob)["expected_in_diff"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({2, 4}), data_type, false),
      {-0.0737048f, -0.1306350f, -0.1182198f, 0.3225595f, -0.875f, 0.875f,
       -0.375f, 0.375f});
  return [bn2blob](const std::string& bn) { return bn2blob->at(bn); };
}

template<typename T, bool has_data_id>
std::function<BlobDesc*(const std::string)> BuildBn2BlobDescFunc(
    HashMap<std::string, BlobDesc*>& bn2blobdesc_map) {
  std::vector<std::vector<int64_t>> in_shapes = {{3, 5}};
  std::vector<std::string> ibns = {"in"};
  std::vector<std::string> obns = {"out"};
  std::vector<std::string> other_bns = {"tmp"};

  return ConstructBn2BlobDescFunc(bn2blobdesc_map, ibns, obns, other_bns,
                                  in_shapes, GetDataType<T>::val, has_data_id);
}

std::shared_ptr<Operator> BuildSoftmaxOp() {
  OperatorConf op_conf;
  op_conf.set_name("softmax_test");
  op_conf.mutable_softmax_conf()->set_in("softmax/in");
  op_conf.mutable_softmax_conf()->set_out("softmax/out");
  return ConstructOp(op_conf);
}

template<DeviceType device_type, typename T, bool has_data_id>
Kernel* BuildSoftmaxKernel(bool is_forward) {
  auto softmax_op = BuildSoftmaxOp();
  HashMap<std::string, BlobDesc*> bn2blobdesc_map;
  auto bn2blobdesc_func = BuildBn2BlobDescFunc<T, has_data_id>(bn2blobdesc_map);
  KernelConf kernel_conf;
  softmax_op->GenKernelConf(bn2blobdesc_func, is_forward, nullptr,
                            &kernel_conf);
  auto softmax_kernel = new SoftmaxKernel<device_type, T>();
  softmax_kernel->Init(nullptr, kernel_conf);
  return softmax_kernel;
}

template<DeviceType device_type, typename T, bool has_data_id>
void TestSoftmaxKernel() {
  auto softmax_kernel_forward =
      BuildSoftmaxKernel<device_type, T, has_data_id>(true);
  auto softmax_kernel_backward =
      BuildSoftmaxKernel<device_type, T, has_data_id>(false);

  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);
  auto bn2blob = BuildBnInOp2BlobMap<device_type, T>();
  softmax_kernel_forward->Launch(ctx, bn2blob);
  softmax_kernel_backward->Launch(ctx, bn2blob);
  SyncStream<device_type>(&ctx);
  using KTC = KTCommon<device_type, T>;
  KTC::CheckResult(bn2blob, "out", "expected_out");
  KTC::CheckResult(bn2blob, "in_diff", "expected_in_diff");
}

template<typename T, bool has_data_id>
void TestSoftmaxOp() {
  auto softmax_op = BuildSoftmaxOp();
  HashMap<std::string, BlobDesc*> bn2blobdesc_map;
  auto bn2blobdesc_func = BuildBn2BlobDescFunc<T, has_data_id>(bn2blobdesc_map);

  // infershape
  softmax_op->InferBlobDescs(bn2blobdesc_func, nullptr);

  // test
  BlobDesc* in_blobdesc = bn2blobdesc_func("in");
  BlobDesc* out_blobdesc = bn2blobdesc_func("out");
  BlobDesc* tmp_blobdesc = bn2blobdesc_func("tmp");

  ASSERT_TRUE(in_blobdesc->shape() == out_blobdesc->shape());
  ASSERT_TRUE(tmp_blobdesc->shape() == Shape({3}));
  ASSERT_TRUE(in_blobdesc->data_type() == out_blobdesc->data_type());
  ASSERT_TRUE(in_blobdesc->data_type() == tmp_blobdesc->data_type());
}

template<DeviceType device_type, typename T, bool has_data_id>
void TestSoftmaxOpKernel() {
  // mock JobDesc
  MockJobDesc mock_job_desc;
  InitJobDescSingleton(&mock_job_desc);
  EXPECT_CALL(mock_job_desc, DefaultDataType())
      .WillRepeatedly(testing::Return(GetDataType<T>::val));
  TestSoftmaxOp<T, has_data_id>();
  TestSoftmaxKernel<device_type, T, has_data_id>();
}

}  // namespace

}  // namespace test

TEST(SoftmaxKernel, softmax) {
#define MAKE_ENTRY(device_type, data_type_pair, has_data_id)               \
  test::TestSoftmaxOpKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair), \
                            has_data_id>();
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, DEVICE_TYPE_SEQ,
                                   FLOATING_DATA_TYPE_SEQ, BOOL_SEQ)
}

}  // namespace oneflow
