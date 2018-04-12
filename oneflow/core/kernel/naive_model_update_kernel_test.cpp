#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
void NaiveModelUpdateTestCase(OpKernelTestCase* mdupdt_test_case,
                              const std::string& job_type,
                              const std::string& fw_or_bw) {
  mdupdt_test_case->set_is_train(job_type == "train");
  mdupdt_test_case->set_is_forward(fw_or_bw == "forward");
  NormalModelUpdateOpConf* normal_mdupdt_conf =
      mdupdt_test_case->mut_op_conf()->mutable_normal_mdupdt_conf();
  normal_mdupdt_conf->set_in_num(1);
  NormalModelUpdateOpUserConf* user_conf =
      normal_mdupdt_conf->mutable_user_conf();
  user_conf->set_learning_rate(1.0f);
  user_conf->mutable_naive_conf();

  mdupdt_test_case->InitJobConf([](JobConf* job_conf) {
    job_conf->set_single_piece_size(1);
    job_conf->set_data_part_num(1);
    job_conf->mutable_train_conf()->set_num_of_pieces_in_batch(1);
  });

  DataType data_type = GetDataType<T>::value;
  Shape shape({1, 3, 2});
  BlobDesc* blob_desc = new BlobDesc(shape, data_type, false, false, 1);
  mdupdt_test_case->template InitBlob<T>("in_0", blob_desc,
                                         std::vector<T>(shape.elem_cnt(), 1));
  mdupdt_test_case->template InitBlob<T>("packed_blob", blob_desc,
                                         std::vector<T>(shape.elem_cnt(), 2));
  mdupdt_test_case->template ForwardCheckBlob<T>(
      "model", blob_desc, std::vector<T>(shape.elem_cnt(), 1));

  auto* other = new std::tuple<int64_t, const Blob*>(
      -1, mdupdt_test_case->mut_bn_in_op2blob()->at("packed_blob"));
  mdupdt_test_case->mut_kernel_ctx()->other = other;
}

TEST_CPU_AND_GPU_OPKERNEL(NaiveModelUpdateTestCase, FLOATING_DATA_TYPE_SEQ,
                          (train), (forward));

DiffKernelImplTestCase* DiffNaiveModelUpdateKernelImpl(
    const std::string& job_type, const std::string& fw_or_bw,
    const std::string& cpp_type) {
  auto* mdupdt_test_case =
      new DiffKernelImplTestCase(job_type == "train", fw_or_bw == "forward",
                                 DataType4CppTypeString(cpp_type));
  mdupdt_test_case->InitJobConf([](JobConf* job_conf) {
    job_conf->set_single_piece_size(1);
    job_conf->set_data_part_num(1);
    job_conf->mutable_train_conf()->set_num_of_pieces_in_batch(1);
  });
  mdupdt_test_case->set_initiate_kernel_ctx(
      [mdupdt_test_case](
          const std::function<Blob*(const std::string&)>& Blob4BnInOp) {
        auto* other = new std::tuple<int64_t, const Blob*>(
            -1, Blob4BnInOp("packed_blob"));
        mdupdt_test_case->mut_kernel_ctx()->other = other;
      });
  NormalModelUpdateOpConf* normal_mdupdt_conf =
      mdupdt_test_case->mut_op_conf()->mutable_normal_mdupdt_conf();
  normal_mdupdt_conf->set_in_num(1);
  NormalModelUpdateOpUserConf* user_conf =
      normal_mdupdt_conf->mutable_user_conf();
  user_conf->set_learning_rate(1.0f);
  user_conf->mutable_naive_conf();

  mdupdt_test_case->SetBlobNames({"in_0", "packed_blob"}, {"model"}, {}, {});
  mdupdt_test_case->SetInputBlobDesc("in_0", Shape({1, 3, 2}),
                                     DataType4CppTypeString(cpp_type));
  mdupdt_test_case->SetInputBlobDesc("packed_blob", Shape({1, 3, 2}),
                                     DataType4CppTypeString(cpp_type));
  mdupdt_test_case->SetInputBlobDesc("model", Shape({1, 3, 2}),
                                     DataType4CppTypeString(cpp_type));
  return mdupdt_test_case;
}

TEST_DIFF_KERNEL_IMPL(DiffNaiveModelUpdateKernelImpl, (train), (forward),
                      OF_PP_SEQ_MAP(OF_PP_PAIR_FIRST, FLOATING_DATA_TYPE_SEQ));

}  // namespace test

}  // namespace oneflow