#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
void TransposeTestCase_2d(OpKernelTestCase* transpose_test_case,
                          const std::string& job_type,
                          const std::string& forward_or_backward) {
  transpose_test_case->set_is_train(job_type == "train");
  transpose_test_case->set_is_forward(forward_or_backward == "forward");
  auto* conf = transpose_test_case->mut_op_conf()->mutable_transpose_conf();
  conf->add_perm(2);
  conf->add_perm(1);

  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 2, 3}), GetDataType<T>::value, false, false, 1);
  BlobDesc* trans_blob_desc =
      new BlobDesc(Shape({1, 3, 2}), GetDataType<T>::value, false, false, 1);
  transpose_test_case->template InitBlob<T>("in", blob_desc,
                                            {1, 2, 3, 4, 5, 6});
  transpose_test_case->template InitBlob<T>(GenDiffBn("out"), trans_blob_desc,
                                            {-8, 7, -6, 5, -4, 3});
  transpose_test_case->template ForwardCheckBlob<T>("out", trans_blob_desc,
                                                    {1, 4, 2, 5, 3, 6});
  transpose_test_case->template BackwardCheckBlob<T>(GenDiffBn("in"), blob_desc,
                                                     {-8, -6, -4, 7, 5, 3});
}
TEST_CPU_AND_GPU_OPKERNEL(TransposeTestCase_2d, FLOATING_DATA_TYPE_SEQ,
                          (train)(predict), (forward)(backward));

DiffKernelImplTestCase* DiffTransposeKernelImpl_cfirst_to_clast(
    const std::string& job_type, const std::string& fw_or_bw,
    const std::string& cpp_type) {
  auto* test_case =
      new DiffKernelImplTestCase(job_type == "train", fw_or_bw == "forward",
                                 DataType4CppTypeString(cpp_type));
  Shape shape({100, 3, 128, 128});
  Shape trans_shape({100, 128, 128, 3});
  auto* conf = test_case->mut_op_conf()->mutable_transpose_conf();
  conf->add_perm(2);
  conf->add_perm(3);
  conf->add_perm(1);

  test_case->SetBlobNames({"in"}, {"out"}, {GenDiffBn("out")},
                          {{GenDiffBn("in")}});
  test_case->SetInputBlobDesc("in", shape, DataType4CppTypeString(cpp_type));
  test_case->SetInputBlobDesc(GenDiffBn("out"), trans_shape,
                              DataType4CppTypeString(cpp_type));
  return test_case;
}
TEST_DIFF_KERNEL_IMPL(DiffTransposeKernelImpl_cfirst_to_clast, (train)(predict),
                      (forward)(backward),
                      OF_PP_SEQ_MAP(OF_PP_PAIR_FIRST, FLOATING_DATA_TYPE_SEQ));

DiffKernelImplTestCase* DiffTransposeKernelImpl_clast_to_cfirst(
    const std::string& job_type, const std::string& fw_or_bw,
    const std::string& cpp_type) {
  auto* test_case =
      new DiffKernelImplTestCase(job_type == "train", fw_or_bw == "forward",
                                 DataType4CppTypeString(cpp_type));
  Shape shape({100, 108, 128, 3});
  Shape trans_shape({100, 3, 108, 128});
  auto* conf = test_case->mut_op_conf()->mutable_transpose_conf();
  conf->add_perm(3);
  conf->add_perm(1);
  conf->add_perm(2);

  test_case->SetBlobNames({"in"}, {"out"}, {GenDiffBn("out")},
                          {{GenDiffBn("in")}});
  test_case->SetInputBlobDesc("in", shape, DataType4CppTypeString(cpp_type));
  test_case->SetInputBlobDesc(GenDiffBn("out"), trans_shape,
                              DataType4CppTypeString(cpp_type));
  return test_case;
}
TEST_DIFF_KERNEL_IMPL(DiffTransposeKernelImpl_clast_to_cfirst, (train)(predict),
                      (forward)(backward),
                      OF_PP_SEQ_MAP(OF_PP_PAIR_FIRST, FLOATING_DATA_TYPE_SEQ));

}  // namespace test

}  // namespace oneflow
