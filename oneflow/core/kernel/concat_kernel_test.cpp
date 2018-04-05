#include "oneflow/core/kernel/opkernel_test_case.h"
#include "oneflow/core/register/register_manager.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
void ConcatTestCase(OpKernelTestCase *concat_test_case,
                    const std::string &job_type,
                    const std::string &forward_or_backward) {
  concat_test_case->set_is_train(job_type == "train");
  concat_test_case->set_is_forward(forward_or_backward == "forward");
  auto concat_conf = concat_test_case->mut_op_conf()->mutable_concat_conf();
  concat_conf->set_axis(1);
  concat_conf->add_in("in_0");
  concat_conf->add_in("in_1");
  concat_conf->add_in("in_2");
  concat_conf->set_out("out");

  static JobConf job_conf;
  concat_test_case->InitJobConf(
      [](JobConf *job_conf_ptr) { job_conf_ptr = &job_conf; });

  static int64_t regst_desc_id = 0;
  RegstDescProto regst_desc_proto;
  regst_desc_proto.set_regst_desc_id(regst_desc_id++);
  regst_desc_proto.set_producer_task_id(0);
  regst_desc_proto.set_min_register_num(1);
  regst_desc_proto.set_max_register_num(1);
  regst_desc_proto.set_register_num(1);
  regst_desc_proto.mutable_mem_case()->mutable_host_pageable_mem();

  if (!Global<RegstMgr>::Get()) { Global<RegstMgr>::New(); }
  Regst *blob_regst = nullptr;
  Global<RegstMgr>::Get()->NewRegsts(
      regst_desc_proto, device_type, RecordTypeProto::kOFRecord,
      [&blob_regst](Regst *regst) { blob_regst = regst; });

  concat_test_case->EnrollBlobRegst("in_0", blob_regst);
  concat_test_case->EnrollBlobRegst("in_1", blob_regst);
  concat_test_case->EnrollBlobRegst("in_2", blob_regst);
  concat_test_case->EnrollBlobRegst("out", blob_regst);
  concat_test_case->EnrollBlobRegst(GenDiffBn("out"), blob_regst);
  concat_test_case->EnrollBlobRegst(GenDiffBn("in_0"), blob_regst);
  concat_test_case->EnrollBlobRegst(GenDiffBn("in_1"), blob_regst);
  concat_test_case->EnrollBlobRegst(GenDiffBn("in_2"), blob_regst);

  BlobDesc *blob_desc_212 =
      new BlobDesc(Shape({2, 1, 2}), GetDataType<T>::value, false, false, 1);
  BlobDesc *blob_desc_222 =
      new BlobDesc(Shape({2, 2, 2}), GetDataType<T>::value, false, false, 1);
  BlobDesc *blob_desc_242 =
      new BlobDesc(Shape({2, 4, 2}), GetDataType<T>::value, false, false, 1);

  concat_test_case->template InitBlob<T>("in_0", blob_desc_212, {1, 2, 3, 4});
  concat_test_case->template InitBlob<T>("in_1", blob_desc_222,
                                         {5, 6, 7, 8, 9, 10, 11, 12});
  concat_test_case->template InitBlob<T>("in_2", blob_desc_212,
                                         {13, 14, 15, 16});
  concat_test_case->template ForwardCheckBlob<T>(
      "out", blob_desc_242,
      {1, 2, 5, 6, 7, 8, 13, 14, 3, 4, 9, 10, 11, 12, 15, 16});

  concat_test_case->template InitBlob<T>(
      GenDiffBn("out"), blob_desc_242,
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  concat_test_case->template BackwardCheckBlob<T>(GenDiffBn("in_0"),
                                                  blob_desc_212, {1, 2, 9, 10});
  concat_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in_1"), blob_desc_222, {3, 4, 5, 6, 11, 12, 13, 14});
  concat_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in_2"), blob_desc_212, {7, 8, 15, 16});
}

TEST_CPU_AND_GPU_OPKERNEL(ConcatTestCase, ARITHMETIC_DATA_TYPE_SEQ,
                          (train)(predict), (forward)(backward));

}  // namespace test

}  // namespace oneflow
