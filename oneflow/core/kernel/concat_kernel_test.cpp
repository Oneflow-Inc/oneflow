#include "oneflow/core/kernel/concat_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/register/blob_desc.pb.h"
#include "oneflow/core/register/register_desc.pb.h"
#include "oneflow/core/kernel/opkernel_test_case.h"
#include "oneflow/core/register/register_manager.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
OpKernelTestCase* ConcatTestCase(const std::string& job_type, const std::string& forward_or_backward) {
  static int64_t regst_desc_id = 0;
  OpKernelTestCase* concat_test_case = new OpKernelTestCase();
  concat_test_case->set_is_train(job_type == "train");
  concat_test_case->set_is_forward(forward_or_backward == "forward");
  concat_test_case->set_device_type(device_type);

  JobDescProto job_desc_proto;
  JobConf job_conf;
  *job_desc_proto.mutable_job_conf() = job_conf;
  if (!Global<JobDesc>::Get()) {
    Global<JobDesc>::New(job_desc_proto);
  }

  if (!Global<RegstMgr>::Get()) {
    Global<RegstMgr>::New();
  }

  auto concat_conf = concat_test_case->mut_op_conf()->mutable_concat_conf();
  concat_conf->set_axis(1);
  concat_conf->add_in("in_0");
  concat_conf->add_in("in_1");
  concat_conf->add_in("in_2");
  concat_conf->set_out("out");

  RegstDescProto regst_desc_proto;
  regst_desc_proto.set_regst_desc_id(regst_desc_id++);
  regst_desc_proto.set_producer_task_id(0);
  regst_desc_proto.set_min_register_num(1);
  regst_desc_proto.set_max_register_num(1);
  regst_desc_proto.set_register_num(1);
  regst_desc_proto.mutable_mem_case()->mutable_host_pageable_mem();

  Regst* blob_regst = nullptr;
  Global<RegstMgr>::Get()->NewRegsts(regst_desc_proto, device_type, RecordTypeProto::kOFRecord, [&blob_regst](Regst* regst) {
    blob_regst = regst;
  });

  concat_test_case->EnrollBlobRegst("in_0", blob_regst);
  concat_test_case->EnrollBlobRegst("in_1", blob_regst);
  concat_test_case->EnrollBlobRegst("in_2", blob_regst);

  BlobDesc* blob_desc_212 = new BlobDesc(Shape({2, 1, 2}), GetDataType<T>::value, false, false, 1);
  BlobDesc* blob_desc_222 = new BlobDesc(Shape({2, 2, 2}), GetDataType<T>::value, false, false, 1);
  BlobDesc* blob_desc_242 = new BlobDesc(Shape({2, 4, 2}), GetDataType<T>::value, false, false, 1);

  concat_test_case->template InitBlob<T>("in_0", blob_desc_212, {1, 2, 3, 4});
  concat_test_case->template InitBlob<T>("in_1", blob_desc_222, {5, 6, 7, 8, 9, 10, 11, 12});
  concat_test_case->template InitBlob<T>("in_2", blob_desc_212, {13, 14, 15, 16});
  concat_test_case->template ForwardCheckBlob<T>("out", blob_desc_242, {1, 2, 5, 6, 7, 8, 13, 14, 3, 4, 9, 10, 11, 12, 15, 16});

  concat_test_case->template InitBlob<T>(GenDiffBn("out"), blob_desc_242, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  concat_test_case->template BackwardCheckBlob<T>(GenDiffBn("in_0"), blob_desc_212, {1, 2, 9, 10});
  concat_test_case->template BackwardCheckBlob<T>(GenDiffBn("in_1"), blob_desc_222, {3, 4, 5, 6, 11, 12, 13, 14});
  concat_test_case->template BackwardCheckBlob<T>(GenDiffBn("in_2"), blob_desc_212, {7, 8, 15, 16});

  /*
  BlobDescProto blob_desc_pb212;
  Shape shape212({2, 1, 2});
  shape212.ToProto(blob_desc_pb212.mutable_shape());
  blob_desc_pb212.set_data_type(GetDataType<T>::val);
  blob_desc_pb212.set_has_data_id_field(false);
  blob_desc_pb212.set_has_col_num_field(false);
  blob_desc_pb212.set_max_col_num(1);

  BlobDescProto blob_desc_pb222;
  Shape shape222({2, 2, 2});
  shape222.ToProto(blob_desc_pb222.mutable_shape());
  blob_desc_pb222.set_data_type(GetDataType<T>::val);
  blob_desc_pb222.set_has_data_id_field(false);
  blob_desc_pb222.set_has_col_num_field(false);
  blob_desc_pb222.set_max_col_num(1);

  BlobDescProto blob_desc_pb242;
  Shape shape242({2, 4, 2});
  shape242.ToProto(blob_desc_pb242.mutable_shape());
  blob_desc_pb242.set_data_type(GetDataType<T>::val);
  blob_desc_pb242.set_has_data_id_field(false);
  blob_desc_pb242.set_has_col_num_field(false);
  blob_desc_pb242.set_max_col_num(1);

  RegstDescProto regst_desc_proto212;
  regst_desc_proto212.set_regst_desc_id(regst_desc_id++);
  regst_desc_proto212.set_producer_task_id(0);
  (*regst_desc_proto212.mutable_lbn2blob_desc())["212"] = blob_desc_pb212;
  (*regst_desc_proto212.mutable_packed_blob_desc()) = blob_desc_pb212;
  regst_desc_proto212.set_min_register_num(1);
  regst_desc_proto212.set_max_register_num(1);
  regst_desc_proto212.set_register_num(1);
  regst_desc_proto212.mutable_mem_case()->mutable_host_pageable_mem();

  RegstDescProto regst_desc_proto222;
  regst_desc_proto222.set_regst_desc_id(regst_desc_id);
  regst_desc_proto222.set_producer_task_id(0);
  (*regst_desc_proto222.mutable_lbn2blob_desc())["222"] = blob_desc_pb222;
  (*regst_desc_proto222.mutable_packed_blob_desc()) = blob_desc_pb222;
  regst_desc_proto222.set_min_register_num(1);
  regst_desc_proto222.set_max_register_num(1);
  regst_desc_proto222.set_register_num(1);
  regst_desc_proto222.mutable_mem_case()->mutable_host_pageable_mem();

  RegstDescProto regst_desc_proto242;
  regst_desc_proto242.set_regst_desc_id(regst_desc_id);
  regst_desc_proto242.set_producer_task_id(0);
  (*regst_desc_proto242.mutable_lbn2blob_desc())["242"] = blob_desc_pb242;
  (*regst_desc_proto242.mutable_packed_blob_desc()) = blob_desc_pb242;
  regst_desc_proto242.set_min_register_num(1);
  regst_desc_proto242.set_max_register_num(1);
  regst_desc_proto242.set_register_num(1);
  regst_desc_proto242.mutable_mem_case()->mutable_host_pageable_mem();

  Regst* regst_in_0 = nullptr;
  Global<RegstMgr>::Get()->NewRegsts(regst_desc_proto212, device_type, RecordTypeProto::kOFRecord, [&regst_in_0](Regst* regst) {
    regst_in_0 = regst;
  });
  Blob* blob_in_0 = regst_in_0->GetBlobByLbn("212");
  std::vector<T> val_in_0 = {1, 2, 3, 4};
  CudaCheck(cudaMemcpy(blob_in_0->mut_dptr(), &(val_in_0[0]), blob_in_0->ByteSizeOfDataContentField(), cudaMemcpyHostToHost));

  Regst* regst_in_1 = nullptr;
  Global<RegstMgr>::Get()->NewRegsts(regst_desc_proto222, device_type, RecordTypeProto::kOFRecord, [&regst_in_1](Regst* regst) {
    regst_in_1 = regst;
  });
  Blob* blob_in_1 = regst_in_1->GetBlobByLbn("222");
  std::vector<T> val_in_1 = {5, 6, 7, 8, 9, 10, 11, 12};
  CudaCheck(cudaMemcpy(blob_in_1->mut_dptr(), &(val_in_1[0]), blob_in_1->ByteSizeOfDataContentField(), cudaMemcpyHostToHost));

  Regst* regst_in_2 = nullptr;
  Global<RegstMgr>::Get()->NewRegsts(regst_desc_proto212, device_type, RecordTypeProto::kOFRecord, [&regst_in_2](Regst* regst) {
    regst_in_2 = regst;
  });
  Blob* blob_in_2 = regst_in_2->GetBlobByLbn("212");
  std::vector<T> val_in_2 = {13, 14, 15, 16};
  CudaCheck(cudaMemcpy(blob_in_2->mut_dptr(), &(val_in_2[0]), blob_in_2->ByteSizeOfDataContentField(), cudaMemcpyHostToHost));

  Regst* regst_out = nullptr;
  Global<RegstMgr>::Get()->NewRegsts(regst_desc_proto242, device_type, RecordTypeProto::kOFRecord, [&regst_out](Regst* regst) {
    regst_out = regst;
  });
  Blob* blob_out = regst_out->GetBlobByLbn("242");
  std::vector<T> val_out = {1, 2, 5, 6, 7, 8, 13, 14, 3, 4, 9, 10, 11, 12, 15, 16};
  CudaCheck(cudaMemcpy(blob_out->mut_dptr(), &(val_out[0]), blob_out->ByteSizeOfDataContentField(), cudaMemcpyHostToHost));

  concat_test_case->InitBlob("in_0", blob_in_0);
  concat_test_case->InitBlob("in_1", blob_in_1);
  concat_test_case->InitBlob("in_2", blob_in_2);
  concat_test_case->ForwardCheckBlob("out", device_type, blob_out);

  Regst* regst_out_diff = nullptr;
  Global<RegstMgr>::Get()->NewRegsts(regst_desc_proto242, device_type, RecordTypeProto::kOFRecord, [&regst_out_diff](Regst* regst) {
    regst_out_diff = regst;
  });
  Blob* blob_out_diff = regst_out_diff->GetBlobByLbn("242");
  std::vector<T> val_out_diff = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  CudaCheck(cudaMemcpy(blob_out_diff->mut_dptr(), &(val_out_diff[0]), blob_out_diff->ByteSizeOfDataContentField(), cudaMemcpyHostToHost));

  Regst* regst_in_diff_0 = nullptr;
  Global<RegstMgr>::Get()->NewRegsts(regst_desc_proto212, device_type, RecordTypeProto::kOFRecord, [&regst_in_diff_0](Regst* regst) {
    regst_in_diff_0 = regst;
  });
  Blob* blob_in_diff_0 = regst_in_diff_0->GetBlobByLbn("212");
  std::vector<T> val_in_diff_0 = {1, 2, 9, 10};
  CudaCheck(cudaMemcpy(blob_in_diff_0->mut_dptr(), &(val_in_diff_0[0]), blob_in_diff_0->ByteSizeOfDataContentField(), cudaMemcpyHostToHost));

  Regst* regst_in_diff_1 = nullptr;
  Global<RegstMgr>::Get()->NewRegsts(regst_desc_proto222, device_type, RecordTypeProto::kOFRecord, [&regst_in_diff_1](Regst* regst) {
    regst_in_diff_1 = regst;
  });
  Blob* blob_in_diff_1 = regst_in_diff_1->GetBlobByLbn("222");
  std::vector<T> val_in_diff_1 = {3, 4, 5, 6, 11, 12, 13, 14};
  CudaCheck(cudaMemcpy(blob_in_diff_1->mut_dptr(), &(val_in_diff_1[0]), blob_in_diff_1->ByteSizeOfDataContentField(), cudaMemcpyHostToHost));

  Regst* regst_in_diff_2 = nullptr;
  Global<RegstMgr>::Get()->NewRegsts(regst_desc_proto212, device_type, RecordTypeProto::kOFRecord, [&regst_in_diff_2](Regst* regst) {
    regst_in_diff_2 = regst;
  });
  Blob* blob_in_diff_2 = regst_in_diff_2->GetBlobByLbn("212");
  std::vector<T> val_in_diff_2 = {7, 8, 15, 16};
  CudaCheck(cudaMemcpy(blob_in_diff_2->mut_dptr(), &(val_in_diff_2[0]), blob_in_diff_2->ByteSizeOfDataContentField(), cudaMemcpyHostToHost));

  concat_test_case->InitBlob(GenDiffBn("out"), blob_out_diff);
  concat_test_case->BackwardCheckBlob(GenDiffBn("in_0"), device_type, blob_in_diff_0);
  concat_test_case->BackwardCheckBlob(GenDiffBn("in_1"), device_type, blob_in_diff_1);
  concat_test_case->BackwardCheckBlob(GenDiffBn("in_2"), device_type, blob_in_diff_2);
  */

  return concat_test_case;
}

TEST_CPU_ONLY_OPKERNEL(ConcatTestCase, FLOATING_DATA_TYPE_SEQ SIGNED_INT_DATA_TYPE_SEQ, (train)(predict), (forward)(backward));

}  // namespace test

}  // namespace oneflow

