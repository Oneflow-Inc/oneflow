#include "oneflow/core/kernel/decode_ofrecord_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/record/ofrecord_decoder.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

namespace {

inline size_t ShapeCount(const ShapeProto& shape, size_t begin_axis = 0) {
  if (shape.dim_size() == 0) { return 0; }
  size_t ret = 1;
  FOR_RANGE(size_t, i, 0, shape.dim_size()) { ret *= shape.dim(i); }
  return ret;
}

void PreprocessBlob(const PreprocessConf& prep_conf, Blob* out_blob) {
#define MAKE_ENTRY(T, TVal)                                                         \
  {GetHashKey(TVal), [](const PreprocessConf& prep_conf, Blob* out_blob) {          \
     DoPreprocess<T>(prep_conf, out_blob->mut_dptr<T>(), out_blob->static_shape()); \
   }},
  static const HashMap<std::string, std::function<void(const PreprocessConf&, Blob*)>>
      preprocessers = {OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, ARITHMETIC_DATA_TYPE_SEQ)};
#undef MAKE_ENTRY
  preprocessers.at(GetHashKey(out_blob->data_type()))(prep_conf, out_blob);
}

}  // namespace

void DecodeOFRecordKernel::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  using namespace oneflow::data;
  // const DecodeOFRecordOpConf& decode_op_conf = op_conf().decode_ofrecord_conf();
  const DecodeOFRecordKernelConf& decode_kernel_conf = kernel_conf().decode_ofrecord_conf();
  random_seed_gen_.reset(new std::mt19937(decode_kernel_conf.random_seed()));
  distribution_.reset(new std::uniform_int_distribution<int32_t>(0, 1024 * 1024));
  parallel_num_ = parallel_ctx->parallel_num();
}

int32_t DecodeOFRecordKernel::NextRandomInt() const { return (*distribution_)(*random_seed_gen_); }

void DecodeOFRecordKernel::Forward(const KernelCtx& ctx,
                                   std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK(ctx.other);
  auto status = static_cast<DecodeStatus*>(ctx.other);
  Blob* in_blob = BnInOp2Blob("in");
  const DecodeOFRecordOpConf& decode_conf = op_conf().decode_ofrecord_conf();
  CHECK_EQ(op_attribute().output_bns_size(), decode_conf.blob_size());
  status->max_col_id_ = -1;

  if (decode_conf.transforms_size() > 0) {
    auto data_inst_vec = ParseRecordAndTransformData(ctx.device_ctx, in_blob);
    FOR_RANGE(int32_t, i, 0, op_attribute().output_bns_size()) {
      Blob* out_blob = BnInOp2Blob(op_attribute().output_bns(i));
      const BlobConf& blob_conf = decode_conf.blob(i);
      WriteDataToOutBlob(ctx.device_ctx, data_inst_vec, blob_conf, out_blob);
    }
  } else {
    FOR_RANGE(int32_t, i, 0, op_attribute().output_bns_size()) {
      Blob* out_blob = BnInOp2Blob(op_attribute().output_bns(i));
      const BlobConf& blob_conf = decode_conf.blob(i);
      OFRecordDecoderIf* decoder =
          GetOFRecordDecoder(blob_conf.encode_case().encode_case(), blob_conf.data_type());
      int32_t max_col_id =
          decoder->DecodeOneCol(ctx.device_ctx, in_blob, blob_conf, status->cur_col_id_, out_blob,
                                std::bind(&DecodeOFRecordKernel::NextRandomInt, this));

      if (status->max_col_id_ == -1) {
        status->max_col_id_ = max_col_id;
      } else {
        CHECK_EQ(status->max_col_id_, 0);
        CHECK_EQ(max_col_id, 0);
      }
      CHECK_LT(status->max_col_id_, out_blob->max_col_num());
    }
    CHECK_GE(status->max_col_id_, 0);
  }
}

std::vector<data::DataInstance> DecodeOFRecordKernel::ParseRecordAndTransformData(
    DeviceCtx* ctx, Blob* in_blob) const {
  using namespace oneflow::data;
  const DecodeOFRecordOpConf& decode_op_conf = op_conf().decode_ofrecord_conf();
  const DecodeOFRecordKernelConf& decode_kernel_conf = kernel_conf().decode_ofrecord_conf();
  const DataInstanceProto& data_inst_proto = decode_kernel_conf.data_instance_conf();
  RecordBlob<OFRecord> record_blob(in_blob);
  std::vector<DataInstance> data_inst_vec(record_blob.record_num());
  MultiThreadLoop(data_inst_vec.size(), [&](int64_t n) {
    const OFRecord& record = record_blob.GetRecord(n);
    DataInstance* data_inst = &data_inst_vec[n];
    data_inst->Init(decode_kernel_conf.data_instance_conf(), record);
    for (const DataTransformProto& transform_proto : decode_op_conf.transforms()) {
      data_inst->Transform(data_inst_proto, transform_proto);
    }
  });
  return data_inst_vec;
}

void DecodeOFRecordKernel::WriteDataToOutBlob(DeviceCtx* ctx,
                                              const std::vector<data::DataInstance>& data_inst_vec,
                                              const BlobConf& blob_conf, Blob* out_blob) const {
  const std::string& field_name = blob_conf.field();
  size_t elem_size = GetSizeOfDataType(blob_conf.data_type());
  size_t out_blob_inst_elem_cnt = ShapeCount(blob_conf.shape());
  size_t data_field_elem_cnt = data_inst_vec.at(0).GetField(field_name)->ElemCnt();
  bool dense = out_blob->has_instance_shape_field();
  const DataInstanceProto& data_inst_proto =
      kernel_conf().decode_ofrecord_conf().data_instance_conf();
  MultiThreadLoop(data_inst_vec.size(), [&](int64_t n) {
    const data::DataField* data_field = data_inst_vec.at(n).GetField(field_name);
    size_t one_inst_size = (dense ? data_field_elem_cnt : out_blob_inst_elem_cnt) * elem_size;
    char* dptr = static_cast<char*>(out_blob->mut_dptr()) + one_inst_size * n;
    size_t written_size = data_field->ToBuffer(dptr, data_inst_proto.fields().at(field_name));
    if (dense) {
      CHECK_EQ(one_inst_size, written_size);
    } else {
      CHECK_GE(one_inst_size, written_size);
      Memset<DeviceType::kCPU>(ctx, dptr + written_size, 0, one_inst_size - written_size);
      if (blob_conf.encode_case().has_raw()) {
        if (out_blob->has_dim1_valid_num_field()) {
          size_t one_col_size = ShapeCount(blob_conf.shape(), 1) * elem_size;
          out_blob->set_dim1_valid_num(n, written_size / one_col_size);
        }
      } else if (blob_conf.encode_case().has_bytes_list()) {
        using DataFieldT = typename data::DataFieldTrait<DataCodec::kBytesList, DataType::kChar,
                                                         DataCase::kSegmentation>::type;
        const auto* spec_field = dynamic_cast<const DataFieldT*>(data_field);
        CHECK_NOTNULL(spec_field);
        if (out_blob->has_dim1_valid_num_field()) {
          size_t one_col_size = spec_field->pb_max_size();
          out_blob->set_dim1_valid_num(n, written_size / one_col_size);
        }
        if (out_blob->has_dim2_valid_num_field()) {
          FOR_RANGE(size_t, i, 0, spec_field->pb_sizes().size()) {
            out_blob->set_dim2_valid_num(n, i, spec_field->pb_sizes().at(i));
          }
        }
      }
    }
  });
  if (dense) {
    size_t total_written_size = data_field_elem_cnt * data_inst_vec.size() * elem_size;
    Memset<DeviceType::kCPU>(ctx, static_cast<char*>(out_blob->mut_dptr()) + total_written_size, 0,
                             out_blob->ByteSizeOfDataContentField() - total_written_size);
  }
  FOR_RANGE(size_t, j, 0, blob_conf.preprocess_size()) {
    PreprocessBlob(blob_conf.preprocess(j), out_blob);
  }
}

REGISTER_KERNEL(OperatorConf::kDecodeOfrecordConf, DecodeOFRecordKernel);

}  // namespace oneflow
