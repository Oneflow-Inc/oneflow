#include "oneflow/core/kernel/debug_kernel.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/record/ofrecord_raw_decoder.h"
#include "oneflow/core/record/ofrecord_raw_encoder.h"

namespace oneflow {

namespace {

size_t FeatureSize(const Feature& feature) {
  if (feature.has_double_list()) { return feature.double_list().value_size(); }
  if (feature.has_float_list()) { return feature.float_list().value_size(); }
  if (feature.has_int32_list()) { return feature.int32_list().value_size(); }
  UNIMPLEMENTED();
  return 0;
}

template<typename T>
void Decode(Blob* blob, const Feature& feature) {
  OFRecordDecoderImpl<EncodeCase::kRaw, T> decoder;
  CHECK_EQ(blob->shape().elem_cnt(), FeatureSize(feature));
  decoder.ReadOneCol(nullptr, feature, BlobConf(), 0, blob->mut_dptr<T>(), blob->shape().elem_cnt(),
                     []() { return 0; });
}

template<typename T>
void EncodeAndDump(const Blob* blob, PersistentOutStream* out_stream) {
  Feature feature;
  OFRecordEncoderImpl<EncodeCase::kRaw, T> encoder;
  encoder.EncodeBlob(nullptr, blob, &feature);
  *out_stream << feature;
  out_stream->Flush();
}

void InitConstFeature(Feature* feature, const std::string& filepath) {
  PersistentInStream in_stream(SnapshotFS(), filepath);
  int feature_size = -1;
  CHECK_EQ(in_stream.Read(reinterpret_cast<char*>(&feature_size), sizeof(int64_t)), 0);
  std::unique_ptr<char[]> buffer(new char[feature_size]);
  CHECK_EQ(in_stream.Read(buffer.get(), feature_size), 0);
  feature->ParseFromArray(buffer.get(), feature_size);
}

}  // namespace

template<typename T>
void DebugKernel<T>::InitOutStream(std::unique_ptr<PersistentOutStream>* out_stream,
                                   const ParallelContext* parallel_ctx, const std::string& dir) {
  const auto& conf = this->op_conf().debug_conf();
  OfCallOnce(dir, SnapshotFS(), &fs::FileSystem::RecursivelyCreateDir);
  int32_t part_name_suffix_length = conf.part_name_suffix_length();
  std::string num = std::to_string(parallel_ctx->parallel_id());
  int32_t zero_count = std::max(part_name_suffix_length - static_cast<int32_t>(num.length()), 0);
  std::string file_path =
      JoinPath(dir, conf.part_name_prefix() + std::string(zero_count, '0') + num);
  out_stream->reset(new PersistentOutStream(SnapshotFS(), file_path));
}

template<typename T>
void DebugKernel<T>::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  const auto& conf = this->op_conf().debug_conf();
  if (conf.has_in_blob_dump_dir()) {
    InitOutStream(&in_blob_out_stream_, parallel_ctx, conf.in_blob_dump_dir());
  }
  if (conf.has_out_diff_blob_dump_dir()) {
    InitOutStream(&out_diff_blob_out_stream_, parallel_ctx, conf.out_diff_blob_dump_dir());
  }
  if (conf.has_const_out_feature_load_filepath()) {
    InitConstFeature(&const_out_blob_feature_, conf.const_out_feature_load_filepath());
  }
  if (conf.has_const_in_diff_feature_load_filepath()) {
    InitConstFeature(&const_in_diff_blob_feature_, conf.const_in_diff_feature_load_filepath());
  }
}

template<typename T>
void DebugKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  const auto& conf = this->op_conf().debug_conf();
  if (conf.has_in_blob_dump_dir()) { EncodeAndDump<T>(in_blob, in_blob_out_stream_.get()); }
  if (conf.const_out_case() == DebugOpConf::ConstOutCase::CONST_OUT_NOT_SET) {
    out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
  } else if (conf.has_const_out_feature_load_filepath()) {
    Decode<T>(out_blob, const_out_blob_feature_);
  } else if (conf.has_const_out_feature()) {
    Decode<T>(out_blob, conf.const_out_feature());
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
void DebugKernel<T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const auto& conf = this->op_conf().debug_conf();
  if (conf.has_out_diff_blob_dump_dir()) {
    EncodeAndDump<T>(out_diff_blob, out_diff_blob_out_stream_.get());
  }
  if (conf.const_in_diff_case() == DebugOpConf::ConstInDiffCase::CONST_IN_DIFF_NOT_SET) {
    in_diff_blob->CopyDataContentFrom(ctx.device_ctx, out_diff_blob);
  } else if (conf.has_const_in_diff_feature_load_filepath()) {
    Decode<T>(in_diff_blob, const_in_diff_blob_feature_);
  } else if (conf.has_const_in_diff_feature()) {
    Decode<T>(in_diff_blob, conf.const_in_diff_feature());
  } else {
    UNIMPLEMENTED();
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kDebugConf, DebugKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
