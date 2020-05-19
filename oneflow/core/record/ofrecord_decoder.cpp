#include "oneflow/core/record/ofrecord_decoder.h"
#include "oneflow/core/record/ofrecord_raw_decoder.h"
#include "oneflow/core/record/ofrecord_jpeg_decoder.h"
#include "oneflow/core/record/ofrecord_bytes_list_decoder.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/record/encode_case_util.h"

namespace oneflow {

namespace {

template<typename T>
void DoSubtractPreprocess(const SubtractPreprocessConf& conf, T* dptr, int64_t n) {
  FOR_RANGE(size_t, i, 0, n) { (*(dptr++)) -= conf.value(); }
}

template<typename T>
void DoNormByChannelPreprocess(const NormByChannelPreprocessConf& conf, T* dptr,
                               const ShapeView& shape) {
  std::vector<float> std_value(conf.mean_value_size(), 1.0);
  if (conf.std_value_size() > 0) {
    CHECK_EQ(conf.mean_value_size(), conf.std_value_size());
    FOR_RANGE(size_t, i, 0, conf.std_value_size()) { std_value[i] = conf.std_value(i); }
  }

  if (conf.data_format() == "channels_last") {
    int64_t channel_dim = shape.NumAxes() - 1;
    CHECK_EQ(shape.At(channel_dim), conf.mean_value_size());
    FOR_RANGE(size_t, i, 0, shape.Count(1, channel_dim)) {
      FOR_RANGE(size_t, j, 0, shape.At(channel_dim)) {
        (*dptr) = ((*dptr) - conf.mean_value(j)) / std_value[j];
        ++dptr;
      }
    }
  } else if (conf.data_format() == "channels_first") {
    CHECK_EQ(shape.At(1), conf.mean_value_size());
    FOR_RANGE(size_t, i, 0, shape.At(1)) {
      FOR_RANGE(size_t, j, 0, shape.Count(2)) {
        (*dptr) = ((*dptr) - conf.mean_value(i)) / std_value[i];
        ++dptr;
      }
    }
  } else if (conf.data_format() == "no_channel") {
    CHECK_EQ(conf.mean_value_size(), 1);
    FOR_RANGE(size_t, i, 0, shape.Count(1)) {
      (*dptr) = ((*dptr) - conf.mean_value(0)) / std_value[0];
      ++dptr;
    }
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
void DoScalePreprocess(const ScalePreprocessConf& conf, T* dptr, int64_t n) {
  FOR_RANGE(size_t, i, 0, n) { (*(dptr++)) *= conf.value(); }
}

}  // namespace

template<typename T>
void DoPreprocess(const PreprocessConf& conf, T* dptr, const ShapeView& shape) {
  int64_t n = shape.Count(1);
  if (conf.has_subtract_conf()) {
    DoSubtractPreprocess<T>(conf.subtract_conf(), dptr, n);
  } else if (conf.has_norm_by_channel_conf()) {
    DoNormByChannelPreprocess<T>(conf.norm_by_channel_conf(), dptr, shape);
  } else if (conf.has_scale_conf()) {
    DoScalePreprocess<T>(conf.scale_conf(), dptr, n);
  } else {
    UNIMPLEMENTED();
  }
}

template<EncodeCase encode_case, typename T>
int32_t OFRecordDecoder<encode_case, T>::DecodeOneCol(
    DeviceCtx* ctx, Blob* in_blob, const BlobConf& blob_conf, int32_t col_id, Blob* out_blob,
    std::function<int32_t(void)> NextRandomInt) const {
  int32_t max_col_id = 0;
  ReadDataContent(ctx, in_blob, blob_conf, col_id, out_blob, NextRandomInt);
  return max_col_id;
}

template<EncodeCase encode_case, typename T>
int32_t OFRecordDecoder<encode_case, T>::ReadColNum(DeviceCtx* ctx, Blob* in_blob,
                                                    const std::string& name, Blob* out_blob) const {
  int32_t i = 0;
  int32_t max_col_num = 0;
  RecordBlob<OFRecord> record_blob(in_blob);
  record_blob.ForEachRecord([&](const OFRecord& record) {
    const Feature& feature = record.feature().at(name);
    int32_t col_num = GetColNumOfFeature(feature, out_blob->shape().Count(1));
    max_col_num = std::max(max_col_num, col_num);
  });
  CHECK_GT(max_col_num, 0);
  return max_col_num;
}

template<EncodeCase encode_case, typename T>
void OFRecordDecoder<encode_case, T>::ReadDataId(DeviceCtx* ctx, Blob* in_blob,
                                                 Blob* out_blob) const {
  // do nothing
}

template<EncodeCase encode_case, typename T>
void OFRecordDecoder<encode_case, T>::ReadDataContent(
    DeviceCtx* ctx, Blob* in_blob, const BlobConf& blob_conf, int32_t col_id, Blob* out_blob,
    std::function<int32_t(void)> NextRandomInt) const {
  RecordBlob<OFRecord> record_blob(in_blob);
  int64_t one_col_elem_num = out_blob->shape().Count(1);
  int32_t random_seed = NextRandomInt();
  ThreadPool* thread_pool = Global<ThreadMgr>::Get()->compute_thread_pool();
  int32_t thread_num = thread_pool->thread_num();
  int32_t part_num = std::min(record_blob.record_num(), thread_num);
  if (part_num >= 2) {
    BlockingCounter bc(part_num);
    FOR_RANGE(int32_t, part_id, 0, part_num) {
      thread_pool->AddWork([&ctx, &in_blob, &blob_conf, &col_id, &out_blob, &bc, part_id, &part_num,
                            &one_col_elem_num, &random_seed, this]() {
        ReadPartDataContent(ctx, in_blob, blob_conf, col_id, out_blob, part_id, part_num,
                            one_col_elem_num, random_seed);
        bc.Decrease();
      });
    }
    bc.WaitUntilCntEqualZero();
  } else {
    ReadPartDataContent(ctx, in_blob, blob_conf, col_id, out_blob, 0, 1, one_col_elem_num,
                        random_seed);
  }
  int64_t left_row_num = out_blob->shape().At(0) - record_blob.record_num();
  if (left_row_num > 0) {
    Memset<DeviceType::kCPU>(ctx,
                             out_blob->mut_dptr<T>() + record_blob.record_num() * one_col_elem_num,
                             0, left_row_num * one_col_elem_num * sizeof(T));
  }
}

template<EncodeCase encode_case, typename T>
void OFRecordDecoder<encode_case, T>::ReadPartDataContent(
    DeviceCtx* ctx, Blob* in_blob, const BlobConf& blob_conf, int32_t col_id, Blob* out_blob,
    int32_t part_id, int32_t part_num, int64_t one_col_elem_num, int32_t random_seed) const {
  RecordBlob<OFRecord> record_blob(in_blob);
  BalancedSplitter bs(record_blob.record_num(), part_num);
  Range range = bs.At(part_id);
  std::mt19937 gen(random_seed + part_id);
  std::uniform_int_distribution<int32_t> distribution;
  FOR_RANGE(int32_t, i, range.begin(), range.end()) {
    const OFRecord& record = record_blob.GetRecord(i);
    CHECK(record.feature().find(blob_conf.name()) != record.feature().end())
        << "Field " << blob_conf.name() << " not found";
    const Feature& feature = record.feature().at(blob_conf.name());
    T* out_dptr = out_blob->mut_dptr<T>() + i * one_col_elem_num;
    ReadOneCol(ctx, feature, blob_conf, col_id, out_dptr, one_col_elem_num,
               [&]() { return distribution(gen); });
    FOR_RANGE(size_t, j, 0, blob_conf.preprocess_size()) {
      DoPreprocess<T>(blob_conf.preprocess(j), out_dptr, out_blob->shape());
    }
  }
}

OFRecordDecoderIf* GetOFRecordDecoder(EncodeCase encode_case, DataType data_type) {
  static const HashMap<std::string, OFRecordDecoderIf*> obj = {
#define MAKE_ENTRY(et, dt) \
  {GetHashKey(et, OF_PP_PAIR_SECOND(dt)), new OFRecordDecoderImpl<et, OF_PP_PAIR_FIRST(dt)>},
      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, ENCODE_CASE_DATA_TYPE_SEQ_PRODUCT)};
#undef MAKE_ENTRY
  return obj.at(GetHashKey(encode_case, data_type));
}

}  // namespace oneflow
