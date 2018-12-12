#include "oneflow/core/record/ofrecord_jpeg_decoder.h"
#include "oneflow/core/record/image_preprocess.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

namespace {

void ConvertChannel(cv::Mat* src, cv::Mat* dst, int32_t src_cn, int32_t dst_cn) {
  if (src_cn == dst_cn) { return; }

  if (src_cn == 3 && dst_cn == 1) {
    cv::cvtColor(*src, *dst, cv::COLOR_BGR2GRAY);
  } else if (src_cn == 1 && dst_cn == 3) {
    cv::cvtColor(*src, *dst, cv::COLOR_GRAY2BGR);
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace

template<typename T>
int32_t OFRecordDecoderImpl<EncodeCase::kJpeg, T>::GetColNumOfFeature(
    const Feature& feature, int64_t one_col_elem_num) const {
  return feature.bytes_list().value_size();
}

template<typename T>
void OFRecordDecoderImpl<EncodeCase::kJpeg, T>::ReadOneCol(
    DeviceCtx* ctx, const Feature& feature, const BlobConf& blob_conf, int32_t col_id, T* out_dptr,
    int64_t one_col_elem_num, std::function<int32_t(void)> NextRandomInt) const {
  CHECK(feature.has_bytes_list());
  const std::string& src_data = feature.bytes_list().value(col_id);
  cv::_InputArray image_data(src_data.data(), src_data.size());
  cv::Mat image = cv::imdecode(image_data, cv::IMREAD_ANYCOLOR);
  FOR_RANGE(size_t, i, 0, blob_conf.encode_case().jpeg().preprocess_size()) {
    ImagePreprocessIf* preprocess =
        GetImagePreprocess(blob_conf.encode_case().jpeg().preprocess(i).preprocess_case());
    preprocess->DoPreprocess(&image, blob_conf.encode_case().jpeg().preprocess(i), NextRandomInt);
  }
  CHECK_EQ(blob_conf.shape().dim_size(), 3);
  CHECK_EQ(blob_conf.shape().dim(0), image.rows);
  CHECK_EQ(blob_conf.shape().dim(1), image.cols);
  if (blob_conf.shape().dim(2) != image.channels()) {
    ConvertChannel(&image, &image, image.channels(), blob_conf.shape().dim(2));
  }
  CHECK_EQ(blob_conf.shape().dim(2), image.channels());
  CHECK_EQ(one_col_elem_num, image.total() * image.channels());

  if (image.isContinuous()) {
    CopyElem(image.data, out_dptr, one_col_elem_num);
  } else {
    FOR_RANGE(size_t, i, 0, image.rows) {
      int64_t one_row_size = image.cols * image.channels();
      CopyElem(image.ptr<uint8_t>(i), out_dptr, one_row_size);
      out_dptr += one_row_size;
    }
  }
}

template<typename T>
void OFRecordDecoderImpl<EncodeCase::kJpeg, T>::ReadDynamicDataContent(
    DeviceCtx* ctx, Blob* in_blob, const BlobConf& blob_conf, int32_t col_id, Blob* out_blob,
    std::function<int32_t(void)> NextRandomInt) const {
  RecordBlob<OFRecord> record_blob(in_blob);
  int32_t random_seed = NextRandomInt();
  int32_t n = record_blob.record_num();
  // read images
  std::vector<cv::Mat> images(n, cv::Mat());
  ThreadPool thread_pool(std::thread::hardware_concurrency() / 4);
  BlockingCounter decode_cnt(n);
  FOR_RANGE(int32_t, i, 0, n) {
    thread_pool.AddWork([&]() {
      RecordBlob<OFRecord> record_blob(in_blob);
      std::mt19937 gen(random_seed + i);
      std::uniform_int_distribution<int32_t> distribution;
      const OFRecord& record = record_blob.GetRecord(i);
      CHECK(record.feature().find(blob_conf.name()) != record.feature().end())
          << "Field " << blob_conf.name() << " not found";
      const Feature& feature = record.feature().at(blob_conf.name());
      CHECK(feature.has_bytes_list());
      const std::string& src_data = feature.bytes_list().value(col_id);
      cv::_InputArray image_data(src_data.data(), src_data.size());
      images[i] = cv::imdecode(image_data, cv::IMREAD_ANYCOLOR);
      FOR_RANGE(size_t, j, 0, blob_conf.encode_case().jpeg().preprocess_size()) {
        ImagePreprocessIf* preprocess =
            GetImagePreprocess(blob_conf.encode_case().jpeg().preprocess(j).preprocess_case());
        preprocess->DoPreprocess(&(images[i]), blob_conf.encode_case().jpeg().preprocess(j),
                                 [&]() { return distribution(gen); });
      }
      CHECK_EQ(blob_conf.shape().dim_size(), 3);
      if (blob_conf.shape().dim(2) != images[i].channels()) {
        ConvertChannel(&images[i], &images[i], images[i].channels(), blob_conf.shape().dim(2));
      }
      CHECK_EQ(blob_conf.shape().dim(2), images[i].channels());
      decode_cnt.Decrease();
    });
  }
  decode_cnt.WaitUntilCntEqualZero();
  // cal dynamic shape
  int64_t max_rows = -1;
  int64_t max_cols = -1;
  FOR_RANGE(int32_t, i, 0, n) {
    max_rows = std::max(max_rows, static_cast<int64_t>(images[i].rows));
    max_cols = std::max(max_cols, static_cast<int64_t>(images[i].cols));
  }
  CHECK_GT(max_rows, 0);
  CHECK_GT(max_cols, 0);
  CHECK_LE(max_rows, blob_conf.shape().dim(0));
  CHECK_LE(max_cols, blob_conf.shape().dim(1));
  Shape instance_shape({max_rows, max_cols, blob_conf.shape().dim(2)});
  out_blob->set_instance_shape(instance_shape);
  int64_t one_col_elem_num = instance_shape.elem_cnt();

  BlockingCounter set_cnt(n);
  FOR_RANGE(int32_t, i, 0, n) {
    thread_pool.AddWork([&]() {
      cv::Mat dst = cv::Mat::zeros(cv::Size(max_cols, max_rows), images[i].type());
      images[i].copyTo(dst(cv::Rect(0, 0, images[i].cols, images[i].rows)));
      CHECK_EQ(one_col_elem_num, dst.total() * dst.channels());
      CHECK(dst.isContinuous());
      T* out_dptr = out_blob->mut_dptr<T>() + i * one_col_elem_num;
      CopyElem(dst.data, out_dptr, one_col_elem_num);
      FOR_RANGE(size_t, j, 0, blob_conf.preprocess_size()) {
        DoPreprocess<T>(blob_conf.preprocess(j), out_dptr, out_blob->shape());
      }
      set_cnt.Decrease();
    });
  }
  set_cnt.WaitUntilCntEqualZero();

  int64_t used_elem_cnt = one_col_elem_num * n;
  int64_t left_elem_cnt = out_blob->static_shape().elem_cnt() - used_elem_cnt;
  if (left_elem_cnt > 0) {
    Memset<DeviceType::kCPU>(ctx, out_blob->mut_dptr<T>() + used_elem_cnt, 0,
                             left_elem_cnt * sizeof(T));
  }
}

#define INSTANTIATE_OFRECORD_JPEG_DECODER(type_cpp, type_proto) \
  template class OFRecordDecoderImpl<EncodeCase::kJpeg, type_cpp>;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_OFRECORD_JPEG_DECODER, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
