/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/record/ofrecord_jpeg_decoder.h"
#include "oneflow/core/record/image_preprocess.h"

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

#define INSTANTIATE_OFRECORD_JPEG_DECODER(type_cpp, type_proto) \
  template class OFRecordDecoderImpl<EncodeCase::kJpeg, type_cpp>;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_OFRECORD_JPEG_DECODER, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
