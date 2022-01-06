#include <iostream>

#include "oneflow/user/image/jpeg_decoder.h"

namespace oneflow {

JpegDecoder::JpegDecoder(): cinfo_(), jerr_(), tmp_buf_() {}

JpegDecoder::~JpegDecoder()
{
    jpeg_destroy_decompress(&cinfo_);
}



JpegReturnType JpegDecoder::PartialDecode(const unsigned char* data, size_t length,
                                 RandomCropGenerator* random_crop_gen, unsigned char* workspace,
                                 size_t workspace_size, const std::string& color_space, 
                                 cv::Mat &out_mat) {
//   struct jpeg_decompress_struct cinfo = {};
//   struct jpeg_error_mgr jerr = {};
//   int rc = 0;
//   unsigned char* crop_buf = nullptr;

  cinfo_.err = jpeg_std_error(&jerr_);
  jpeg_create_decompress(&cinfo_);
  if (cinfo_.err->msg_code != 0) { return JpegReturnType::kError; }

  jpeg_mem_src(&cinfo_, data, length);
  if (cinfo_.err->msg_code != 0) {
    // jpeg_destroy_decompress(&cinfo_);
    return JpegReturnType::kError;
  }

  int rc = jpeg_read_header(&cinfo_, TRUE);
  if (rc != 1) {
    // jpeg_destroy_decompress(&cinfo_);
    return JpegReturnType::kError;
  }

  jpeg_start_decompress(&cinfo_);
  int width = cinfo_.output_width;
  int height = cinfo_.output_height;
  int pixel_size = cinfo_.output_components;

  unsigned char* crop_buf = nullptr;
//   std::vector<unsigned char> tmp_buf;
  if (width * height * pixel_size > workspace_size) {
    tmp_buf_.resize(width * height * pixel_size);
    crop_buf = tmp_buf_.data();
  } else {
    crop_buf = workspace;
  }

//   std::vector<unsigned char> tmp_buf(width * height * pixel_size);
//   unsigned char* crop_buf = tmp_buf.data();

  unsigned int u_crop_x = 0, u_crop_y = 0, u_crop_w = 0, u_crop_h = 0;
  if (random_crop_gen) {
    CropWindow crop;
    random_crop_gen->GenerateCropWindow({height, width}, &crop);
    u_crop_y = crop.anchor.At(0);
    u_crop_x = crop.anchor.At(1);
    u_crop_h = crop.shape.At(0);
    u_crop_w = crop.shape.At(1);
  } else {
    u_crop_y = 0;
    u_crop_x = 0;
    u_crop_h = height;
    u_crop_w = width;
  }

  unsigned int tmp_w = u_crop_w;
  jpeg_crop_scanline(&cinfo_, &u_crop_x, &tmp_w);
  int row_stride = tmp_w * pixel_size;
  if (jpeg_skip_scanlines(&cinfo_, u_crop_y) != u_crop_y) {
    // jpeg_destroy_decompress(&cinfo_);
    return JpegReturnType::kError;
  }

  while (cinfo_.output_scanline < u_crop_y + u_crop_h) {
    unsigned char* buffer_array[1];
    buffer_array[0] = crop_buf + (cinfo_.output_scanline - u_crop_y) * row_stride;
    jpeg_read_scanlines(&cinfo_, buffer_array, 1);
  }

  jpeg_skip_scanlines(&cinfo_, cinfo_.output_height - u_crop_y - u_crop_h);
  jpeg_finish_decompress(&cinfo_);
//   jpeg_destroy_decompress(&cinfo_);

  cv::Mat image(u_crop_h, tmp_w, CV_8UC3, crop_buf, cv::Mat::AUTO_STEP);

  cv::Rect roi;
//   cv::Mat cropped;

  if (u_crop_w != tmp_w) {
    roi.x = tmp_w - u_crop_w;
    roi.y = 0;
    roi.width = u_crop_w;
    roi.height = u_crop_h;
    out_mat = image(roi);
  } else {
    out_mat = image;
  }

  return JpegReturnType::kOk;
}

//   // convert color space
//   if (ImageUtil::IsColor(color_space) && color_space != "RGB") {
//     ImageUtil::ConvertColor("RGB", cropped, color_space, cropped);
//   }

//   const int c = ImageUtil::IsColor(color_space) ? 3 : 1;
//   CHECK_EQ(c, cropped.channels());
//   Shape image_shape({u_crop_h, u_crop_w, c});
//   buffer->Resize(image_shape, DataType::kUInt8);
//   CHECK_EQ(image_shape.elem_cnt(), buffer->nbytes());
//   CHECK_EQ(image_shape.elem_cnt(), cropped.total() * cropped.elemSize());
//   memcpy(buffer->mut_data<uint8_t>(), cropped.ptr(), image_shape.elem_cnt());
//   return JpegReturnType::kOk;
// }

}  // namespace oneflow
