#ifndef ONEFLOW_USER_IMAGE_JPEG_DECODER_H_
#define ONEFLOW_USER_IMAGE_JPEG_DECODER_H_
#include <jpeglib.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

#include "oneflow/user/image/random_crop_generator.h"


namespace oneflow {

enum class JpegReturnType {
  kOk = 0,
  kError = 1,
};

class JpegDecoder {
public:
    JpegDecoder();
    ~JpegDecoder();
    OF_DISALLOW_COPY_AND_MOVE(JpegDecoder);

    JpegReturnType PartialDecode(const unsigned char* data, size_t length,
                                 RandomCropGenerator* random_crop_gen, unsigned char* workspace,
                                 size_t workspace_size, const std::string& color_space, 
                                 cv::Mat &out);

private:

    struct jpeg_decompress_struct cinfo_;
    struct jpeg_error_mgr jerr_;

    std::vector<unsigned char> tmp_buf_;

};

#endif
}  // namespace oneflow
