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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/user/image/image_util.h"
#include <opencv2/opencv.hpp>
#include <cfenv>

namespace oneflow {

namespace {

enum class FlipCode : int8_t {
  kNonFlip = 0x00,
  kHorizontalFlip = 0x01,
  kVerticalFlip = 0x10,
  kBothDirectionFlip = 0x11,
};

bool operator&(FlipCode lhs, FlipCode rhs) {
  return static_cast<bool>(static_cast<std::underlying_type<FlipCode>::type>(lhs)
                           & static_cast<std::underlying_type<FlipCode>::type>(rhs));
}

int CvFlipCode(FlipCode flip_code) {
  if (flip_code == FlipCode::kHorizontalFlip) {
    return 1;
  } else if (flip_code == FlipCode::kVerticalFlip) {
    return 0;
  } else if (flip_code == FlipCode::kBothDirectionFlip) {
    return -1;
  } else {
    UNIMPLEMENTED();
  }
}

void FlipImage(TensorBuffer* image_buffer, FlipCode flip_code) {
  cv::Mat image_mat = GenCvMat4ImageBuffer(*image_buffer);
  cv::flip(image_mat, image_mat, CvFlipCode(flip_code));
}

template<typename T>
void FlipBoxes(TensorBuffer* boxes_buffer, int32_t image_width, int32_t image_height,
               FlipCode flip_code) {
  int num_boxes = boxes_buffer->shape().At(0);
  FOR_RANGE(int, i, 0, num_boxes) {
    T* cur_box_ptr = boxes_buffer->mut_data<T>() + i * 4;
    if (flip_code & FlipCode::kHorizontalFlip) {
      T xmin = cur_box_ptr[0];
      T xmax = cur_box_ptr[2];
      cur_box_ptr[0] = image_width - xmax - static_cast<T>(1);
      cur_box_ptr[2] = image_width - xmin - static_cast<T>(1);
    }
    if (flip_code & FlipCode::kVerticalFlip) {
      T ymin = cur_box_ptr[1];
      T ymax = cur_box_ptr[3];
      cur_box_ptr[1] = image_height - ymax - static_cast<T>(1);
      cur_box_ptr[3] = image_height - ymin - static_cast<T>(1);
    }
  }
}

#define MAKE_FLIP_BOXES_SWITCH_ENTRY(func_name, T) func_name<T>
DEFINE_STATIC_SWITCH_FUNC(void, FlipBoxes, MAKE_FLIP_BOXES_SWITCH_ENTRY,
                          MAKE_DATA_TYPE_CTRV_SEQ(FLOATING_DATA_TYPE_SEQ));

#undef MAKE_FLIP_BOXES_SWITCH_ENTRY

template<typename T>
void ScaleBoxes(TensorBuffer* boxes_buffer, T scale_w, T scale_h) {
  int num_boxes = boxes_buffer->shape().At(0);
  FOR_RANGE(int, i, 0, num_boxes) {
    T* cur_box_ptr = boxes_buffer->mut_data<T>() + i * 4;
    cur_box_ptr[0] *= scale_w;
    cur_box_ptr[1] *= scale_h;
    cur_box_ptr[2] *= scale_w;
    cur_box_ptr[3] *= scale_h;
  }
}

#define MAKE_SCALE_BOXES_SWITCH_ENTRY(func_name, T) func_name<T>
DEFINE_STATIC_SWITCH_FUNC(void, ScaleBoxes, MAKE_SCALE_BOXES_SWITCH_ENTRY,
                          MAKE_DATA_TYPE_CTRV_SEQ(FLOATING_DATA_TYPE_SEQ));

#undef MAKE_SCALE_BOXES_SWITCH_ENTRY

template<typename T>
void FlipPolygons(TensorBuffer* polygons_buffer, int32_t image_width, int32_t image_height,
                  FlipCode flip_code) {
  int num_points = polygons_buffer->shape().At(0);
  FOR_RANGE(int, i, 0, num_points) {
    T* cur_poly_ptr = polygons_buffer->mut_data<T>() + i * 2;
    if (flip_code & FlipCode::kHorizontalFlip) { cur_poly_ptr[0] = image_width - cur_poly_ptr[0]; }
    if (flip_code & FlipCode::kVerticalFlip) { cur_poly_ptr[1] = image_height - cur_poly_ptr[1]; }
  }
}

#define MAKE_FLIP_POLYGONS_SWITCH_ENTRY(func_name, T) func_name<T>
DEFINE_STATIC_SWITCH_FUNC(void, FlipPolygons, MAKE_FLIP_POLYGONS_SWITCH_ENTRY,
                          MAKE_DATA_TYPE_CTRV_SEQ(FLOATING_DATA_TYPE_SEQ));

#undef MAKE_FLIP_POLYGONS_SWITCH_ENTRY

template<typename T>
void ScalePolygons(TensorBuffer* poly_buffer, T scale_w, T scale_h) {
  int num_pts = poly_buffer->shape().At(0);
  FOR_RANGE(int, i, 0, num_pts) {
    T* cur_pt = poly_buffer->mut_data<T>() + i * 2;
    cur_pt[0] *= scale_w;
    cur_pt[1] *= scale_h;
  }
}

#define MAKE_SCALE_POLYGONS_SWITCH_ENTRY(func_name, T) func_name<T>
DEFINE_STATIC_SWITCH_FUNC(void, ScalePolygons, MAKE_SCALE_POLYGONS_SWITCH_ENTRY,
                          MAKE_DATA_TYPE_CTRV_SEQ(FLOATING_DATA_TYPE_SEQ));

#undef MAKE_SCALE_POLYGONS_SWITCH_ENTRY

template<typename T>
void ImageNormalizeByChannel(TensorBuffer* image_buffer, const std::vector<float>& std_vec,
                             const std::vector<float>& mean_vec) {
  CHECK_EQ(image_buffer->shape().NumAxes(), 3);
  int h = image_buffer->shape().At(0);
  int w = image_buffer->shape().At(1);
  int c = image_buffer->shape().At(2);
  CHECK_EQ(std_vec.size(), c);
  CHECK_EQ(mean_vec.size(), c);
  FOR_RANGE(int, i, 0, (h * w)) {
    T* image_data = image_buffer->mut_data<T>() + i * c;
    FOR_RANGE(int, j, 0, c) { image_data[j] = (image_data[j] - mean_vec.at(j)) / std_vec.at(j); }
  }
}

#define MAKE_IMAGE_NORMALIZE_SWITCH_ENTRY(func_name, T) func_name<T>
DEFINE_STATIC_SWITCH_FUNC(void, ImageNormalizeByChannel, MAKE_IMAGE_NORMALIZE_SWITCH_ENTRY,
                          MAKE_DATA_TYPE_CTRV_SEQ(FLOATING_DATA_TYPE_SEQ));

#undef MAKE_IMAGE_NORMALIZE_SWITCH_ENTRY

template<typename T>
void Rle4DensePoints(std::vector<std::vector<cv::Point_<T>>> &poly_point_vec,
                     std::vector<std::vector<int32_t>> &upsample_poly_point_vec,
                     int32_t im_w, int32_t im_h) {
  /* upsample and get discrete points densely along entire boundary */
  double scale = 5;
  std::vector<int32_t> x_vec, y_vec;
  std::vector<int32_t> up_x_vec, up_y_vec;
  FOR_RANGE(int, i, 0, poly_point_vec.size()) {
    std::vector<cv::Point_<T>>* poly_point = &poly_point_vec[i];
    size_t num_points = poly_point->size();
    FOR_RANGE(int, j, 0, num_points) {
      x_vec.push_back(static_cast<int>(scale * (*poly_point)[j].x + .5));
      y_vec.push_back(static_cast<int>(scale * (*poly_point)[j].y + .5));
    }
    x_vec.push_back(x_vec[0]);
    y_vec.push_back(y_vec[0]);
    FOR_RANGE(int, j, 0, num_points) {
      int x_start = x_vec[j], x_end = x_vec[j+1], y_start = y_vec[j], y_end = y_vec[j+1], temp = 0;
      int dx = std::abs(x_end - x_start);
      int dy = std::abs(y_start - y_end);
      bool flip = (dx >= dy && x_start > x_end) || (dx < dy && y_start > y_end);
      if (flip) {
        temp = x_start;
        x_start = x_end;
        x_end = temp;
        temp = y_start;
        y_start = y_end;
        y_end = temp;
      }
      double point_scale = dx >= dy ? static_cast<double>(y_end - y_start) / dx : static_cast<double>(x_end - x_start) / dy;
      if (dx >= dy) {
        FOR_RANGE(int, d, 0, dx + 1) {
          temp = flip ? dx - d : d;
          up_x_vec.push_back(temp + x_start);
          up_y_vec.push_back(static_cast<int>(y_start + point_scale * temp + .5));
        }
      } else {
        FOR_RANGE(int, d, 0, dy + 1) {
          temp = flip ? dy - d : d;
          up_y_vec.push_back(temp + y_start);
          up_x_vec.push_back(static_cast<int>(x_start + point_scale * temp + .5));
        }
      }
    }
    x_vec.clear();
    y_vec.clear();
    /* get points along y-boundary and downsample */
    FOR_RANGE(int, j, 1, up_y_vec.size()) {
      if (up_y_vec[j] != up_y_vec[j - 1]) {
        double down_y = static_cast<double>(up_y_vec[j] < up_y_vec[j-1] ? up_y_vec[j] : up_y_vec[j] - 1);
        down_y = (down_y + .5) / scale - .5;
        if (floor(down_y) != down_y || down_y < 0 || down_y > im_h - 1) { continue; }
        double down_x = static_cast<double>(up_x_vec[j] < up_x_vec[j-1] ? up_x_vec[j] : up_x_vec[j - 1]);
        down_x = (down_x + .5) / scale - .5;
        if (down_x < 0) { 
          down_x = 0;
        } else if (down_x > im_w) { down_x = im_w; }
        down_x = ceil(down_x);
        x_vec.push_back(static_cast<int>(down_x));
        y_vec.push_back(static_cast<int>(down_y));
      }
    }
    /* compute rle encoding given x-boundary points */
    std::vector<int32_t> temp_rle_vec;
    FOR_RANGE(int, j, 0, y_vec.size()) {
      temp_rle_vec.push_back(static_cast<int>(y_vec[j] * static_cast<int>(im_w) + x_vec[j]));
    }
    temp_rle_vec.push_back(static_cast<int>(im_h * im_w));
    x_vec.clear();
    y_vec.clear();
    up_x_vec.clear();
    up_y_vec.clear();
    std::sort(temp_rle_vec.begin(), temp_rle_vec.end());
    unsigned int p = 0;
    size_t rle_size = temp_rle_vec.size();
    FOR_RANGE(int, j, 0, rle_size) {
      unsigned int temp = temp_rle_vec[j];
      temp_rle_vec[j] -= p;
      p = temp;
    }
    int j = 0;
    upsample_poly_point_vec[i].push_back(temp_rle_vec[j++]);
    while (j < rle_size) {
      if (temp_rle_vec[j] > 0) {
        upsample_poly_point_vec[i].push_back(temp_rle_vec[j++]);
      } else {
        ++j;
        if (j < rle_size) {
          upsample_poly_point_vec[i].back() += temp_rle_vec[j++];
        }
      }
    }
    temp_rle_vec.clear();
  }
}

std::vector<uchar> RleDecode(std::vector<int32_t> &rle_encode_vec) {
  uchar v = 0;
  std::vector<uchar> mask_vec;
  FOR_RANGE(int, i, 0, rle_encode_vec.size()) {
    FOR_RANGE(int, j, 0, rle_encode_vec[i]) {
      mask_vec.push_back(v);
    }
    v = !v;
  }
  return mask_vec;
}

std::vector<int32_t> RleMerge(std::vector<std::vector<int32_t>> &upsample_poly_point_vec) {
  if (upsample_poly_point_vec.size() == 1) {
    return upsample_poly_point_vec[0];
  }
  std::vector<int32_t> merge_rle_vec;
  std::vector<int32_t> rle_encode_vec = upsample_poly_point_vec[0];
  FOR_RANGE(int, i, 1, upsample_poly_point_vec.size()) {
    std::vector<int32_t> temp_encode_vec = upsample_poly_point_vec[i];
    if (temp_encode_vec.size() == 0) { break; }
    unsigned int ct = 1, ca = rle_encode_vec[0], cb = temp_encode_vec[0], cc = 0;
    size_t a = 1, b = 1;
    int v = 0, vp = 0, va = 0, vb = 0;
    while (ct > 0) {
      unsigned int c = std::min(ca, cb);
      cc += c;
      ct = 0;

      ca -= c;
      if (!ca && a < rle_encode_vec.size()) {
        ca = rle_encode_vec[a++];
        va = !va;
      }
      ct += ca;

      cb -= c;
      if (!cb && b < temp_encode_vec.size()) {
        cb = temp_encode_vec[b++];
        vb = !vb;
      }
      ct += cb;

      vp = v;
      v = va || vb;
      if (v != vp || ct == 0) {
        merge_rle_vec.push_back(cc);
        cc = 0;
      }
    }
  }
  return merge_rle_vec;
}

template<typename T, typename I>
void PolygonsToMask(const TensorBuffer& polys, const TensorBuffer& polys_nd_index,
                    TensorBuffer* masks, int32_t im_w, int32_t im_h) {
  CHECK_EQ(polys.shape().NumAxes(), 2);
  CHECK_EQ(polys.shape().At(1), 2);
  CHECK_EQ(polys_nd_index.shape().NumAxes(), 2);
  CHECK_EQ(polys_nd_index.shape().At(1), 3);
  int num_points = polys.shape().At(0);
  CHECK_EQ(polys_nd_index.shape().At(0), num_points);

  std::vector<std::vector<cv::Point_<T>>> poly_point_vec;
  std::vector<std::vector<int32_t>> upsample_poly_point_vec;
  std::vector<cv::Mat> mask_mat_vec;
  auto PolyToMask = [&]() {
    CHECK_GT(poly_point_vec.size(), 0);
    CHECK_GT(poly_point_vec.front().size(), 0);
    Rle4DensePoints<T>(poly_point_vec, upsample_poly_point_vec, im_w, im_h);
    std::vector<int32_t> rle_encode_vec = RleMerge(upsample_poly_point_vec);
    std::vector<uchar> mask_vec = RleDecode(rle_encode_vec);
    cv::Mat mask_mat = cv::Mat(mask_vec, true).reshape(1, im_h);
    mask_mat_vec.emplace_back(std::move(mask_mat));
    poly_point_vec.clear();
    upsample_poly_point_vec.clear();
  };

  int origin_round_way = std::fegetround();
  CHECK_EQ(std::fesetround(FE_TONEAREST), 0);
  FOR_RANGE(int, i, 0, num_points) {
    const I pt_idx = polys_nd_index.data<I>()[i * 3 + 0];
    const I poly_idx = polys_nd_index.data<I>()[i * 3 + 1];
    const I segm_idx = polys_nd_index.data<I>()[i * 3 + 2];
    if (segm_idx != mask_mat_vec.size()) { PolyToMask(); }
    if (poly_idx == poly_point_vec.size()) {
      poly_point_vec.emplace_back(std::vector<cv::Point_<T>>());
      upsample_poly_point_vec.emplace_back(std::vector<int32_t>());
    }
    CHECK_EQ(segm_idx, mask_mat_vec.size());
    CHECK_EQ(poly_idx, poly_point_vec.size() - 1);
    CHECK_EQ(pt_idx, poly_point_vec.back().size());
    const T* pts_ptr = polys.data<T>() + i * 2;
    cv::Point_<T> pt{pts_ptr[0], pts_ptr[1]};
    poly_point_vec.back().emplace_back(std::move(pt));
  }
  PolyToMask();
  CHECK_EQ(std::fesetround(origin_round_way), 0);

  masks->Resize(Shape({static_cast<int64_t>(mask_mat_vec.size()), static_cast<int64_t>(im_h),
                       static_cast<int64_t>(im_w)}),
                DataType::kInt8);
  int mask_idx = 0;
  for (const auto& mask_mat : mask_mat_vec) {
    CHECK(mask_mat.isContinuous());
    CHECK_EQ(mask_mat.total(), im_h * im_w);
    memcpy(masks->mut_data<int8_t>() + mask_idx * im_h * im_w, mask_mat.ptr<int8_t>(),
           mask_mat.total() * sizeof(int8_t));
    mask_idx += 1;
  }
}

#define MAKE_POLYGONS_TO_MASK_SWITCH_ENTRY(func_name, T, I) func_name<T, I>
DEFINE_STATIC_SWITCH_FUNC(void, PolygonsToMask, MAKE_POLYGONS_TO_MASK_SWITCH_ENTRY,
                          MAKE_DATA_TYPE_CTRV_SEQ(FLOATING_DATA_TYPE_SEQ),
                          MAKE_DATA_TYPE_CTRV_SEQ(INDEX_DATA_TYPE_SEQ));

#undef MAKE_POLYGONS_TO_MASK_SWITCH_ENTRY

}  // namespace

class ImageFlipKernel final : public user_op::OpKernel {
 public:
  ImageFlipKernel() = default;
  ~ImageFlipKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* flip_code_tensor = ctx->Tensor4ArgNameAndIndex("flip_code", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    int num_images = in_tensor->shape().elem_cnt();
    CHECK_EQ(out_tensor->shape().elem_cnt(), num_images);

    MultiThreadLoop(num_images, [&](size_t i) {
      const TensorBuffer& in_buffer = in_tensor->dptr<TensorBuffer>()[i];
      CHECK_EQ(in_buffer.shape().NumAxes(), 3);
      TensorBuffer* out_buffer = out_tensor->mut_dptr<TensorBuffer>() + i;
      out_buffer->CopyFrom(in_buffer);
      FlipCode flip_code = static_cast<FlipCode>(flip_code_tensor->dptr<int8_t>()[i]);
      if (flip_code != FlipCode::kNonFlip) { FlipImage(out_buffer, flip_code); }
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class ObjectBboxFlipKernel final : public user_op::OpKernel {
 public:
  ObjectBboxFlipKernel() = default;
  ~ObjectBboxFlipKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* bbox_tensor = ctx->Tensor4ArgNameAndIndex("bbox", 0);
    const user_op::Tensor* image_size_tensor = ctx->Tensor4ArgNameAndIndex("image_size", 0);
    const user_op::Tensor* flip_code_tensor = ctx->Tensor4ArgNameAndIndex("flip_code", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);

    int num_images = bbox_tensor->shape().elem_cnt();
    CHECK_GT(num_images, 0);
    CHECK_EQ(out_tensor->shape().elem_cnt(), num_images);
    CHECK_EQ(image_size_tensor->shape().At(0), num_images);
    CHECK_EQ(flip_code_tensor->shape().elem_cnt(), num_images);

    MultiThreadLoop(num_images, [&](size_t i) {
      const TensorBuffer& bbox_buffer = bbox_tensor->dptr<TensorBuffer>()[i];
      CHECK_EQ(bbox_buffer.shape().NumAxes(), 2);
      CHECK_EQ(bbox_buffer.shape().At(1), 4);
      TensorBuffer* out_bbox_buffer = out_tensor->mut_dptr<TensorBuffer>() + i;
      out_bbox_buffer->CopyFrom(bbox_buffer);
      int32_t image_width = image_size_tensor->dptr<int32_t>()[i * 2 + 0];
      int32_t image_height = image_size_tensor->dptr<int32_t>()[i * 2 + 1];
      FlipCode flip_code = static_cast<FlipCode>(flip_code_tensor->dptr<int8_t>()[i]);
      SwitchFlipBoxes(SwitchCase(out_bbox_buffer->data_type()), out_bbox_buffer, image_width,
                      image_height, flip_code);
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class ObjectBboxScaleKernel final : public user_op::OpKernel {
 public:
  ObjectBboxScaleKernel() = default;
  ~ObjectBboxScaleKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* bbox_tensor = ctx->Tensor4ArgNameAndIndex("bbox", 0);
    const user_op::Tensor* scale_tensor = ctx->Tensor4ArgNameAndIndex("scale", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);

    int num_images = bbox_tensor->shape().elem_cnt();
    CHECK_GT(num_images, 0);
    CHECK_EQ(scale_tensor->shape().At(0), num_images);
    CHECK_EQ(out_tensor->shape().elem_cnt(), num_images);

    MultiThreadLoop(num_images, [&](size_t i) {
      const TensorBuffer& bbox_buffer = bbox_tensor->dptr<TensorBuffer>()[i];
      CHECK_EQ(bbox_buffer.shape().NumAxes(), 2);
      CHECK_EQ(bbox_buffer.shape().At(1), 4);
      TensorBuffer* out_bbox_buffer = out_tensor->mut_dptr<TensorBuffer>() + i;
      out_bbox_buffer->CopyFrom(bbox_buffer);
      float scale_w = scale_tensor->dptr<float>()[i * 2 + 0];
      float scale_h = scale_tensor->dptr<float>()[i * 2 + 1];
      SwitchScaleBoxes(SwitchCase(out_bbox_buffer->data_type()), out_bbox_buffer, scale_w, scale_h);
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class ObjectSegmentationPolygonFlipKernel final : public user_op::OpKernel {
 public:
  ObjectSegmentationPolygonFlipKernel() = default;
  ~ObjectSegmentationPolygonFlipKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* polygon_tensor = ctx->Tensor4ArgNameAndIndex("poly", 0);
    const user_op::Tensor* image_size_tensor = ctx->Tensor4ArgNameAndIndex("image_size", 0);
    const user_op::Tensor* flip_code_tensor = ctx->Tensor4ArgNameAndIndex("flip_code", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);

    int num_images = polygon_tensor->shape().elem_cnt();
    CHECK_GT(num_images, 0);
    CHECK_EQ(out_tensor->shape().elem_cnt(), num_images);
    CHECK_EQ(image_size_tensor->shape().At(0), num_images);
    CHECK_EQ(flip_code_tensor->shape().elem_cnt(), num_images);

    MultiThreadLoop(num_images, [&](size_t i) {
      const TensorBuffer& polygons_buffer = polygon_tensor->dptr<TensorBuffer>()[i];
      CHECK_EQ(polygons_buffer.shape().NumAxes(), 2);
      CHECK_EQ(polygons_buffer.shape().At(1), 2);
      TensorBuffer* out_polygons_buffer = out_tensor->mut_dptr<TensorBuffer>() + i;
      out_polygons_buffer->CopyFrom(polygons_buffer);
      int32_t image_width = image_size_tensor->dptr<int32_t>()[i * 2 + 0];
      int32_t image_height = image_size_tensor->dptr<int32_t>()[i * 2 + 1];
      FlipCode flip_code = static_cast<FlipCode>(flip_code_tensor->dptr<int8_t>()[i]);
      SwitchFlipPolygons(SwitchCase(out_polygons_buffer->data_type()), out_polygons_buffer,
                         image_width, image_height, flip_code);
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class ObjectSegmentationPolygonScaleKernel final : public user_op::OpKernel {
 public:
  ObjectSegmentationPolygonScaleKernel() = default;
  ~ObjectSegmentationPolygonScaleKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* poly_tensor = ctx->Tensor4ArgNameAndIndex("poly", 0);
    const user_op::Tensor* scale_tensor = ctx->Tensor4ArgNameAndIndex("scale", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);

    int num_images = poly_tensor->shape().elem_cnt();
    CHECK_GT(num_images, 0);
    CHECK_EQ(scale_tensor->shape().At(0), num_images);
    CHECK_EQ(out_tensor->shape().elem_cnt(), num_images);

    MultiThreadLoop(num_images, [&](size_t i) {
      const TensorBuffer& poly_buffer = poly_tensor->dptr<TensorBuffer>()[i];
      CHECK_EQ(poly_buffer.shape().NumAxes(), 2);
      CHECK_EQ(poly_buffer.shape().At(1), 2);
      TensorBuffer* out_poly_buffer = out_tensor->mut_dptr<TensorBuffer>() + i;
      out_poly_buffer->CopyFrom(poly_buffer);
      float scale_w = scale_tensor->dptr<float>()[i * 2 + 0];
      float scale_h = scale_tensor->dptr<float>()[i * 2 + 1];
      SwitchScalePolygons(SwitchCase(out_poly_buffer->data_type()), out_poly_buffer, scale_w,
                          scale_h);
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class ImageNormalize final : public user_op::OpKernel {
 public:
  ImageNormalize() = default;
  ~ImageNormalize() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    int num_images = in_tensor->shape().elem_cnt();
    CHECK_EQ(out_tensor->shape().elem_cnt(), num_images);
    const auto& std_vec = ctx->Attr<std::vector<float>>("std");
    const auto& mean_vec = ctx->Attr<std::vector<float>>("mean");

    MultiThreadLoop(num_images, [&](size_t i) {
      const TensorBuffer& in_buffer = in_tensor->dptr<TensorBuffer>()[i];
      CHECK_EQ(in_buffer.shape().NumAxes(), 3);
      TensorBuffer* out_buffer = out_tensor->mut_dptr<TensorBuffer>() + i;
      out_buffer->CopyFrom(in_buffer);
      SwitchImageNormalizeByChannel(SwitchCase(out_buffer->data_type()), out_buffer, std_vec,
                                    mean_vec);
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class ObjectSegmentationPolygonToMask final : public user_op::OpKernel {
 public:
  ObjectSegmentationPolygonToMask() = default;
  ~ObjectSegmentationPolygonToMask() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* poly_tensor = ctx->Tensor4ArgNameAndIndex("poly", 0);
    const user_op::Tensor* poly_index_tensor = ctx->Tensor4ArgNameAndIndex("poly_index", 0);
    const user_op::Tensor* image_size_tensor = ctx->Tensor4ArgNameAndIndex("image_size", 0);
    user_op::Tensor* mask_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);

    int num_images = poly_tensor->shape().elem_cnt();
    CHECK_GT(num_images, 0);
    CHECK_EQ(poly_index_tensor->shape().elem_cnt(), num_images);
    CHECK_EQ(image_size_tensor->shape().At(0), num_images);
    CHECK_EQ(mask_tensor->shape().elem_cnt(), num_images);

    MultiThreadLoop(num_images, [&](size_t i) {
      const TensorBuffer& poly_buffer = poly_tensor->dptr<TensorBuffer>()[i];
      const TensorBuffer& poly_index_buffer = poly_index_tensor->dptr<TensorBuffer>()[i];
      int32_t image_width = image_size_tensor->dptr<int32_t>()[i * 2 + 0];
      int32_t image_height = image_size_tensor->dptr<int32_t>()[i * 2 + 1];
      TensorBuffer* mask_buffer = mask_tensor->mut_dptr<TensorBuffer>() + i;
      SwitchPolygonsToMask(SwitchCase(poly_buffer.data_type(), poly_index_buffer.data_type()),
                           poly_buffer, poly_index_buffer, mask_buffer, image_width, image_height);
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

namespace {

std::function<Maybe<void>(const user_op::InferContext&, user_op::AddInplaceArgPair)>
MakeInplaceProposalFn(const std::string& input_arg_name) {
  return [input_arg_name](const user_op::InferContext& ctx,
                          user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {
    OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, input_arg_name, 0, true));
    return Maybe<void>::Ok();
  };
}

}  // namespace

REGISTER_USER_KERNEL("image_flip")
    .SetCreateFn<ImageFlipKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("in", 0) == DataType::kTensorBuffer)
                     & (user_op::HobDataType("flip_code", 0) == DataType::kInt8)
                     & (user_op::HobDataType("out", 0) == DataType::kTensorBuffer))
    .SetInplaceProposalFn(MakeInplaceProposalFn("in"));

REGISTER_USER_KERNEL("object_bbox_flip")
    .SetCreateFn<ObjectBboxFlipKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("bbox", 0) == DataType::kTensorBuffer)
                     & (user_op::HobDataType("image_size", 0) == DataType::kInt32)
                     & (user_op::HobDataType("flip_code", 0) == DataType::kInt8)
                     & (user_op::HobDataType("out", 0) == DataType::kTensorBuffer))
    .SetInplaceProposalFn(MakeInplaceProposalFn("bbox"));

REGISTER_USER_KERNEL("object_bbox_scale")
    .SetCreateFn<ObjectBboxScaleKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("bbox", 0) == DataType::kTensorBuffer)
                     & (user_op::HobDataType("scale", 0) == DataType::kFloat)
                     & (user_op::HobDataType("out", 0) == DataType::kTensorBuffer))
    .SetInplaceProposalFn(MakeInplaceProposalFn("bbox"));

REGISTER_USER_KERNEL("object_segmentation_polygon_flip")
    .SetCreateFn<ObjectSegmentationPolygonFlipKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("poly", 0) == DataType::kTensorBuffer)
                     & (user_op::HobDataType("image_size", 0) == DataType::kInt32)
                     & (user_op::HobDataType("flip_code", 0) == DataType::kInt8)
                     & (user_op::HobDataType("out", 0) == DataType::kTensorBuffer))
    .SetInplaceProposalFn(MakeInplaceProposalFn("poly"));

REGISTER_USER_KERNEL("object_segmentation_polygon_scale")
    .SetCreateFn<ObjectSegmentationPolygonScaleKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("poly", 0) == DataType::kTensorBuffer)
                     & (user_op::HobDataType("scale", 0) == DataType::kFloat)
                     & (user_op::HobDataType("out", 0) == DataType::kTensorBuffer))
    .SetInplaceProposalFn(MakeInplaceProposalFn("poly"));

REGISTER_USER_KERNEL("image_normalize")
    .SetCreateFn<ImageNormalize>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("in", 0) == DataType::kTensorBuffer)
                     & (user_op::HobDataType("out", 0) == DataType::kTensorBuffer))
    .SetInplaceProposalFn(MakeInplaceProposalFn("in"));

REGISTER_USER_KERNEL("object_segmentation_polygon_to_mask")
    .SetCreateFn<ObjectSegmentationPolygonToMask>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("poly", 0) == DataType::kTensorBuffer)
                     & (user_op::HobDataType("poly_index", 0) == DataType::kTensorBuffer)
                     & (user_op::HobDataType("image_size", 0) == DataType::kInt32)
                     & (user_op::HobDataType("out", 0) == DataType::kTensorBuffer));

}  // namespace oneflow
