#ifndef ONEFLOW_CUSTOMIZED_IMAGE_CROP_WINDOW_H_
#define ONEFLOW_CUSTOMIZED_IMAGE_CROP_WINDOW_H_

#include "oneflow/core/common/shape.h"

namespace oneflow {

struct CropWindow {
  Shape anchor;
  Shape shape;

  CropWindow() : anchor{0, 0}, shape{0, 0} {}
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_IMAGE_CROP_WINDOW_H_
