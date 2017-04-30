#include <stdint.h>
#include <utility>
#include <functional>
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "common/shape.h"
#include "common/util.h"


namespace oneflow {


// check the parameter which user set in convolution layer and pooling layer
// is legal and return h and w
// Param:
//  val = [pad | kernel | stride]
//  val_h = [pad_h | kernel_h | stride_h]
//  val_w = [pad_w | kernel_w | stride_w]
// Return:
//  (val_h, val_w)
// if user both set the val and (val_h, val_w), a LOG(FATAL) will be called
// else return the real val_h and val_w
std::pair<uint32_t, uint32_t> CheckDimPara4CnnAndPooling(uint32_t val, uint32_t val_h, 
                                           uint32_t val_w) {
  if (val != 0 && (val_h != 0 || val_w != 0) && (val != val_h || val != val_w)) {
    LOG(FATAL) << "don't set parameter both 'val' and ('val_h','val_w')"
      << " in convolution_conf or pooling_conf ";
  }
  if (val != 0) {
    return {val, val};
  }
  return {val_h, val_w};
}

class TestShapeFactory {
 public:
   TestShapeFactory() = default;
   ~TestShapeFactory() = default;

   Shape* bn2ShapePtr(const std::string& bn) {
     CHECK_NE(bn2shape_ptr_.find(bn), bn2shape_ptr_.end());
     return bn2shape_ptr_.at(bn);
   }

   void add_bn_shape_ptr(const std::string& bn, Shape* shape_ptr) {
     bn2shape_ptr_.emplace(bn,shape_ptr);
   }

 private:
  HashMap<std::string, Shape*> bn2shape_ptr_;
};

} // namespace oneflow
