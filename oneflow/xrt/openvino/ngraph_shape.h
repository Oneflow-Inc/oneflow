#ifndef ONEFLOW_XRT_OPENVINO_NGRAPH_SHAPE_H_
#define ONEFLOW_XRT_OPENVINO_NGRAPH_SHAPE_H_

#include "glog/logging.h"

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {
namespace xrt {
namespace openvino {

inline ngraph::element::Type DataTypeToNgraphDataType(const DataType &data_type) {
  switch (data_type) {
    case oneflow::kDouble: return ngraph::element::f64;
    case oneflow::kFloat: return ngraph::element::f32;
    case oneflow::kFloat16: return ngraph::element::f16;
    case oneflow::kInt8: return ngraph::element::i8;
    case oneflow::kInt32: return ngraph::element::i32;
    case oneflow::kInt64: return ngraph::element::i64;
    default: {
      LOG(FATAL) << "Unsupported data type " << data_type << " for Ngraph.";
      return ngraph::element::f32;
    }
  }
}

inline ngraph::Shape ShapeToNgraphShape(const Shape &shape) {
  CHECK_LE(shape.NumAxes(), 8) << "The maximum dimensions is 8 supported by Ngraph.";
  std::vector<size_t> dim_vec;
  for (int i = 0; i < shape.NumAxes(); ++i) { dim_vec.push_back(shape.At(i)); }
  return ngraph::Shape(dim_vec);
}

class NgraphShape {
 public:
  NgraphShape() = default;

  NgraphShape(const Shape &shape, const DataType &data_type)
      : shape_(ShapeToNgraphShape(shape)), data_type_(DataTypeToNgraphDataType(data_type)) {}

  const ngraph::element::Type &data_type() const { return data_type_; }

  const ngraph::Shape &shape() const { return shape_; }

 private:
  ngraph::Shape shape_;
  ngraph::element::Type data_type_;
};

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_OPENVINO_NGRAPH_SHAPE_H_
