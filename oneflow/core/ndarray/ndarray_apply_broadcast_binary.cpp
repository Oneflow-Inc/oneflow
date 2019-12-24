#include "oneflow/core/ndarray/ndarray_apply_broadcast_binary.h"

namespace oneflow {

void SimplifyBroadcastBinaryShapes(const XpuShape& y, const XpuShape& b, DimVector* simplified_y,
                                   DimVector* simplified_b) {
  DimVector simplified_a;
  SimplifyBroadcastBinaryShapes(y, y, b, simplified_y, &simplified_a, simplified_b);
}

void SimplifyBroadcastBinaryShapes(const XpuShape& y, const XpuShape& a, const XpuShape& b,
                                   DimVector* simplified_y, DimVector* simplified_a,
                                   DimVector* simplified_b) {
  CHECK_EQ(y.NumAxes(), a.NumAxes());
  CHECK_EQ(b.NumAxes(), a.NumAxes());
  CHECK(simplified_y->empty());
  CHECK(simplified_a->empty());
  CHECK(simplified_b->empty());
  simplified_y->push_back(y.At(0));
  simplified_a->push_back(a.At(0));
  simplified_b->push_back(b.At(0));
  bool a_prev_axis_is_broadcast = (a.At(0) == 1);
  bool b_prev_axis_is_broadcast = (b.At(0) == 1);
  FOR_RANGE(int, i, 1, y.NumAxes()) {
    const int64_t y_dim = y.At(i);
    const int64_t a_dim = a.At(i);
    const int64_t b_dim = b.At(i);
    if ((a_dim == 1) && (b_dim == 1)) { continue; }
    const bool a_cur_axis_is_broadcast = (a_dim == 1);
    const bool b_cur_axis_is_broadcast = (b_dim == 1);
    if (a_prev_axis_is_broadcast == a_cur_axis_is_broadcast
        && b_prev_axis_is_broadcast == b_cur_axis_is_broadcast) {
      simplified_y->back() *= y_dim;
      simplified_a->back() *= a_dim;
      simplified_b->back() *= b_dim;
    } else {
      simplified_y->push_back(y_dim);
      simplified_a->push_back(a_dim);
      simplified_b->push_back(b_dim);
    }
    a_prev_axis_is_broadcast = a_cur_axis_is_broadcast;
    b_prev_axis_is_broadcast = b_cur_axis_is_broadcast;
  }
}

}  // namespace oneflow
