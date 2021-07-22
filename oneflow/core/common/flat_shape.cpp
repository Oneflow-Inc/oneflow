#include "oneflow/core/common/flat_shape.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

Maybe<void> FlatShape::Init(const std::shared_ptr<const Shape>& shape) {
	CHECK_LE_OR_RETURN(shape->NumAxes(), SHAPE_MAX_AXIS_SIZE);
	this->set_num_axes(shape->NumAxes());
	for (int i = 0; i < this->num_axes(); ++i) { *this->mutable_dim()->Mutable(i) = shape->At(i); }
	return Maybe<void>::Ok();
}

Maybe<void> FlatShape::Check(const std::shared_ptr<const Shape>& shape) const {
	CHECK_EQ_OR_RETURN(this->num_axes(), shape->NumAxes());
	for (int i = 0; i < this->num_axes(); ++i) {
		CHECK_EQ_OR_RETURN(this->dim().Get(i), shape->At(i));
	}
	return Maybe<void>::Ok();
}

}
