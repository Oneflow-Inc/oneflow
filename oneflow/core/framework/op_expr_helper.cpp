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

#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/operator/op_attribute.cfg.h"

namespace oneflow {

namespace op_expr_helper {

Maybe<one::UserOpExpr> AddNOp(int32_t n) {
  return one::OpBuilder("add_n").Input("in", n).Output("out").Build();
}

Maybe<one::UserOpExpr> AddOp() { return AddNOp(2); }

Maybe<one::UserOpExpr> ZeroLikeOp() {
  return one::OpBuilder("zero_like").Input("in").Output("out").Build();
}

Maybe<one::UserOpExpr> OnesLikeOp() {
  AttrValue conf;
  conf.set_at_float(1.0);
  return one::OpBuilder("constant_like").Input("like").Output("out").Attr("value", conf).Build();
}

}  // namespace op_expr_helper

}  // namespace oneflow
