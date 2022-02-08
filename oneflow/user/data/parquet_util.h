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
#ifndef ONEFLOW_USER_DATA_PARQUET_UTIL_H_
#define ONEFLOW_USER_DATA_PARQUET_UTIL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/maybe.h"

#include "nlohmann/json.hpp"

namespace oneflow {

namespace data {

struct ParquetColumnDesc {
  ParquetColumnDesc()
      : col_id(-1),
        col_name(""),
        shape(Shape()),
        dtype(DataType::kInvalidDataType),
        is_variadic(false) {}

  int col_id;
  std::string col_name;
  Shape shape;
  DataType dtype;
  bool is_variadic;
};

struct ParquetColumnSchema {
  std::vector<ParquetColumnDesc> col_descs;
};

// JSON format
// [
//   {
//     shape: [...],
//     dtype: ...,
//   },
//   {
//     shape: [],
//     dtype: ...,
//     is_variadic: true,
//   },
//   ...
// ]
inline Maybe<void> ParseParquetColumnSchemaFromJson(ParquetColumnSchema* schema,
                                                    const std::string& json_str) {
  nlohmann::json json;
  std::istringstream json_ss(json_str);
  json_ss >> json;
  CHECK_OR_RETURN(json.is_array());
  for (auto& col : json) {
    schema->col_descs.emplace_back();
    auto& col_desc = schema->col_descs.back();
    if (col.contains("col_id")) {
      CHECK_OR_RETURN(col["col_id"].is_number_integer());
      col_desc.col_id = col["col_id"].get<int>();
    }

    if (col.contains("col_name")) {
      CHECK_OR_RETURN(col["col_name"].is_string());
      col_desc.col_name = col["col_name"].get<std::string>();
    }

    if (col.contains("is_variadic") && col["is_variadic"].get<bool>()) {
      col_desc.is_variadic = true;
    } else {
      col_desc.is_variadic = false;
      CHECK_OR_RETURN(col.contains("shape"));
      CHECK_OR_RETURN(col.contains("dtype"));
    }

    if (col.contains("shape")) {
      CHECK_OR_RETURN(col["shape"].is_array());
      col_desc.shape = Shape({col["shape"].begin(), col["shape"].end()});
    }

    if (col.contains("dtype")) {
      CHECK_OR_RETURN(col["dtype"].is_number_integer());
      col_desc.dtype = static_cast<DataType>(col["dtype"].get<int>());
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace data

}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_PARQUET_UTIL_H_
