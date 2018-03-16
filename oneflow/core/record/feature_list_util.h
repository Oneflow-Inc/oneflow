#ifndef ONEFLOW_CORE_RECORD_FEATURE_LIST_UTIL_H_
#define ONEFLOW_CORE_RECORD_FEATURE_LIST_UTIL_H_

#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

class FeatureListHandler {
 public:
  FeatureListHandler() = default;
  ~FeatureListHandler() = default;

  virtual int64_t SizeOf(const Feature& feature) const = 0;
  virtual DataType data_type() const = 0;
  virtual const void* DptrOf(const Feature& feature) const = 0;
};

class BytesListHandler final : public FeatureListHandler {
 public:
  BytesListHandler() = default;
  ~BytesListHandler() = default;

  int64_t SizeOf(const Feature& feature) const override;
  DataType data_type() const override { return DataType::kInt8; }
  const void* DptrOf(const Feature& feature) const override;
};

class FloatListHandler final : public FeatureListHandler {
 public:
  FloatListHandler() = default;
  ~FloatListHandler() = default;

  int64_t SizeOf(const Feature& feature) const override;
  DataType data_type() const override { return DataType::kFloat; }
  const void* DptrOf(const Feature& feature) const override;
};

class DoubleListHandler final : public FeatureListHandler {
 public:
  DoubleListHandler() = default;
  ~DoubleListHandler() = default;

  int64_t SizeOf(const Feature& feature) const override;
  DataType data_type() const override { return DataType::kDouble; }
  const void* DptrOf(const Feature& feature) const override;
};

class Int32ListHandler final : public FeatureListHandler {
 public:
  Int32ListHandler() = default;
  ~Int32ListHandler() = default;

  int64_t SizeOf(const Feature& feature) const override;
  DataType data_type() const override { return DataType::kInt32; }
  const void* DptrOf(const Feature& feature) const override;
};

static FeatureListHandler* GetFeatureListHandler(const Feature& feature) {
  static BytesListHandler bytes_list_handler;
  static FloatListHandler float_list_handler;
  static DoubleListHandler double_list_handler;
  static Int32ListHandler int32_list_handler;

  if (feature.has_bytes_list()) {
    return &bytes_list_handler;
  } else if (feature.has_float_list()) {
    return &float_list_handler;
  } else if (feature.has_double_list()) {
    return &double_list_handler;
  } else if (feature.has_int32_list()) {
    return &int32_list_handler;
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_FEATURE_LIST_UTIL_H_
