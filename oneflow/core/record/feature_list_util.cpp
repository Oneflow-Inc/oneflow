#include "oneflow/core/record/feature_list_util.h"

namespace oneflow {

int64_t BytesListHandler::SizeOf(const Feature& feature) const {
  return feature.bytes_list().value().size();
}

const void* BytesListHandler::DptrOf(const Feature& feature) const {
  return static_cast<const void*>(feature.bytes_list().value().data());
}

int64_t FloatListHandler::SizeOf(const Feature& feature) const {
  return feature.float_list().value().size();
}

const void* FloatListHandler::DptrOf(const Feature& feature) const {
  return static_cast<const void*>(feature.float_list().value().data());
}

int64_t DoubleListHandler::SizeOf(const Feature& feature) const {
  return feature.double_list().value().size();
}

const void* DoubleListHandler::DptrOf(const Feature& feature) const {
  return static_cast<const void*>(feature.double_list().value().data());
}

int64_t Int32ListHandler::SizeOf(const Feature& feature) const {
  return feature.int32_list().value().size();
}

const void* Int32ListHandler::DptrOf(const Feature& feature) const {
  return static_cast<const void*>(feature.int32_list().value().data());
}

}  // namespace oneflow
