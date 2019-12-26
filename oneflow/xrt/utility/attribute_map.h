#ifndef ONEFLOW_XRT_UTILITY_ATTRIBUTE_MAP_H_
#define ONEFLOW_XRT_UTILITY_ATTRIBUTE_MAP_H_

#include "glog/logging.h"

#include "oneflow/xrt/any.h"
#include "oneflow/xrt/utility/stl.h"

namespace oneflow {
namespace xrt {
namespace util {

class AttributeMap {
 public:
  AttributeMap() = default;

  AttributeMap(const util::Map<std::string, Any> &attributes) : attributes_(attributes) {}

  virtual ~AttributeMap() = default;

  bool HasAttr(const std::string &name) const { return attributes_.count(name) > 0; }

  template<typename T>
  const T &Attr(const std::string &name) const {
    CHECK_GT(attributes_.count(name), 0);
    return any_cast<T>(attributes_.at(name));
  }

  template<typename T>
  T &Attr(const std::string &name) {
    CHECK_GT(attributes_.count(name), 0);
    return *any_cast<T *>(&attributes_[name]);
  }

  template<typename T>
  void SetAttr(const std::string &name, const T &valule) {
    attributes_[name] = valule;
  }

 protected:
  util::Map<std::string, Any> attributes_;
};

}  // namespace util
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_UTILITY_ATTRIBUTE_MAP_H_
