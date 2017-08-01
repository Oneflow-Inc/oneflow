#ifndef ONEFLOW_CORE_SCHEDULE_UTILS_UTILS_H_
#define ONEFLOW_CORE_SCHEDULE_UTILS_UTILS_H_

namespace oneflow {
namespace schedule {

inline std::string GetClassName(const std::string& prettyFunction) {
  size_t colons = prettyFunction.rfind("::");
  if (colons == std::string::npos) return "::";
  size_t begin = prettyFunction.substr(0, colons).rfind("::") + 2;
  size_t end = colons - begin;

  return prettyFunction.substr(begin, end);
}

#ifdef _MSC_VER
#define __CLASS_NAME__ GetClassName(__FUNCSIG__)
#else
#define __CLASS_NAME__ GetClassName(__PRETTY_FUNCTION__)
#endif

#define DEFINE_PURE_VIRTUAL_TYPE() virtual const std::string type() const = 0

#define DEFINE_METHOD_TYPE() \
  virtual const std::string type() const { return __CLASS_NAME__; }

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_UTILS_UTILS_H_
