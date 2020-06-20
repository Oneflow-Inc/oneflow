#ifndef ONEFLOW_CORE_JOB_FOREIGN_CALLBACK_H_
#define ONEFLOW_CORE_JOB_FOREIGN_CALLBACK_H_

namespace oneflow {

class ForeignCallback {
 public:
  ForeignCallback() = default;
  virtual ~ForeignCallback() = default;

  virtual void EagerMirroredCast(const std::string& op_attribute_str,
                                 const std::string& parallel_conf_str) const {
    UNIMPLEMENTED();
  }
  virtual void EagerInterpretCompletedOp(const std::string& op_attribute_str,
                                         const std::string& parallel_conf_str) const {
    UNIMPLEMENTED();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_FOREIGN_CALLBACK_H_
