#ifndef ONEFLOW_CORE_FRAMEWORK_SBP_CONTEXT_H_
#define ONEFLOW_CORE_FRAMEWORK_SBP_CONTEXT_H_

namespace oneflow {

namespace user_op {

class SbpContext final {
 public:
  SbpContext();
  ~SbpContext() = default;
  SbpContext(const SbpContext&) = delete;
  SbpContext(SbpContext&&) = delete;

  const BlobDef* BlobDef4ArgNameAndIndex(const std::string&, int32_t) const;
  const ArgVec& inputs() const;
  const ArgVec& outputs() const;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_SBP_CONTEXT_H_
