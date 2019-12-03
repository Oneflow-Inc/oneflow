#ifndef ONEFLOW_CORE_FRAMEWORK_SBP_CONTEXT_H_
#define ONEFLOW_CORE_FRAMEWORK_SBP_CONTEXT_H_

namespace oneflow {

namespace user_op {

using BlobDesc4IbnFn = std::function<Maybe<const BlobDesc*>(const std::string&)>;

class SbpContext final {
 public:
  SbpContext();
  ~SbpContext() = default;
  SbpContext(const SbpContext&) = delete;
  SbpContext(SbpContext&&) = delete;

  Maybe<const BlobDesc*> LogicalBlobDesc4Ibn(const std::string&) const;
  const ArgVec& inputs() const;
  const ArgVec& outputs() const;

 private:
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_SBP_CONTEXT_H_
