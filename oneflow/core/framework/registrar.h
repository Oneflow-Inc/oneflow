#ifndef ONEFLOW_CORE_FRAMEWORK_REGISTAR_H_
#define ONEFLOW_CORE_FRAMEWORK_REGISTAR_H_

namespace oneflow {

namespace user_op {

template<typename BuilderT>
struct Registrar final {
  Registrar(const BuilderT& builder) { builder.Build().InsertToGlobalRegistry(); }
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_REGISTAR_H_
