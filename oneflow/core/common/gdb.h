#ifndef ONEFLOW_CORE_COMMON_GDB_H_
#define ONEFLOW_CORE_COMMON_GDB_H_

namespace oneflow {

namespace gdb {

void ForwardEnterBreakPoint(const OpAttribute& op_attribute,
                            const std::function<Blob*(const std::string&)>& BnInOp2Blob);

void ForwardLeaveBreakPoint(const OpAttribute& op_attribute,
                            const std::function<Blob*(const std::string&)>& BnInOp2Blob);

void BackwardEnterBreakPoint(const OpAttribute& op_attribute,
                             const std::function<Blob*(const std::string&)>& BnInOp2Blob);

void BackwardLeaveBreakPoint(const OpAttribute& op_attribute,
                             const std::function<Blob*(const std::string&)>& BnInOp2Blob);

}  // namespace gdb

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_GDB_H_
