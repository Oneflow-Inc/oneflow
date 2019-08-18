#ifndef TENSORFLOW_COMPILER_JIT_XLA_LIB_RUNTIME_WORKSPACE_BYTES_H_
#define TENSORFLOW_COMPILER_JIT_XLA_LIB_RUNTIME_WORKSPACE_BYTES_H_

#include <vector>

#include "tensorflow/compiler/xla/client/local_client.h"

namespace xla {

size_t CalcWorkspaceByteSize(LocalExecutable *local_executable);

Status ResultAllocationIndices(LocalExecutable *local_executable,
                               std::vector<int64_t> *indices);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_JIT_XLA_LIB_RUNTIME_WORKSPACE_BYTES_H_
