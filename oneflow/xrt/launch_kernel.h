/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_XRT_XRT_LAUNCH_KERNEL_H_
#define ONEFLOW_XRT_XRT_LAUNCH_KERNEL_H_

#include <unordered_map>

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/xrt/compilation_cache.h"
#include "oneflow/xrt/executable.h"
#include "oneflow/xrt/graph_compiler.h"
#include "oneflow/xrt/parameter.h"
#include "oneflow/xrt/types.h"

namespace oneflow {

template<DeviceType device_type>
class BlobDescGetter {
 public:
  BlobDescGetter() = default;
  BlobDescGetter(const KernelIf<device_type>* kernel,
                 std::function<Blob*(const std::string&)> get_blob_fn)
      : kernel_(kernel), get_blob_fn_(get_blob_fn) {}

  void DumpEntryBlobDescTo(std::unordered_map<std::string, BlobDesc>* entry_blob_desc) const;

 private:
  const KernelIf<device_type>* kernel_;
  std::function<Blob*(const std::string&)> get_blob_fn_;
};

template<DeviceType device_type>
class XrtLaunchKernel : public KernelIf<device_type> {
 public:
  XrtLaunchKernel() = default;
  virtual ~XrtLaunchKernel() {}

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  xrt::Executable* BuildExecutable(const std::vector<xrt::Parameter>& entry_params,
                                   const std::vector<xrt::Parameter>& return_params,
                                   const std::vector<xrt::InputOutputAlias>& aliases,
                                   const int device_ordinal) const;

  void MakeInputOutputAlias(                            // NOLINT
      const std::vector<xrt::Parameter>& entry_params,  // NOLINT
      std::vector<xrt::Parameter>* return_params,
      std::vector<xrt::InputOutputAlias>* aliases) const;

  void MappingParamsToFunctionNames(std::vector<xrt::Parameter>* entry_params,
                                    std::vector<xrt::Parameter>* return_params) const;

  bool IsStateless() const override { return false; }

 private:
  mutable BlobDescGetter<device_type> desc_getter_;
  mutable std::shared_ptr<xrt::CompilationCache> compilation_cache_;
};

}  // namespace oneflow

#endif  // ONEFLOW_XRT_XRT_LAUNCH_KERNEL_H_
