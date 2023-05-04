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
#include "oneflow/opencl/common/cl_kernel_pool.h"

#include "oneflow/opencl/common/cl_context.h"
#include "oneflow/opencl/common/cl_util.h"

namespace oneflow {

extern std::map<std::string, std::string> cl_program_table;

namespace {

cl_int clBuildProgram(clContext* context, const std::string& program_name, cl::Program* program,
                      const std::string& build_options) {
  const auto& it = cl_program_table.find(program_name);
  if (it == cl_program_table.end()) { return CL_INVALID_PROGRAM; }
  *program = cl::Program(context->context, it->second);
  cl_int ret = program->build({context->device}, build_options.c_str());
  if (ret != CL_SUCCESS) {
    if (program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context->device) == CL_BUILD_ERROR) {
      LOG(INFO) << program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(context->device);
    }
  }
  return ret;
}

}  // namespace

cl_int clKernelPool::buildKernel(clContext* context, const std::string& program_name,
                                 const std::string& kernel_name, cl::Kernel* kernel,
                                 const std::string& build_options) {
  std::unique_lock<std::mutex> lock(mutex_);
  std::string options = build_options + " -cl-fast-relaxed-math -cl-mad-enable";
  std::tuple<std::string, std::string> program_tag{program_name, options};
  auto it = programs_.find(program_tag);
  if (it == programs_.end()) {
    cl::Program program;
    CL_CHECK_OR_RETURN(clBuildProgram(context, program_name, &program, options));
    it = programs_.emplace(program_tag, program).first;
  }
  cl_int ret = CL_SUCCESS;
  *kernel = cl::Kernel(it->second, kernel_name.c_str(), &ret);
  return ret;
}

}  // namespace oneflow
