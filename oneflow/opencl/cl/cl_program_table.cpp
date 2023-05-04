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
#include <map>
#include <string>

namespace oneflow {

const char* cl_binary_source = " \
__kernel void cl_binary(int count, __global DT* in0, __global DT* in1, \
                        __global DT* out) { \
  int id = get_global_id(0); \
  for (; id < count; id += get_global_size(0)) { out[id] = in0[id] OP in1[id]; } \
} \
";

std::map<std::string, std::string> cl_program_table = {
    {"cl_binary", cl_binary_source},
};

}  // namespace oneflow
