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
#include "oneflow/core/dl/include/wrapper.h"
#include <dlfcn.h>

namespace oneflow {
namespace dl {

static void* checkDL(void* x) {
  if (!x) { LOG(FATAL) << "Error in dlopen or dlsym: " << dlerror(); }
  return x;
}

DynamicLibrary::DynamicLibrary(const char* name, const char* alt_name) {
  handle_ = dlopen(name, RTLD_LOCAL | RTLD_NOW);
  if (!handle_) {
    if (alt_name) {
      handle_ = dlopen(alt_name, RTLD_LOCAL | RTLD_NOW);
      if (!handle_) { LOG(FATAL) << "Error in dlopen for library " << name << "and " << alt_name; }
    } else {
      LOG(FATAL) << "Error in dlopen: " << dlerror();
    }
  }
}

void* DynamicLibrary::sym(const char* name) {
  CHECK(handle_);
  return checkDL(dlsym(handle_, name));
}

DynamicLibrary::~DynamicLibrary() {
  if (!handle_) return;
  dlclose(handle_);
}

}  // namespace dl
}  // namespace oneflow
