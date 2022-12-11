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
#include "oneflow/core/platform/include/wrapper.h"
#include <dlfcn.h>

#ifdef __linux__
#include <link.h>
#endif  // __linux__

namespace oneflow {
namespace platform {

namespace {

void* OpenSymbol(void* handle, const char* name) {
  void* ret = dlsym(handle, name);
  if (!ret) {
    std::cerr << "Error in dlopen or dlsym: " << dlerror() << "\n";
    abort();
  }
  return ret;
}

}  // namespace

// original implementation is from pytorch:
// https://github.com/pytorch/pytorch/blob/259d19a7335b32c4a27a018034551ca6ae997f6b/aten/src/ATen/DynamicLibrary.cpp

std::unique_ptr<DynamicLibrary> DynamicLibrary::Load(const std::vector<std::string>& names) {
  for (const std::string& name : names) {
    void* handle = dlopen(name.c_str(), RTLD_LOCAL | RTLD_NOW);
    if (handle != nullptr) {
      DynamicLibrary* lib = new DynamicLibrary(handle);
      return std::unique_ptr<DynamicLibrary>(lib);
    }
  }
  return std::unique_ptr<DynamicLibrary>();
}

void* DynamicLibrary::LoadSym(const char* name) { return OpenSymbol(handle_, name); }

#ifdef __linux__
std::string DynamicLibrary::AbsolutePath() {
  struct link_map* map;
  dlinfo(handle_, RTLD_DI_LINKMAP, &map);
  return map->l_name;
}
#endif  // __linux__

DynamicLibrary::~DynamicLibrary() { dlclose(handle_); }

}  // namespace platform
}  // namespace oneflow
