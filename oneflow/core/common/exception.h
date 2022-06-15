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
#ifndef ONEFLOW_CORE_COMMON_EXCEPTION_H_
#define ONEFLOW_CORE_COMMON_EXCEPTION_H_

#include <exception>
#include <string>

namespace oneflow {

class Exception : public std::exception {
 public:
  explicit Exception(const std::string& what) : what_(what) {}
  virtual ~Exception() = default;

  const char* what() const noexcept override { return what_.c_str(); }

 private:
  std::string what_;
};

class RuntimeException : public Exception {
 public:
  using Exception::Exception;
};

class TypeException : public Exception {
 public:
  using Exception::Exception;
};

class IndexException : public Exception {
 public:
  using Exception::Exception;
};

class NotImplementedException : public Exception {
 public:
  using Exception::Exception;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_EXCEPTION_H_
