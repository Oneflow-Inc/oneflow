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
#ifndef ONEFLOW_CORE_FRAMEWORK_INTERPRETER_H_
#define ONEFLOW_CORE_FRAMEWORK_INTERPRETER_H_

#include <functional>
#include "oneflow/core/vm/id_generator.h"

namespace oneflow {

class InstructionsBuilder;

class Interpreter {
 public:
  Interpreter(const Interpreter&) = delete;
  Interpreter(Interpreter&&) = delete;
  ~Interpreter() = default;

  virtual Maybe<void> Run(const std::function<Maybe<void>(InstructionsBuilder*)>& Build) = 0;

 protected:
  explicit Interpreter(const std::shared_ptr<vm::IdGenerator>& id_generator)
      : id_generator_(id_generator) {}

  const std::shared_ptr<vm::IdGenerator>& mut_id_generator() { return id_generator_; }

 private:
  std::shared_ptr<vm::IdGenerator> id_generator_;
};

class LogicalInterpreter : public Interpreter {
 public:
  LogicalInterpreter(const LogicalInterpreter&) = delete;
  LogicalInterpreter(LogicalInterpreter&&) = delete;
  LogicalInterpreter();
  ~LogicalInterpreter() = default;

  Maybe<void> Run(const std::function<Maybe<void>(InstructionsBuilder*)>& Build) override;
};

class PhysicalInterpreter : public Interpreter {
 public:
  PhysicalInterpreter(const PhysicalInterpreter&) = delete;
  PhysicalInterpreter(PhysicalInterpreter&&) = delete;
  PhysicalInterpreter();
  ~PhysicalInterpreter() = default;

  Maybe<void> Run(const std::function<Maybe<void>(InstructionsBuilder*)>& Build) override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_INTERPRETER_H_
