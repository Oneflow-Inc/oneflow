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
#include "PyAst/AstMlirGen.h"

// declare any scope variables in the front of function block to ensure the enough lifetime.
mlir::LogicalResult BuilderWithSymbolTable::Declare(const std::string& var, mlir::Value value) {
  auto iter = symbolTable_.find(var);
  if (iter != symbolTable_.end()) {
    builder_.create<mlir::memref::StoreOp>(Loc(), value, iter->second);
    return mlir::failure();
  }

  auto history_block = builder_.getInsertionBlock();
  auto history_point = builder_.getInsertionPoint();

  builder_.setInsertionPointToStart(symbolTableForDeclareBlock_);

  auto single_type = mlir::Float32Type::getF32(builder_.getContext());
  auto type = mlir::MemRefType::get({}, single_type);
  auto key = builder_.create<mlir::memref::AllocOp>(Loc(), type);

  builder_.setInsertionPoint(history_block, history_point);
  builder_.create<mlir::memref::StoreOp>(Loc(), value, key);
  symbolTable_[var] = key;
  return mlir::success();
}

// look up memref of the special symbol with variable name
mlir::Value BuilderWithSymbolTable::LoopUp(const std::string& var) {
  if (symbolTable_.count(var) == 1) { return symbolTable_[var]; }
  theModule_->emitError("error: unknown variable '" + var + "'");
  return nullptr;
}

// generate a location of mlir for ops
mlir::Location BuilderWithSymbolTable::Loc(const std::string& file_name, int line, int col) {
  return mlir::FileLineColLoc::get(builder_.getStringAttr(file_name), line, col);
}

// dump the current whole module up
void BuilderWithSymbolTable::Dump() { theModule_.dump(); }

// generate a module op for lr jit registry from a ast
mlir::ModuleOp MLIRGenImpl::GenModule(pyast::FunctionDef* func) {
  theModule_ = mlir::ModuleOp::create(Loc());

  if (failed(verify(theModule_))) {
    theModule_.emitError("module verification error");
    return nullptr;
  }

  builder_.setInsertionPointToEnd(theModule_.getBody());

  auto args = func->get_args()->get_args();
  auto type = mlir::Float32Type::getF32(builder_.getContext());
  llvm::SmallVector<mlir::Type> arg_types(args.size(), type);
  llvm::SmallVector<mlir::Type> res_types(1, type);

  auto func_type = builder_.getFunctionType(arg_types, res_types);
  auto function = mlir::func::FuncOp::create(Loc(), func->get_name(), func_type);

  auto* entry_block = function.addEntryBlock();
  symbolTableForDeclareBlock_ = entry_block;
  theModule_.push_back(function);
  builder_.setInsertionPointToStart(entry_block);

  for (const auto nameValue : llvm::zip(args, entry_block->getArguments())) {
    if (failed(Declare(std::get<0>(nameValue)->get_arg(), std::get<1>(nameValue)))) {
      return nullptr;
    }
  }

  builder_.setInsertionPointToStart(entry_block);
  for (auto& stmt : func->get_body()) { MlirGen(stmt.get()); }

  return theModule_;
}

// use llvm rtti to dispatch respective code gen tasks of stmt
void MLIRGenImpl::MlirGen(pyast::stmt* stmt) {
  llvm::TypeSwitch<pyast::stmt*>(stmt)
      .Case<pyast::Return, pyast::Assign, pyast::If>([&](auto* node) { MlirGen(node); })
      .Default([&](auto* node) { theModule_->emitError("StmtKind not support yet"); });
}

// use llvm rtti to dispatch respective code gen tasks of expr
mlir::Value MLIRGenImpl::MlirGen(pyast::expr* expr) {
  mlir::Value res;
  llvm::TypeSwitch<pyast::expr*>(expr)
      .Case<pyast::BinOp, pyast::Compare, pyast::Call, pyast::Constant, pyast::Name>(
          [&](auto* node) { res = MlirGen(node); })
      .Default([&](auto* node) { theModule_->emitError("ExprKind not support yet"); });
  return res;
}

void MLIRGenImpl::MlirGen(pyast::If* expr) {
  auto test = MlirGen(expr->get_test().get());

  if (test.getType().isF32()) {
    auto eq = mlir::arith::CmpFPredicate::ONE;
    auto zero_attr = builder_.getF32FloatAttr(0);
    auto zero = builder_.create<mlir::arith::ConstantOp>(Loc(), zero_attr);
    test = builder_.create<mlir::arith::CmpFOp>(Loc(), eq, test, zero);
  }

  mlir::Block* then_block = builder_.createBlock(builder_.getBlock()->getParent());
  mlir::Block* else_block = builder_.createBlock(builder_.getBlock()->getParent());
  mlir::Block* after_block = builder_.createBlock(builder_.getBlock()->getParent());
  builder_.setInsertionPointAfterValue(test);
  builder_.create<mlir::cf::CondBranchOp>(Loc(), test, then_block, llvm::None, else_block,
                                          llvm::None);

  builder_.setInsertionPointToStart(then_block);
  for (auto& expr : expr->get_body()) { MlirGen(expr.get()); }
  if (then_block->empty() || !llvm::dyn_cast<mlir::func::ReturnOp>(then_block->back())) {
    builder_.create<mlir::cf::BranchOp>(Loc(), after_block);
  }

  builder_.setInsertionPointToStart(else_block);
  for (auto& expr : expr->get_orelse()) { MlirGen(expr.get()); }
  if (else_block->empty() || !llvm::dyn_cast<mlir::func::ReturnOp>(else_block->back())) {
    builder_.create<mlir::cf::BranchOp>(Loc(), after_block);
  }

  builder_.setInsertionPointToStart(after_block);
}

mlir::Value MLIRGenImpl::MlirGen(pyast::Compare* expr) {
  if (expr->get_comparators().size() != 1 || expr->get_ops().size() != 1) {
    theModule_->emitError("compare only support once compare now");
  }

  mlir::arith::CmpFPredicate op = mlir::arith::CmpFPredicate::OEQ;
  switch (expr->get_ops()[0]) {
    case pyast::Compare::kEq: op = mlir::arith::CmpFPredicate::OEQ; break;
    case pyast::Compare::kNotEq: op = mlir::arith::CmpFPredicate::ONE; break;
    case pyast::Compare::kLt: op = mlir::arith::CmpFPredicate::OLT; break;
    case pyast::Compare::kLtE: op = mlir::arith::CmpFPredicate::OLE; break;
    case pyast::Compare::kGt: op = mlir::arith::CmpFPredicate::OGT; break;
    case pyast::Compare::kGtE: op = mlir::arith::CmpFPredicate::OGE; break;
    default: theModule_->emitError("compare_ not support op now");
  }

  auto lhs = MlirGen(expr->get_left().get());
  auto rhs = MlirGen(expr->get_comparators()[0].get());
  auto res = builder_.create<mlir::arith::CmpFOp>(Loc(), op, lhs, rhs);
  return res;
}

mlir::Value MLIRGenImpl::MlirGen(pyast::BinOp* expr) {
  auto lhs = MlirGen(expr->get_left().get());
  auto rhs = MlirGen(expr->get_right().get());
  mlir::Value res;

  switch (expr->get_op()) {
    case pyast::BinOp::kAdd: res = builder_.create<mlir::arith::AddFOp>(Loc(), lhs, rhs); break;
    case pyast::BinOp::kSub: res = builder_.create<mlir::arith::SubFOp>(Loc(), lhs, rhs); break;
    case pyast::BinOp::kDiv: res = builder_.create<mlir::arith::DivFOp>(Loc(), lhs, rhs); break;
    case pyast::BinOp::kMult: res = builder_.create<mlir::arith::MulFOp>(Loc(), lhs, rhs); break;
    case pyast::BinOp::kPow: res = builder_.create<mlir::math::PowFOp>(Loc(), lhs, rhs); break;
    default: break;
  }

  return res;
}

mlir::Value MLIRGenImpl::MlirGen(pyast::Call* expr) {
  mlir::Value res;
  if (expr->get_func()->get_kind() == pyast::expr::kAttribute) {
    auto func_ = expr->get_func().get();
    auto func = *dynamic_cast<pyast::Attribute*>(func_);
    auto func_value = func.get_value();

    if (func_value->get_kind() != pyast::expr::kName
        || dynamic_cast<pyast::Name*>(func_value.get())->get_id() != "math") {
      theModule_->emitError("only support call func is python math lib");
    }
    if (expr->get_args().size() != 1) {
      theModule_->emitError("attribute node only support call func with one param");
    }

    auto value = MlirGen(expr->get_args()[0].get());
    auto attr = func.get_attr();

    if (attr == "floor") {
      res = builder_.create<mlir::math::FloorOp>(Loc(), value);
    } else if (attr == "cos") {
      res = builder_.create<mlir::math::CosOp>(Loc(), value);
    } else if (attr == "ceil") {
      res = builder_.create<mlir::math::CeilOp>(Loc(), value);
    } else {
      theModule_->emitError(attr + " not support yet");
    }
  } else if (expr->get_func()->get_kind() == pyast::expr::kName) {
    auto func_ = expr->get_func().get();
    auto func = *dynamic_cast<pyast::Name*>(func_);

    if (expr->get_args().size() != 2) {
      theModule_->emitError("name node only support call func with two param");
    }

    auto left = MlirGen(expr->get_args()[0].get());
    auto right = MlirGen(expr->get_args()[1].get());

    auto attr = func.get_id();

    if (attr == "max") {
      res = builder_.create<mlir::arith::MaxFOp>(Loc(), left, right);
    } else if (attr == "min") {
      res = builder_.create<mlir::arith::MinFOp>(Loc(), left, right);
    } else {
      theModule_->emitError(attr + " not support yet");
    }

  } else {
    theModule_->emitError("only support call func is attribute and name node");
  }

  return res;
}

mlir::Value MLIRGenImpl::MlirGen(pyast::Constant* expr) {
  float value = expr->get_value();
  auto constant = builder_.create<mlir::arith::ConstantOp>(Loc(), builder_.getF32FloatAttr(value));
  return constant;
}

mlir::Value MLIRGenImpl::MlirGen(pyast::Name* expr) {
  auto key = LoopUp(expr->get_id());
  builder_.setInsertionPointToEnd(builder_.getInsertionBlock());
  auto value = builder_.create<mlir::memref::LoadOp>(Loc(), key);
  return value;
}

void MLIRGenImpl::MlirGen(pyast::Assign* stmt) {
  auto value = MlirGen(stmt->get_value().get());

  for (auto& target : stmt->get_targets()) {
    if (target->get_kind() != pyast::expr::kName) {
      theModule_->emitError("only support assign to name node");
    }
    auto name = dynamic_cast<pyast::Name*>(target.get())->get_id();

    Declare(name, value);
  }
}

void MLIRGenImpl::MlirGen(pyast::Return* stmt) {
  auto value = MlirGen(stmt->get_value().get());

  builder_.create<mlir::func::ReturnOp>(Loc(), mlir::ValueRange({value}));
}
