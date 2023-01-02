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
#ifndef ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_PYAST_AST_H_
#define ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_PYAST_AST_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>

namespace pyast {

using namespace std;
typedef string identifier;

class arg {
  identifier id;

 public:
  explicit arg(const identifier& arg) : id(arg) {}

  identifier get_arg() { return id; }

  static shared_ptr<arg> arg_(const identifier& arg_) { return make_shared<arg>(arg_); }
};

class arguments {
  vector<shared_ptr<arg>> args;

 public:
  explicit arguments(vector<shared_ptr<arg>> args) : args(std::move(args)) {}

  vector<shared_ptr<arg>> get_args() { return args; }

  static shared_ptr<arguments> arguments_(vector<shared_ptr<arg>> args) {
    return make_shared<arguments>(args);
  }
};

class stmt {
 public:
  enum StmtKind {
    kFunctionDef,
    kReturn,
    kAssign,
    kIf,
    kRaise,
    kAssert,
    kExpr,
  };

  explicit stmt(StmtKind kind) : kind(kind) {}
  virtual ~stmt() = default;

  StmtKind get_kind() const { return kind; }

 private:
  StmtKind kind;
};

class expr {
 public:
  enum ExprKind {
    kBoolOp,
    kBinOp,
    kLambda,
    kCompare,
    kCall,
    kNum,
    kConstant,
    kAttribute,
    kName,
  };

  explicit expr(ExprKind kind) : kind(kind) {}
  virtual ~expr() = default;

  ExprKind get_kind() const { return kind; }

 private:
  ExprKind kind;
};

class FunctionDef : public stmt {
  identifier name;
  shared_ptr<arguments> args;
  vector<shared_ptr<stmt>> body;

 public:
  FunctionDef(identifier name, shared_ptr<arguments> args, vector<shared_ptr<stmt>> body)
      : stmt(kFunctionDef), name(std::move(name)), args(std::move(args)), body(std::move(body)) {}

  static shared_ptr<FunctionDef> FunctionDef_(identifier name, shared_ptr<arguments> args,
                                              vector<shared_ptr<stmt>> body) {
    return make_shared<FunctionDef>(name, args, body);
  }

  identifier get_name() { return name; }
  shared_ptr<arguments> get_args() { return args; }
  vector<shared_ptr<stmt>> get_body() { return body; }

  static bool classof(const stmt* c) { return c->get_kind() == kFunctionDef; }
};

class Return : public stmt {
  shared_ptr<expr> value;

 public:
  explicit Return(shared_ptr<expr> value) : stmt(kReturn), value(std::move(value)) {}

  static shared_ptr<Return> Return_(shared_ptr<expr> value) { return make_shared<Return>(value); }

  shared_ptr<expr> get_value() { return value; }

  static bool classof(const stmt* c) { return c->get_kind() == kReturn; }
};

class Assign : public stmt {
  vector<shared_ptr<expr>> targets;
  shared_ptr<expr> value;

 public:
  Assign(vector<shared_ptr<expr>> targets, shared_ptr<expr> value)
      : stmt(kAssign), targets(std::move(targets)), value(std::move(value)) {}

  static shared_ptr<Assign> Assign_(vector<shared_ptr<expr>> targets, shared_ptr<expr> value) {
    return make_shared<Assign>(targets, value);
  }

  shared_ptr<expr> get_value() { return value; }
  vector<shared_ptr<expr>> get_targets() { return targets; }

  static bool classof(const stmt* c) { return c->get_kind() == kAssign; }
};

class If : public stmt {
  shared_ptr<expr> test;
  vector<shared_ptr<stmt>> body;
  vector<shared_ptr<stmt>> orelse;

 public:
  If(shared_ptr<expr> test, vector<shared_ptr<stmt>> body, vector<shared_ptr<stmt>> orelse)
      : stmt(kIf), test(std::move(test)), body(std::move(body)), orelse(orelse) {}

  static shared_ptr<If> If_(shared_ptr<expr> test, vector<shared_ptr<stmt>> body,
                            vector<shared_ptr<stmt>> orelse) {
    return make_shared<If>(test, body, orelse);
  }

  shared_ptr<expr> get_test() { return test; }
  vector<shared_ptr<stmt>> get_body() { return body; }
  vector<shared_ptr<stmt>> get_orelse() { return orelse; }

  static bool classof(const stmt* c) { return c->get_kind() == kIf; }
};

class Raise : public stmt {
  shared_ptr<expr> exc;
  shared_ptr<expr> cause;

 public:
  Raise(shared_ptr<expr> exc, shared_ptr<expr> cause)
      : stmt(kRaise), exc(std::move(exc)), cause(std::move(cause)) {}

  static shared_ptr<Raise> Raise_(shared_ptr<expr> exc, shared_ptr<expr> cause) {
    return make_shared<Raise>(exc, cause);
  }

  shared_ptr<expr> get_exc() { return exc; }
  shared_ptr<expr> get_cause() { return cause; }

  static bool classof(const stmt* c) { return c->get_kind() == kRaise; }
};

class Assert : public stmt {
  shared_ptr<expr> test;
  shared_ptr<expr> msg;

 public:
  Assert(shared_ptr<expr> test, shared_ptr<expr> msg)
      : stmt(kAssert), test(std::move(test)), msg(std::move(msg)) {}

  static shared_ptr<Assert> Assert_(shared_ptr<expr> test, shared_ptr<expr> msg) {
    return make_shared<Assert>(test, msg);
  }
  shared_ptr<expr> get_test() { return test; }
  shared_ptr<expr> get_msg() { return msg; }

  static bool classof(const stmt* c) { return c->get_kind() == kAssert; }
};

class Expr : public stmt {
  shared_ptr<expr> value;

 public:
  explicit Expr(shared_ptr<expr> value) : stmt(kExpr), value(std::move(value)) {}

  static shared_ptr<Expr> Expr_(shared_ptr<expr> value) { return make_shared<Expr>(value); }

  shared_ptr<expr> get_value() { return value; }

  static bool classof(const stmt* c) { return c->get_kind() == kExpr; }
};

class BoolOp : public expr {
 public:
  enum boolop_t {
    kAnd = 1,
    kOr,
  };
  BoolOp(boolop_t op, vector<shared_ptr<expr>> values)
      : expr(kBoolOp), op(op), values(std::move(values)) {}

  static shared_ptr<BoolOp> BoolOp_(boolop_t op, vector<shared_ptr<expr>> values) {
    return make_shared<BoolOp>(op, values);
  }

  boolop_t get_op() { return op; }
  vector<shared_ptr<expr>> get_values() { return values; }

  static bool classof(const expr* c) { return c->get_kind() == kBoolOp; }

 private:
  boolop_t op;
  vector<shared_ptr<expr>> values;
};

class BinOp : public expr {
 public:
  enum operator_t {
    kAdd = 1,
    kSub,
    kMult,
    kDiv,
    kPow,
  };

  BinOp(shared_ptr<expr> left, operator_t op, shared_ptr<expr> right)
      : expr(kBinOp), left(std::move(left)), right(std::move(right)), op(std::move(op)) {}

  BinOp(shared_ptr<expr> left, int op, shared_ptr<expr> right)
      : expr(kBinOp), left(std::move(left)), right(std::move(right)), op(int2op(op)) {}

  static shared_ptr<BinOp> BinOp_(shared_ptr<expr> left, int op, shared_ptr<expr> right) {
    return make_shared<BinOp>(left, op, right);
  }

  static operator_t int2op(int op) { return operator_t(op); }

  operator_t get_op() { return op; }
  shared_ptr<expr> get_left() { return left; }
  shared_ptr<expr> get_right() { return right; }

  static bool classof(const expr* c) { return c->get_kind() == kBinOp; }

 private:
  shared_ptr<expr> left;
  shared_ptr<expr> right;
  operator_t op;
};

class Lambda : public expr {
  shared_ptr<arguments> args;
  shared_ptr<expr> body;

 public:
  Lambda(shared_ptr<arguments> args, shared_ptr<expr> body)
      : expr(kLambda), args(std::move(args)), body(std::move(body)) {}

  static shared_ptr<Lambda> Lambda_(shared_ptr<arguments> args, shared_ptr<expr> body) {
    return make_shared<Lambda>(args, body);
  }

  shared_ptr<arguments> get_args() { return args; }
  shared_ptr<expr> get_body() { return body; }

  static bool classof(const expr* c) { return c->get_kind() == kLambda; }
};

class Compare : public expr {
 public:
  enum cmpop_t {
    kEq = 1,
    kNotEq,
    kLt,
    kLtE,
    kGt,
    kGtE,
  };

  Compare(shared_ptr<expr> left, vector<cmpop_t> ops, vector<shared_ptr<expr>> comparators)
      : expr(kCompare),
        left(std::move(left)),
        ops(std::move(ops)),
        comparators(std::move(comparators)) {}

  Compare(shared_ptr<expr> left, const vector<int>& ops, vector<shared_ptr<expr>> comparators)
      : expr(kCompare),
        left(std::move(left)),
        ops(int2op(ops)),
        comparators(std::move(comparators)) {}

  static shared_ptr<Compare> Compare_(shared_ptr<expr> left, vector<int> ops,
                                      vector<shared_ptr<expr>> comparators) {
    return make_shared<Compare>(left, ops, comparators);
  }

  static vector<cmpop_t> int2op(const vector<int>& op) {
    vector<cmpop_t> res;
    for (auto i : op) res.emplace_back(cmpop_t(i));
    return res;
  }

  vector<cmpop_t> get_ops() { return ops; }
  shared_ptr<expr> get_left() { return left; }
  vector<shared_ptr<expr>> get_comparators() { return comparators; }

  static bool classof(const expr* c) { return c->get_kind() == kCompare; }

 private:
  shared_ptr<expr> left;
  vector<cmpop_t> ops;
  vector<shared_ptr<expr>> comparators;
};

class Call : public expr {
  shared_ptr<expr> func;
  vector<shared_ptr<expr>> args;

 public:
  Call(shared_ptr<expr> func, vector<shared_ptr<expr>> args)
      : expr(kCall), func(std::move(func)), args(std::move(args)) {}

  static shared_ptr<Call> Call_(shared_ptr<expr> func, vector<shared_ptr<expr>> args) {
    return make_shared<Call>(func, args);
  }

  shared_ptr<expr> get_func() { return func; }
  vector<shared_ptr<expr>> get_args() { return args; }

  static bool classof(const expr* c) { return c->get_kind() == kCall; }
};

class Num : public expr {
  double value;

 public:
  explicit Num(double value) : expr(kNum), value(value) {}

  static shared_ptr<Num> Num_(double value) { return make_shared<Num>(value); }

  double get_value() { return value; }
  static bool classof(const expr* c) { return c->get_kind() == kNum; }
};

class Constant : public expr {
  double value;

 public:
  explicit Constant(double value) : expr(kConstant), value(value) {}

  static shared_ptr<Constant> Constant_(double value) { return make_shared<Constant>(value); }

  double get_value() { return value; }
  static bool classof(const expr* c) { return c->get_kind() == kConstant; }
};

class Attribute : public expr {
  shared_ptr<expr> value;
  identifier attr;

 public:
  Attribute(shared_ptr<expr> value, const identifier& attr)
      : expr(kAttribute), value(std::move(value)), attr(attr) {}

  static shared_ptr<Attribute> Attribute_(shared_ptr<expr> value, const identifier& attr) {
    return make_shared<Attribute>(value, attr);
  }

  shared_ptr<expr> get_value() { return value; }
  identifier get_attr() { return attr; }

  static bool classof(const expr* c) { return c->get_kind() == kAttribute; }
};

class Name : public expr {
  identifier id;

 public:
  explicit Name(const identifier& id) : expr(kName), id(id) {}

  static shared_ptr<Name> Name_(const identifier& id) { return make_shared<Name>(id); }

  identifier get_id() { return id; }
  static bool classof(const expr* c) { return c->get_kind() == kName; }
};

}  // namespace pyast

#endif  // ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_PYAST_AST_H_
