#pragma once

#include <utility>
#include <vector>
#include <string>
#include <memory>

namespace pyast {
using namespace std;

typedef string identifier;

class stmt_;
typedef shared_ptr<stmt_> stmt_ptr;

class FunctionDef_;
typedef shared_ptr<FunctionDef_> FunctionDef_ptr;

class Return_;
typedef shared_ptr<Return_> Return_ptr;

class Assign_;
typedef shared_ptr<Assign_> Assign_ptr;

class If_;
typedef shared_ptr<If_> If_ptr;

class Raise_;
typedef shared_ptr<Raise_> Raise_ptr;

class Assert_;
typedef shared_ptr<Assert_> Assert_ptr;

class Expr_;
typedef shared_ptr<Expr_> Expr_ptr;

class expr_;
typedef shared_ptr<expr_> expr_ptr;

class BoolOp_;
typedef shared_ptr<BoolOp_> BoolOp_ptr;

class BinOp_;
typedef shared_ptr<BinOp_> BinOp_ptr;

class Lambda_;
typedef shared_ptr<Lambda_> Lambda_ptr;

class Compare_;
typedef shared_ptr<Compare_> Compare_ptr;

class Call_;
typedef shared_ptr<Call_> Call_ptr;

class Num_;
typedef shared_ptr<Num_> Num_ptr;

class Constant_;
typedef shared_ptr<Constant_> Constant_ptr;

class Attribute_;
typedef shared_ptr<Attribute_> Attribute_ptr;

class Name_;
typedef shared_ptr<Name_> Name_ptr;

class arguments_;
typedef shared_ptr<arguments_> arguments_ptr;

class arg_;
typedef shared_ptr<arg_> arg_ptr;

class arg_ {
  identifier arg;

 public:
  explicit arg_(const identifier& arg_) : arg(arg_) {}
  identifier get_arg() { return arg; }
};

class arguments_ {
  vector<arg_ptr> args;

 public:
  explicit arguments_(vector<arg_ptr> args_) : args(move(args_)) {}
  vector<arg_ptr> get_args() { return args; }
};

class stmt_ {
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
  explicit stmt_(StmtKind kind) : kind(kind) {}
  StmtKind get_kind() const { return kind; }
  virtual ~stmt_() = default;

 private:
  StmtKind kind;
};

class expr_ {
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
  explicit expr_(ExprKind kind) : kind(kind) {}
  ExprKind get_kind() const { return kind; }
  virtual ~expr_() = default;

 private:
  ExprKind kind;
};

class FunctionDef_ : public stmt_ {
  identifier name;
  arguments_ptr args;
  vector<stmt_ptr> body;

 public:
  FunctionDef_(identifier name_, arguments_ptr args_, vector<stmt_ptr> body_)
      : stmt_(kFunctionDef), name(move(name_)), args(move(args_)), body(move(body_)) {}

  identifier get_name() { return name; }
  arguments_ptr get_args() { return args; }
  vector<stmt_ptr> get_body() { return body; }

  static bool classof(const stmt_* c) { return c->get_kind() == kFunctionDef; }
};

class Return_ : public stmt_ {
  expr_ptr value;

 public:
  explicit Return_(expr_ptr value_) : stmt_(kReturn), value(value_) {}
  expr_ptr get_value() { return value; }

  static bool classof(const stmt_* c) { return c->get_kind() == kReturn; }
};

class Assign_ : public stmt_ {
  vector<expr_ptr> targets;
  expr_ptr value;

 public:
  Assign_(vector<expr_ptr> targets_, expr_ptr value_)
      : stmt_(kAssign), targets(targets_), value(value_) {}

  expr_ptr get_value() { return value; }
  vector<expr_ptr> get_targets() { return targets; }

  static bool classof(const stmt_* c) { return c->get_kind() == kAssign; }
};

class If_ : public stmt_ {
  expr_ptr test;
  vector<stmt_ptr> body;
  vector<stmt_ptr> orelse;

 public:
  If_(expr_ptr test_, const vector<stmt_ptr>& body_, const vector<stmt_ptr>& orelse_)
      : stmt_(kIf), test(test_), body(body_), orelse(orelse_) {}

  expr_ptr get_test() { return test; }
  vector<stmt_ptr> get_body() { return body; }
  vector<stmt_ptr> get_orelse() { return orelse; }
  static bool classof(const stmt_* c) { return c->get_kind() == kIf; }
};

class Raise_ : public stmt_ {
  expr_ptr exc;
  expr_ptr cause;

 public:
  Raise_(expr_ptr exc_, expr_ptr cause_) : stmt_(kRaise), exc(exc_), cause(cause_) {}

  expr_ptr get_exc() { return exc; }
  expr_ptr get_cause() { return cause; }
  static bool classof(const stmt_* c) { return c->get_kind() == kRaise; }
};

class Assert_ : public stmt_ {
  expr_ptr test;
  expr_ptr msg;

 public:
  Assert_(expr_ptr test_, expr_ptr msg_) : stmt_(kAssert), test(test_), msg(msg_) {}

  expr_ptr get_test() { return test; }
  expr_ptr get_msg() { return msg; }
  static bool classof(const stmt_* c) { return c->get_kind() == kAssert; }
};

class Expr_ : public stmt_ {
  expr_ptr value;

 public:
  explicit Expr_(expr_ptr value_) : stmt_(kExpr), value(value_) {}

  expr_ptr get_value() { return value; }
  static bool classof(const stmt_* c) { return c->get_kind() == kExpr; }
};

class BoolOp_ : public expr_ {
 public:
  enum boolop_t {
    kAnd,
    kOr,
  };
  BoolOp_(boolop_t op_, const vector<expr_ptr>& values_)
      : expr_(kBoolOp), op(op_), values(values_) {}

  boolop_t get_op() { return op; }
  vector<expr_ptr> get_values() { return values; }
  static bool classof(const expr_* c) { return c->get_kind() == kBoolOp; }

 private:
  boolop_t op;
  vector<expr_ptr> values;
};

class BinOp_ : public expr_ {
 public:
  enum operator_t {
    kAdd,
    kSub,
    kMult,
    kDiv,
    kPow,
  };

  BinOp_(expr_ptr left_, operator_t op_, expr_ptr right_)
      : expr_(kBinOp), left(move(left_)), op(op_), right(move(right_)) {}

  operator_t get_op() { return op; }
  expr_ptr get_left() { return left; }
  expr_ptr get_right() { return right; }
  static bool classof(const expr_* c) { return c->get_kind() == kBinOp; }

 private:
  expr_ptr left;
  operator_t op;
  expr_ptr right;
};

class Lambda_ : public expr_ {
  arguments_ptr args;
  expr_ptr body;

 public:
  Lambda_(arguments_ptr args_, expr_ptr body_) : expr_(kLambda), args(args_), body(body_) {}
  arguments_ptr get_args() { return args; }
  expr_ptr get_body() { return body; }
  static bool classof(const expr_* c) { return c->get_kind() == kLambda; }
};

class Compare_ : public expr_ {
 public:
  enum cmpop_t {
    kEq,
    kNotEq,
    kLt,
    kLtE,
    kGt,
    kGtE,
  };
  Compare_(expr_ptr left_, const vector<cmpop_t>& ops_, const vector<expr_ptr>& comparators_)
      : expr_(kCompare), left(move(left_)), ops(ops_), comparators(comparators_) {}

  vector<cmpop_t> get_ops() { return ops; }
  vector<expr_ptr> get_comparators() { return comparators; }
  static bool classof(const expr_* c) { return c->get_kind() == kCompare; }

 private:
  expr_ptr left;
  vector<cmpop_t> ops;
  vector<expr_ptr> comparators;
};

class Call_ : public expr_ {
  expr_ptr func;
  vector<expr_ptr> args;

 public:
  Call_(expr_ptr func_, const vector<expr_ptr>& args_)
      : expr_(kCall), func(move(func_)), args(args_) {}

  expr_ptr get_func() { return func; }
  vector<expr_ptr> get_args() { return args; }
  static bool classof(const expr_* c) { return c->get_kind() == kCall; }
};

class Num_ : public expr_ {
  double value;

 public:
  explicit Num_(double value_) : expr_(kNum), value(value_) {}

  double get_value() { return value; }
  static bool classof(const expr_* c) { return c->get_kind() == kNum; }
};

class Constant_ : public expr_ {
  double value;

 public:
  explicit Constant_(double value_) : expr_(kConstant), value(value_) {}

  double get_value() { return value; }
  static bool classof(const expr_* c) { return c->get_kind() == kConstant; }
};

class Attribute_ : public expr_ {
  expr_ptr value;
  identifier attr;

 public:
  Attribute_(expr_ptr value_, const identifier& attr_)
      : expr_(kAttribute), value(move(value_)), attr(attr_) {}

  expr_ptr get_value() { return value; }
  identifier get_attr() { return attr; }
  static bool classof(const expr_* c) { return c->get_kind() == kAttribute; }
};

class Name_ : public expr_ {
  identifier id;

 public:
  explicit Name_(const identifier& id_) : expr_(kName), id(id_) {}

  identifier get_id() { return id; }
  static bool classof(const expr_* c) { return c->get_kind() == kName; }
};

stmt_ptr FunctionDef(const identifier& name, arguments_ptr args, const vector<stmt_ptr>& body) {
  return make_shared<FunctionDef_>(name, move(args), body);
}

stmt_ptr Return(expr_ptr value) { return make_shared<Return_>(value); }

stmt_ptr Assign(const vector<expr_ptr>& targets, expr_ptr value) {
  return make_shared<Assign_>(targets, value);
}

stmt_ptr If(expr_ptr test, const vector<stmt_ptr>& body, const vector<stmt_ptr>& orelse) {
  return make_shared<If_>(test, body, orelse);
}

stmt_ptr Raise(expr_ptr exc, expr_ptr cause) { return make_shared<Raise_>(exc, cause); }

stmt_ptr Assert(expr_ptr test, expr_ptr msg) { return make_shared<Assert_>(test, msg); }
stmt_ptr Expr(expr_ptr value) { return make_shared<Expr_>(value); }

expr_ptr BoolOp(BoolOp_::boolop_t op, const vector<expr_ptr>& values) {
  return make_shared<BoolOp_>(op, values);
}

expr_ptr BinOp(expr_ptr left, BinOp_::operator_t op, expr_ptr right) {
  return make_shared<BinOp_>(left, op, right);
}

expr_ptr Lambda(arguments_ptr args, expr_ptr body) { return make_shared<Lambda_>(args, body); }

expr_ptr Compare(expr_ptr left, const vector<Compare_::cmpop_t>& ops,
                 const vector<expr_ptr>& comparators) {
  return make_shared<Compare_>(left, ops, comparators);
}

expr_ptr Call(expr_ptr func, const vector<expr_ptr>& args) {
  return make_shared<Call_>(func, args);
}
expr_ptr Num(double n) { return make_shared<Num_>(n); }

expr_ptr Constant(double value) { return make_shared<Constant_>(value); }

expr_ptr Attribute(expr_ptr value, const identifier& attr) {
  return make_shared<Attribute_>(value, attr);
}
expr_ptr Name(const identifier& id) { return make_shared<Name_>(id); }

arguments_ptr arguments(const vector<arg_ptr>& args) { return make_shared<arguments_>(args); }

arg_ptr arg(const identifier& arg) { return make_shared<arg_>(arg); }

}  // namespace pyast
