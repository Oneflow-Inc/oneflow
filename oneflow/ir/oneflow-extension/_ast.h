#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "llvm/ADT/ArrayRef.h"

namespace pyast {

#define BASE_KIND(name)                          \
  explicit name(name##Kind kind) : kind(kind) {} \
  virtual ~name() = default;                     \
  name##Kind getKind() const { return kind; }    \
                                                 \
 private:                                        \
  const name##Kind kind;

#define DERIVED_KIND(name, basic)                                                 \
  explicit name(name##Kind kind) : basic(basic::basic##Kind::name), kind(kind) {} \
  virtual ~name() = default;                                                      \
  name##Kind getKind() const { return kind; }                                     \
                                                                                  \
 private:                                                                         \
  const name##Kind kind;

using namespace std;

typedef std::string identifier;

class SingleArg {
  identifier name;

 public:
  explicit SingleArg(identifier name) : name(name){};
  identifier getName() { return name; }
};

class Arguments {
  std::vector<unique_ptr<SingleArg>> args;

 public:
  explicit Arguments(std::vector<std::unique_ptr<SingleArg>> args) : args(move(args)){};
  llvm::ArrayRef<unique_ptr<SingleArg>> getArgs() { return args; }
};

class ExprASTBase {
 public:
  enum ExprASTBaseKind {
    BinOpAST,
    LambdaAST,
    CallAST,
    CompareAST,
    ConstantAST,
    AttributeAST,
    NameAST,
  };
  BASE_KIND(ExprASTBase);
};

class BinOpAST : ExprASTBase {
 public:
  enum BinOpASTKind {
    Add,
    Sub,
    Mult,
    Div,
    Pow,
  };
  explicit BinOpAST(BinOpASTKind kind, unique_ptr<ExprASTBase> lhs, unique_ptr<ExprASTBase> rhs)
      : ExprASTBase(ExprASTBase ::ExprASTBaseKind ::BinOpAST),
        kind(kind),
        lhs(move(lhs)),
        rhs(move(rhs)) {}
  virtual ~BinOpAST() = default;
  BinOpASTKind getKind() const { return kind; }

  ExprASTBase* getLHS() { return lhs.get(); }
  ExprASTBase* getRHS() { return rhs.get(); }

 private:
  const BinOpASTKind kind;
  unique_ptr<ExprASTBase> lhs;
  unique_ptr<ExprASTBase> rhs;
};

// class LambdaAST : ExprASTBase {};

class CallAST : ExprASTBase {
  // this need to convert py lib math to arith dialect
};

class CompareAST : ExprASTBase {
 public:
  enum CompareASTKind { Eq, NotEq, Lt, LtE, Gt, GtE, Is, IsNot, In, NotIn };
  explicit CompareAST(unique_ptr<ExprASTBase> left,
                      std::vector<unique_ptr<ExprASTBase>> comparators,
                      std::vector<CompareASTKind> kinds)
      : ExprASTBase(ExprASTBase ::ExprASTBaseKind ::CompareAST),
        left(move(left)),
        kinds(move(kinds)),
        comparators(move(comparators)){};

  virtual ~CompareAST() = default;

  int getLen() { return kinds.size() == comparators.size() ? kinds.size() : -1; }

  std::pair<CompareASTKind, ExprASTBase*> getPair(int index) {
    if (index < getLen()) { return {kinds[index], comparators[index].get()}; }
    return {CompareASTKind::Eq, nullptr};
  }

 private:
  unique_ptr<ExprASTBase> left;
  std::vector<CompareASTKind> kinds;
  std::vector<unique_ptr<ExprASTBase>> comparators;
};
class ConstantAST : ExprASTBase {};
class AttributeAST : ExprASTBase {};
class NameAST : ExprASTBase {};

class StmtASTBase {
 public:
  enum StmtASTBaseKind {
    FunctionAST,
    ReturnAST,
    AssignAST,
    IfAST,
    RaiseAST,
    ExprAST,
  };
  BASE_KIND(StmtASTBase);
};
class FunctionAST : public StmtASTBase {};
class ReturnAST : public StmtASTBase {
  ExprASTBase value;
};
class AssignAST : public StmtASTBase {};
class IfAST : public StmtASTBase {};
class RaiseAST : public StmtASTBase {};
class ExprAST : public StmtASTBase {};

class ModuleAST {};

}  // namespace pyast

/**
--ASDL's 4 builtin types are: --identifier, int, string,
    constant

        module Python {
    stmt = FunctionDef(identifier name, arguments args,
                       stmt* body, expr* decorator_list, expr? returns,
                       string? type_comment)

        //   | AsyncFunctionDef(identifier name, arguments args,
        //                      stmt* body, expr* decorator_list, expr? returns,
        //                      string? type_comment)

        //   | ClassDef(identifier name,
        //      expr* bases,
        //      keyword* keywords,
        //      stmt* body,
        //      expr* decorator_list)
          | Return(expr? value)

        //   | Delete(expr* targets)
          | Assign(expr* targets, expr value, string? type_comment)
        //   | AugAssign(expr target, operator op, expr value)
        //   -- 'simple' indicates that we annotate simple name without parens
        //   | AnnAssign(expr target, expr annotation, expr? value, int simple)

        //   -- use 'orelse' because else is a keyword in target languages
        //   | For(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
        //   | AsyncFor(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
        //   | While(expr test, stmt* body, stmt* orelse)
          | If(expr test, stmt* body, stmt* orelse)
        //   | With(withitem* items, stmt* body, string? type_comment)
        //   | AsyncWith(withitem* items, stmt* body, string? type_comment)

        //   | Match(expr subject, match_case* cases)

          | Raise(expr? exc, expr? cause)
        //   | Try(stmt* body, excepthandler* handlers, stmt* orelse, stmt* finalbody)
        //   | Assert(expr test, expr? msg)

        //   | Import(alias* names)
        //   | ImportFrom(identifier? module, alias* names, int? level)

        //   | Global(identifier* names)
        //   | Nonlocal(identifier* names)
          | Expr(expr value)
        //   | Pass | Break | Continue

        //   -- col_offset is the byte offset in the utf8 string the parser uses
        //   attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)

        //   -- BoolOp() can use left & right?
    expr = BoolOp(boolop op, expr* values)
        //  | NamedExpr(expr target, expr value)
         | BinOp(expr left, operator op, expr right)
        //  | UnaryOp(unaryop op, expr operand)
         | Lambda(arguments args, expr body)
        //  | IfExp(expr test, expr body, expr orelse)
        //  | Dict(expr* keys, expr* values)
        //  | Set(expr* elts)
        //  | ListComp(expr elt, comprehension* generators)
        //  | SetComp(expr elt, comprehension* generators)
        // //  | DictComp(expr key, expr value, comprehension* generators)
        //  | GeneratorExp(expr elt, comprehension* generators)
        //  -- the grammar constrains where yield expressions can occur
        //  | Await(expr value)
        //  | Yield(expr? value)
        //  | YieldFrom(expr value)
        //  -- need sequences for compare to distinguish between
        //  -- x < 4 < 3 and (x < 4) < 3
         | Compare(expr left, cmpop* ops, expr* comparators)
         | Call(expr func, expr* args, keyword* keywords)
        //  | FormattedValue(expr value, int conversion, expr? format_spec)
        //  | JoinedStr(expr* values)
         | Constant(constant value, string? kind)

        //  -- the following expression can appear in assignment context
         | Attribute(expr value, identifier attr, expr_context ctx)
        //  | Subscript(expr value, expr slice, expr_context ctx)
        //  | Starred(expr value, expr_context ctx)
         | Name(identifier id, expr_context ctx)
        //  | List(expr* elts, expr_context ctx)
        //  | Tuple(expr* elts, expr_context ctx)

        //  -- can appear only in Subscript
        //  | Slice(expr? lower, expr? upper, expr? step)

          -- col_offset is the byte offset in the utf8 string the parser uses
          attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)

    expr_context = Load | Store | Del

    boolop = And | Or

    operator = Add | Sub | Mult | MatMult | Div | Mod | Pow | LShift
                 | RShift | BitOr | BitXor | BitAnd | FloorDiv

    unaryop = Invert | Not | UAdd | USub

    cmpop = Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn

    comprehension = (expr target, expr iter, expr* ifs, int is_async)

    excepthandler = ExceptHandler(expr? type, identifier? name, stmt* body)
                    attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)

    arguments = (arg* posonlyargs, arg* args, arg? vararg, arg* kwonlyargs,
                 expr* kw_defaults, arg? kwarg, expr* defaults)

    arg = (identifier arg, expr? annotation, string? type_comment)
           attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)

    -- keyword arguments supplied to call (NULL identifier for **kwargs)
    keyword = (identifier? arg, expr value)
               attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)

    -- import name with optional 'as' alias.
    alias = (identifier name, identifier? asname)
             attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)

    withitem = (expr context_expr, expr? optional_vars)

    match_case = (pattern pattern, expr? guard, stmt* body)

    // pattern = MatchValue(expr value)
    //         | MatchSingleton(constant value)
    //         | MatchSequence(pattern* patterns)
    //         | MatchMapping(expr* keys, pattern* patterns, identifier? rest)
    //         | MatchClass(expr cls, pattern* patterns, identifier* kwd_attrs, pattern*
kwd_patterns)

    //         | MatchStar(identifier? name)
    //         -- The optional "rest" MatchMapping parameter handles capturing extra mapping keys

    //         | MatchAs(pattern? pattern, identifier? name)
    //         | MatchOr(pattern* patterns)

    //          attributes (int lineno, int col_offset, int end_lineno, int end_col_offset)

    type_ignore = TypeIgnore(int lineno, string tag)
}

**/
