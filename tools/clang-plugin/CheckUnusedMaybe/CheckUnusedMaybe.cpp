#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CrashRecoveryContext.h"

using namespace clang;

class CheckUnusedMaybeVisitor
    : public RecursiveASTVisitor<CheckUnusedMaybeVisitor> {
public:
  explicit CheckUnusedMaybeVisitor(ASTContext *Context) : Context(Context) {
    {
      auto skip_filenames_env =
          llvm::StringRef(std::getenv("ONEFLOW_MAYBE_CHECK_SKIP_FN"));
      if (skip_filenames_env.empty()) {
        return;
      }
      skip_filenames_env.split(skip_filenames, ";");
      for (const auto &x : skip_filenames) {
        llvm::outs() << "skip: " << x << "\n";
      }
    }
    {
      auto only_filenames_env =
          llvm::StringRef(std::getenv("ONEFLOW_MAYBE_CHECK_ONLY_FN"));
      if (only_filenames_env.empty()) {
        return;
      }
      only_filenames_env.split(only_filenames, ";");
      for (const auto &x : only_filenames) {
        llvm::outs() << "only: " << x << "\n";
      }
    }
  }

  virtual bool VisitCompoundStmt(CompoundStmt *stmt) {
    if (!only_filenames.empty()) {
      bool skip = true;
      for (const auto &x : only_filenames) {
        if (Context->getSourceManager()
                .getFilename(stmt->getBeginLoc())
                .contains(x)) {
          skip = false;
        }
      }
      if (skip) {
        return true;
      }
    }

    for (const auto &x : skip_filenames) {
      if (Context->getSourceManager()
              .getFilename(stmt->getBeginLoc())
              .contains(x)) {
        return true;
      }
    }

    for (const auto &x : stmt->children()) {
      if (ExprWithCleanups *expr = dyn_cast<ExprWithCleanups>(x)) {
        std::string type = expr->getType().getAsString();
        if (type.substr(0, 12) == "class Maybe<" ||
            type.substr(0, 6) == "Maybe<") {
          DiagnosticsEngine &DE = Context->getDiagnostics();
          unsigned DiagID =
              DE.getCustomDiagID(DiagnosticsEngine::Error,
                                 "This function returns Maybe but the return "
                                 "value is ignored. Wrap it with JUST(..)?");
          auto DB = DE.Report(expr->getBeginLoc(), DiagID);
          DB.AddSourceRange(
              clang::CharSourceRange::getCharRange(expr->getSourceRange()));
        }
      }
      if (CallExpr *call = dyn_cast<CallExpr>(x)) {
        std::string returnType;
        llvm::CrashRecoveryContext CRC;
        CRC.RunSafely([&call, &returnType, this]() {
          if (auto *callee = call->getDirectCallee()) {
            returnType = callee->getReturnType().getAsString();
          } else {
            returnType = call->getCallReturnType(*this->Context).getAsString();
          }
        });
        if (returnType.substr(0, 12) == "class Maybe<" ||
            returnType.substr(0, 6) == "Maybe<") {
          DiagnosticsEngine &DE = Context->getDiagnostics();
          unsigned DiagID =
              DE.getCustomDiagID(DiagnosticsEngine::Error,
                                 "This function returns Maybe but the return "
                                 "value is ignored. Wrap it with JUST(..)?");
          auto DB = DE.Report(call->getBeginLoc(), DiagID);
          DB.AddSourceRange(
              clang::CharSourceRange::getCharRange(call->getSourceRange()));
        }
      }
    }
    return true;
  }

private:
  ASTContext *Context;
  llvm::SmallVector<llvm::StringRef, 10> skip_filenames;
  llvm::SmallVector<llvm::StringRef, 10> only_filenames;
};

class CheckUnusedMaybeConsumer : public clang::ASTConsumer {
public:
  explicit CheckUnusedMaybeConsumer(ASTContext *Context) : Visitor(Context) {}

  void HandleTranslationUnit(clang::ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }

private:
  CheckUnusedMaybeVisitor Visitor;
};

class CheckUnusedMaybeAction : public clang::PluginASTAction {
public:
  virtual std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef InFile) override {
    return std::make_unique<CheckUnusedMaybeConsumer>(
        &Compiler.getASTContext());
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    return true;
  }

  // Automatically run the plugin after the main AST action
  ActionType getActionType() override { return AddAfterMainAction; }
};

static FrontendPluginRegistry::Add<CheckUnusedMaybeAction>
    X("check-unused-maybe", "Check unused maybe");
