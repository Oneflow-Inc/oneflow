## context相关概念与其生命周期：
### LauncherState
LauncherState 是OpKernelState的派生类，在okl kernel的初始化kernel state的阶段被创建。
``` c++
LauncherState final : public user_op::OpKernelState
```
每个LauncherState拥有一个LauncherContext管理运行的上下文和一个JIT Engine管理运行时引擎。
    单个okl kernel的资源的管理者
     - LauncherContext的维护者，负责对于context信息的更新;
     - JIT Engine的所有者

### LauncherContext
LauncherContext作为单个okl kernel的上下文的管理者，维护若干有序的oneflow op的上下文资源信息，每个oneflow op对应的上下文资源对应一个专门的WrapperContext作为一个总体的维护者。
因此LauncherContext下维护一系列编译期状态的WrapperContext和运行时状态的WrapperContext以对应不同阶段的上下文。这些ctx与oneflow op一一对应。
```
class LauncherContext final {
  bool inferred_ = false;

  std::vector<CompileTimeWrapperContext> compile_ctx_vec_;
  std::vector<RunTimeWrapperContext> run_ctx_vec_;
};
``` 

### WrapperContext(op, ctx):
    单个被okl wrap的oneflow op的管理者，编译期存在的东西在初始化后不可被改变，运行时需要做一个懒汉模式的infer推导流程。
    1. 推导前
    - reg_ctx_(op) 
     - device
     - inputs/outputs
     - kernel
     - user config
    
    2. 推导后
    - init_ctx_(reg_ctx_, ctx)
    - state_(reg_ctx, init_ctx_)
    - cache_(reg_ctx, init_ctx_)
    - compute_ctx_(ctx)

```
class CompileTimeWrapperContext {
  std::shared_ptr<const RegContext> reg_ctx_;
};

class RunTimeWrapperContext : public CompileTimeWrapperContext {
  std::shared_ptr<ComputeContext> compute_ctx_;
  std::shared_ptr<InitContext> init_ctx_;

  std::shared_ptr<user_op::OpKernelState> kernel_state_;
  std::shared_ptr<user_op::OpKernelCache> kernel_cache_;
};
```
CompileTimeWrapperContext维护着从ir获取得到的上下文信息并加以封装到reg ctx，用于作为后面推导RunTimeWrapperContext的输入之一。
RunTimeWrapperContext通过CompileTimeWrapperContext的信息以及okl kernel所创建的comp ctx以及tmp buffer等资源组成了单个op运行时计算所需的实际上下文环境。通过创建的init ctx，创建kernel state，kernel cache等资源用于kernel的compute计算。