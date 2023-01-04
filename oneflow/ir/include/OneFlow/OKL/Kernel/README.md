## context相关概念与其生命周期：
### LauncherState
    单个okl kernel的资源的管理者
     - LauncherContext的维护者，负责对于context信息的更新;
     - JIT Engine的所有者

### LauncherContext
    单个okl kernel的运行时管理者，维护被该kernel所wrap的所有oneflow ops的资源
    op_ctx_vec

### WrapperContext(op, ctx):
    单个被okl wrap的oneflow op的管理者，用于统一管理编译、运行时准备、运行时计算三个部分的生命周期。
    1. 编译时期
    - reg_ctx_(op) 
     - device
     - inputs/outputs
     - kernel
     - user config
    
    2. 运行时准备
    - init_ctx_(reg_ctx_, ctx)
    - state_(reg_ctx, init_ctx_)
    - cache_(reg_ctx, init_ctx_)
    - buffer_(ctx)

    3. 运行时计算
    - compute_ctx_(ctx)