import oneflow as flow


class BackwardHookFunction(flow.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        ctx.mark_non_differentiable(*[arg for arg in args if not arg.requires_grad])
        return args

    @staticmethod
    def backward(ctx, *args):
        return args
