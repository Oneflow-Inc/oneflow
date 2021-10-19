import oneflow as torch


class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        # This parameter will be copied to the new ScriptModule
        self.weight = torch.nn.Parameter(torch.rand(N, M))

        # When this submodule is used, it will be compiled
        self.linear = torch.nn.Linear(N, M)

    @torch.jit.script
    def forward(self, input):
        print("linear before")
        output = self.linear(input)
        print("linear after")
        return output


# scripted_module = torch.jit.script(MyModule(2, 3))
linear = MyModule(2, 3)
print(linear(torch.randn(2, 2)))
