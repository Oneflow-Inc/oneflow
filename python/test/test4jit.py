import oneflow

@autotest()
def testcase4module():
    model = oneflow.nn.Sequential(
    oneflow.nn.Linear(5, 3),
    oneflow.nn.Linear(3, 1)
    )
    if isinstance(model, oneflow.jit.ScriptModule):
        print(True)
    else:
        print(False)
#原先报错为ModuleNotFoundError: No module named 'oneflow.jit.__ScriptModule'
#修改后打印信息为false，修改完毕

