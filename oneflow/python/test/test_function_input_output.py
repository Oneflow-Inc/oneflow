import oneflow as flow
import numpy as np

def test_FixedTensorDef(test_case):
    @flow.function()
    def Foo(x=flow.FixedTensorDef((2, 5))): return x
    data = np.ones((2, 5), dtype=np.float32)
    Foo(data).get().ndarray()

def test_MirroredTensorDef(test_case):
    @flow.function()
    def Foo(x=flow.MirroredTensorDef((2, 5))): return x
    data = np.ones((1, 5), dtype=np.float32)
    Foo([data]).get()

def test_MirroredTensorListDef(test_case):
    @flow.function()
    def Foo(x=flow.MirroredTensorListDef((2, 5))): return x
    data = np.ones((1, 5), dtype=np.float32)
    Foo([[data]]).get()
