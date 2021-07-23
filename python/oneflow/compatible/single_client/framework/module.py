from oneflow.compatible.single_client.python.framework import id_util as id_util


class Module(object):
    def __init__(self, name=None):
        if name is None:
            name = id_util.UniqueStr("Module_")
        self.module_name_ = name
        self.call_seq_no_ = 0

    @property
    def module_name(self):
        return self.module_name_

    @property
    def call_seq_no(self):
        return self.call_seq_no_

    def forward(self, *args):
        raise NotImplementedError()

    def __call__(self, *args):
        ret = self.forward(*args)
        self.call_seq_no_ = self.call_seq_no_ + 1
        return ret

    def __del__(self):
        assert (
            getattr(type(self), "__call__") is Module.__call__
        ), "do not override __call__"
