import oneflow.python.framework.runtime_mode as runtime_mode
import inspect
import re

def oneflow_export(*api_names, **kwargs):
    if len(kwargs) == 1:
        assert 'mode' in kwargs
        modes = kwargs['mode']
        if not isinstance(modes, (list, tuple)): modes = [modes]
        modes = list(modes)
        for mode in modes: assert runtime_mode.IsValidMode(mode)
    else:
        assert len(kwargs) == 0
        modes = [runtime_mode.NORMAL_MODE, runtime_mode.GLOBAL_MODE, runtime_mode.DEVICE_MODE]
    return _GetOneflowExportDecorator(api_names, modes)

def _GetOneflowExportDecorator(api_names, oneflow_modes=[]):
    assert type(oneflow_modes) is list
    assert len(oneflow_modes) > 0
    def Decorator(func_or_class):
        for api_name in api_names:
            fields = api_name.split(".")
            assert len(fields) > 0
            global exported
            api = exported
            for field in fields: api = api._FindOrCreateSubApi(field)
            for mode in oneflow_modes: api._SetApiFunction(mode, func_or_class)
        return func_or_class
    return Decorator


class OneflowApi(object):
    def __init__(self, api_name = ""):
        self.api_name_ = api_name
        self.mode2api_function_ = {}
        self.sub_api_ = {}

    def __call__(self, *args, **kwargs):
        mode = runtime_mode.CurrentMode()
        supported = self.mode2api_function_
        if mode in supported: return supported[mode](*args, **kwargs)
        raise ApiNotImplementedError(self.api_name_, mode, supported.keys())

    @property
    def _class(self):
        assert len(self.mode2api_function_) > 0
        ret_class = None
        for _, value_class in self.mode2api_function_.items():
            if ret_class is None:
                ret_class = value_class
            else:
                assert ret_class is value_class
        assert inspect.isclass(ret_class)
        return ret_class

    def _SubApi(self): return self.sub_api_

    def _SetApiFunction(self, mode, api_function):
        self.mode2api_function_[mode] = api_function

    def _FindOrCreateSubApi(self, field):
        assert re.match("^[\w]+[_\w\d]*$", field)
        if field in self.sub_api_: return self.sub_api_[field]
        sub_api = OneflowApi(self.api_name_ + "." + field)
        self.sub_api_[field] = sub_api
        setattr(self, field, sub_api)
        return sub_api

exported = OneflowApi("oneflow")

@oneflow_export('error.ApiNotImplementedError')
class ApiNotImplementedError(Exception):
    def __init__(self, api_name, current_mode, supported_modes):
        self.api_name_ = api_name
        self.current_mode_ = current_mode
        self.supported_modes_ = supported_modes

    def __str__(self):
        return ("\n\napi name: %s\nsupported modes: [%s]\ncurrent mode: %s\n"
                %(self.api_name_, ", ".join(self.supported_modes_), self.current_mode_))
        return ret
