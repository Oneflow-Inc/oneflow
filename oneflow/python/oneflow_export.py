import oneflow.python.framework.runtime_mode
import re

def oneflow_export(*field_paths):
    modes = [runtime_mode.NORMAL_MODE, runtime_mode.GLOBAL_MODE, runtime_mode.DEVICE_MODE]
    return _GetOneflowExportDecorator(field_paths, modes)

def _GetOneflowExportDecorator(field_paths, oneflow_modes):
    def Decorator(func_or_class):
        for field_path in field_paths:
            fields = field_path.split(".")
            assert len(fields) > 0
            global exported
            api = exported
            for field in fields: api = api._FindOrCreateSubApi(field)
            for mode in oneflow_modes: api._SetApiFunction(mode, func_or_class)
        return func_or_class
    return Decorator


class OneflowApi(object):
    def __init__(self, name_path = ""):
        self.name_path_ = name_path
        self.mode2api_function_ = {}
        self.sub_api_ = {}

    def __call__(self, *args, **kwargs):
        mode = runtime_mode.CurrentMode()
        supported = self.mode2api_function_
        if mode in supported: return supported[mode](*args, **kwargs)
        raise ApiNotImplementedError(self.name_path_, mode, supported.keys())

    def _SubApi(self): return self.sub_api_

    def _SetApiFunction(self, mode, api_function):
        self.mode2api_function_[mode] = api_function

    def _FindOrCreateSubApi(self, field):
        assert re.match("^[\w]+[_\w\d]*$", field)
        if field in self.sub_api_: return self.sub_api_[field]
        sub_api = OneflowApi(self.name_path_ + "." + field)
        self.sub_api_[field] = sub_api
        setattr(self, field, sub_api)
        return sub_api

def ApiNotImplementedError(Exception):
    def __init__(self, api_name, current_mode, supported_modes):
        self.api_name_ = api_name
        self.current_mode_ = current_mode
        self.supported_modes_ = supported_modes

    def __str__(self):
        return ("\napi name: %s\nsupported mode: %s\n current mode: %s\n"
                %(self.api_name_, ", ".join(self.supported_modes_), self.current_mode_))
        return ret


exported = OneflowApi()
