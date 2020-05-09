import re

def oneflow_export(*field_paths):
    def Decorator(func_or_class):
        for field_path in field_paths:
            fields = field_path.split(".")
            assert len(fields) > 0
            global exported
            api = exported
            for field in fields: api = api._FindOrCreateSubApi(field)
            api._SetApiFunction(func_or_class)
        return func_or_class
    return Decorator


class OneflowApi(object):
    def __init__(self, name_path = ""):
        self.name_path_ = name_path
        self.api_function_ = _GetNotImplemented(name_path)
        self.sub_api_ = {}

    def __call__(self, *args, **kwargs):
        return self.api_function_(*args, **kwargs)

    def _SubApi(self): return self.sub_api_

    def _SetApiFunction(self, api_function):
        self.api_function_ = api_function

    def _FindOrCreateSubApi(self, field):
        assert re.match("^[\w]+[_\w\d]*$", field)
        if field in self.sub_api_: return self.sub_api_[field]
        sub_api = OneflowApi(self.name_path_ + "." + field)
        self.sub_api_[field] = sub_api
        setattr(self, field, sub_api)
        return sub_api

def _GetNotImplemented(api_name):
    def _NotImplemented(*args, **kwargs):
        raise NotImplementedError("api %s is not found".api_name)
    return _NotImplemented


exported = OneflowApi()
