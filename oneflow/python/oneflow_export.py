from oneflow.python.lib.core.enable_if import enable_if
import inspect
import re

def oneflow_export(*api_names, **kwargs):
    if len(kwargs) == 1:
        assert 'enable_if' in kwargs
        hob_expr = kwargs['enable_if']
    else:
        assert len(kwargs) == 0
        hob_expr = None
    return _GetOneflowExportDecorator(api_names, hob_expr)

def _GetOneflowExportDecorator(api_names, hob_expr = None):
    def Decorator(func_or_class):
        for api_name in api_names:
            fields = api_name.split(".")
            assert len(fields) > 0
            global exported
            api = exported
            for field in fields: api = api._FindOrCreateSubApi(field)
            api._SetApiFuncOrClass(func_or_class, hob_expr)
        return func_or_class
    return Decorator

class OneflowApi(object):
    def __init__(self, api_name = ""):
        self.api_name_ = api_name
        self.func_or_class_ = None
        self.sub_api_ = {}

    def __call__(self, *args, **kwargs):
        assert self.func_or_class_ is not None
        return self.func_or_class_(*args, **kwargs)

    @property
    def _class(self):
        assert inspect.isclass(self.func_or_class_)
        return self.func_or_class_

    def _SubApi(self): return self.sub_api_

    def _SetApiFuncOrClass(self, func_or_class, hob_expr = None):
        func_or_class = _GetInvokerIfIsSpecializedFunctor(func_or_class)
        if hob_expr is not None:
            func_or_class = _GetSpecializedFunctor(func_or_class, hob_expr, self.api_name_)
            func_or_class = _GetInvokerIfIsSpecializedFunctor(func_or_class)
        self.func_or_class_ = func_or_class

    def _FindOrCreateSubApi(self, field):
        assert re.match("^[\w]+[_\w\d]*$", field)
        if field in self.sub_api_: return self.sub_api_[field]
        sub_api = OneflowApi(self.api_name_ + "." + field)
        self.sub_api_[field] = sub_api
        setattr(self, field, sub_api)
        return sub_api

def _GetInvokerIfIsSpecializedFunctor(func_or_class):
    if (inspect.isclass(func_or_class) and hasattr(func_or_class, 'invoke')
            and hasattr(func_or_class.invoke, '__is_specialization_supported__')):
        return func_or_class.invoke
    return func_or_class

def _GetSpecializedFunctor(func, hob_expr, api_name):
    class Functor:
        def default(get_failed_info, *args, **kwargs):
            raise NotImplementedError(get_failed_info(api_name))

        @enable_if(hob_expr, default)
        def invoke(*args, **kwargs):
            return func(*args, **kwargs)
    return Functor

exported = OneflowApi("oneflow")
