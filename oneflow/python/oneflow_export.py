import re

class OneflowModule(object):
    def __init__(self):
        pass

exported_objects = OneflowModule()
exported_object_names = set()

def oneflow_export(*field_paths):
    def Decorator(func_or_class):
        global exported_objects
        global exported_object_names
        for field_path in field_paths:
            fields = field_path.split(".")
            assert len(fields) > 0
            exported_object_names.add(fields[0])
            obj = exported_objects
            for field in fields[:-1]:
                assert re.match("^[_\w]+[_\w\d]*$", field)
                if hasattr(obj, field) == False: setattr(obj, field, OneflowModule())
                obj = getattr(obj, field)
            field = fields[-1]
            assert hasattr(obj, field) == False
            setattr(obj, field, func_or_class)
        return func_or_class
    return Decorator
