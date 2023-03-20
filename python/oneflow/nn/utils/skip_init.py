import inspect
from oneflow.nn.modules.module import Module

def skip_init(module_cls, *args, **kwargs):
  if not issubclass(module_cls, Module):
    raise RuntimeError('Expected a Module; got {}'.format(module_cls))
  if 'device' not in inspect.signature(module_cls).parameters:
    raise RuntimeError('Module must support a \'device\' arg to skip initialization')
  
  final_device = kwargs.pop('device', 'cpu')
  kwargs['device'] = 'meta'
  # return module_cls(*args, **kwargs).to_empty(device=final_device)
  module_obj = module_cls(*args, **kwargs)
  return module_obj.to_empty(device=final_device)
