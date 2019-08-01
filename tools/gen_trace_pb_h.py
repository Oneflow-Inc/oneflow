import os
from importlib import import_module

TRACE_TEMPLATE_BASE = '''
template<typename T>
class Trace{
public:
    Trace(std::string name, std::string path="") {
    }
    std::string _name, _path;
};

template<Typename T>
class Trace<Repeated<T>> {
};
'''

TRACE_TEMPLATE = '''#include "trace_base.h"
template<>
class Trace<%s> {
public:
%s
private:
%s
};
'''

def get_instance(typename):
    '''
    Return a instance of a typename
    '''
    ctor = ''.join(map(lambda x: x.capitalize(), typename.split('_')))
    if typename in globals():
        ctor_method = None
        if hasattr(globals()[typename], ctor):
            ctor_method = getattr(globals()[typename], ctor)
        if callable(ctor_method):
            return ctor_method()
    return None

def get_children_list(instance):
    '''
    Get the instance's children by reflection
    Return A python list contains tuples, defined below
    [(name, typename, label),(name, typename, label)...]
    label:
    1 : optional
    2 : required
    3 : repeated
    Note: This Method is NOT recursive
    '''
    ret = []
    fields = instance.DESCRIPTOR.fields
    for child in fields:
        child_inst = getattr(instance, child.name)
        typename = type(child_inst).__name__
        if typename.startswith('Repeated'):
            if typename == 'RepeatedScalarContainer':
                typename = 'std::string'
            else:
                typename = type(child_inst.add()).__name__
        ret.append((child.name, typename, child.label))
    return ret

def generate_output(typename, children_list):
    '''
    Return the generated Str
    '''
    attributes = []
    methods = []
    for name, childtype, label in children_list:
        methods.append('  const Trace<{}>& {}();'.format(childtype, name))
        if label == 3: # repeated
            attributes.append('  Trace<Repeated<{}>> {}_;'.format(childtype, name))
        elif label == 2: # required
            attributes.append('  Trace<{}> {}_;'.format(childtype, name))
        elif label == 1: # optional
            attributes.append('  Trace<{}> {}_;'.format(childtype, name))
            methods.append('  bool has_{}() const;'.format(name))
        else:
            raise Exception('No such label!')
    return TRACE_TEMPLATE % (typename, '\n'.join(methods), '\n'.join(attributes))

def handle(typename, pb_file_path):
    trace_pb_file_name = typename + '.trace.pb.h'
    inst = get_instance(typename)
    if inst == None: return
    typename = type(inst).__name__
    chd_lst = get_children_list(inst)
    gen_output_str = generate_output(typename, chd_lst)

    with open(os.path.join(pb_file_path, trace_pb_file_name), 'w') as f:
        f.write(gen_output_str)

if __name__ == '__main__':
    output_path = './trace_tmp'
    blacklist = set(['ibverbs'])
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with open(os.path.join(output_path, 'trace_base.h'), 'w') as f:
        f.write(TRACE_TEMPLATE_BASE)

    for rt, ps, fs in os.walk('../build/oneflow/core'):
        for f in fs:
            if f.endswith('.pb.h'):
                typename = f[:-5]
                if typename in blacklist: continue
                globals()[typename] = import_module('.'.join(rt.split('/')[2:]) + '.%s_pb2'%(typename))
                print(globals()[typename])
                print('-----------------------------')
                handle(typename, output_path)
