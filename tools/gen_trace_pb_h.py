import oneflow.core.job.job_set_pb2 as job_set_pb2
import oneflow.core.job.job_pb2 as job_pb2


import os

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

TRACE_TEMPLATE = '''
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
    if typename + '_pb2' in globals():
        return getattr(globals()[typename+'_pb2'], ctor)()
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
        methods.append('  const auto& get_{}();'.format(name))
        if label == 3: # repeated
            attributes.append('  Trace<Repeated<{}>> {};'.format(childtype, name))
        elif label == 2: # required
            attributes.append('  Trace<{}> {};'.format(childtype, name))
        elif label == 1: # optional
            attributes.append('  Trace<{}> {};'.format(childtype, name))
            methods.append('  bool has_{}() const;'.format(name))
        else:
            raise Exception('No such label!')
    return TRACE_TEMPLATE % (typename, '\n'.join(methods), '\n'.join(attributes))

def handle(typename, pb_file_path):
    trace_pb_file_name = typename + '.trace.pb.h'
    inst = get_instance(typename)
    if inst == None: return
    chd_lst = get_children_list(inst)
    gen_output_str = generate_output(typename, chd_lst)

    with open(os.path.join(pb_file_path, trace_pb_file_name), 'w') as f:
        f.write(gen_output_str)

if __name__ == '__main__':
    for rt, ps, fs in os.walk('../build/oneflow/core'):
        for f in fs:
            if f.endswith('.pb.h'):
                handle(f[:-5], './trace_tmp')