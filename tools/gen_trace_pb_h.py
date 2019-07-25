import oneflow.core.job.job_set_pb2
import oneflow.core.job.job_pb2
# import other pyprotos here

import os

TRACE_TEMPLATE_BASE = '''
template<typename T>
class Trace{
};
'''

TRACE_TEMPLATE = '''
template<>
class Trace<{}> {
public:
    std::string path;
    {}
};
'''

def first_letter_captain(name):
    name = name.split('_')
    for each in name:
        each[0] = each[0] + 'A' - 'a'
    return ''.join(name)

def get_instance(typename):
    '''
    Return a instance of a typename
    '''
    ctor = first_letter_captain(typename)
    return getattr(globals()['oneflow.core.{}'.format(typename+'_pb2')], ctor)

def get_children_list(instance):
    '''
    Get the instance's children by reflection
    Return A python list contains tuples, defined below
    [(Typename, message_type),(Typename, message_type)...]
    Note: This Method is NOT recursive
    '''
    raise NotImplementedError

def generate_output(typename, children_list):
    '''
    Return the generated Str

    '''
    raise NotImplementedError

def handle(pb_file_name, pb_file_path):
    typename = pb_file_name.split('.')[0]
    trace_pb_file_name = typename + '.trace.pb.h'

    inst = get_instance(typename)
    chd_lst = get_children_list(inst)
    gen_output_str = generate_output(typename, chd_lst)

    with open(trace_pb_file_name, 'w') as f:
        f.write(gen_output_str)

if __name__ == '__main__':
    for rt, ps, fs in os.walk('../build/oneflow/core'):
        for f in fs:
            if f.endswith('.pb.h'):
                handle(f[:-5], rt)