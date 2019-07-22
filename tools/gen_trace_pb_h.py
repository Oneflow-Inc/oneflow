import oneflow.core.job.job_set_pb2
import oneflow.core.job.job_pb2
# import other pyprotos here

TRACE_TEMPLATE_BASE = '''
'''
TRACE_TEMPLATE = '''
'''

def get_all_proto_type(path='../build/pyproto'):
    '''
    Return a list of all proto types(*.pb.h)
    '''
    raise NotImplementedError

def get_instance(typename):
    '''
    Return a instance of a typename
    '''
    raise NotImplementedError

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


if __name__ == '__main__':
    import os
    output_prefix = '../../xxx/'
    typenames = get_all_proto_type()
    
    for each_type in typenames:
        inst = get_instance(each_type)
        chd_lst = get_children_list(inst)
        gen_output_str = generate_output(each_type, chd_lst)
        output_filename = os.path.join(output_prefix, each_type + '.trace.pb.h')
        with open(output_filename, 'w') as f:
            f.write(gen_output_str)