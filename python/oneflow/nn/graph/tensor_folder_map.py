import oneflow as flow
from oneflow.framework.tensor import Tensor
# import torch as flow
# from torch import Tensor
from typing import List, Callable, Tuple

_unary_ops = set(['Transpose', 'Reshape', 'ScalarAdd', 'Sqrt',])
_binary_ops = set(['BroadcastMul', 'BroadcastDiv', 'BroadcastSub'])

def _is_unary_op(token):
    global _unary_ops
    return any(token.find(op)!=-1 for op in _unary_ops)

def _is_binary_op(token):
    global _binary_ops
    return token in _binary_ops

def _cal_transpose(token, var):
    arg_start_index = token.find('[')
    arg_end_index = token.find(']')
    arg = [int(x) for x in token[arg_start_index+1:arg_end_index].split(",")[:-1]]
    # TODO: 不确定 transpose的参数是什么样的，因为 resnet18 没用到 transpose
    return flow.transpose(var, *arg)

def _cal_reshape(token, var):
    arg_start_index = token.find('[')
    arg_end_index = token.find(']')
    arg = [int(x) for x in token[arg_start_index+1:arg_end_index].split(",")[:-1]]
    return flow.reshape(var, arg)

def _cal_scalar_add(token, var):
    arg_start_index = token.find('(')
    arg_end_index = token.find(')')
    arg = int(token[arg_start_index+1:arg_end_index])
    return var + arg

def _get_eval_func(postfix_rule: List[str]):
    
    def eval(*inputs: List[Tensor]) -> Tensor:
        global _unary_ops
        global _binary_ops
        cnt = 0
        stack: List[Tensor] = []
        for token in postfix_rule:
            if _is_unary_op(token):
                var = stack.pop()
                if token.find('Transpose') != -1 :
                    stack.append(_cal_transpose(token, var))
                elif token.find('Reshape') != -1 :
                    stack.append(_cal_reshape(token, var))
                elif token.find('ScalarAdd') != -1 :
                    stack.append(_cal_scalar_add(token, var))
                elif token == 'Sqrt' :
                    stack.append(flow.sqrt(var))
                else :
                    raise ValueError("Bad Unary Operator " + token)
            elif _is_binary_op(token):
                rhs = stack.pop()
                lhs = stack.pop()
                if token == 'BroadcastMul':
                    stack.append(lhs * rhs)
                elif token == 'BroadcastDiv':
                    stack.append(lhs / rhs)
                elif token == 'BroadcastSub':
                    stack.append(lhs - rhs)
                else:
                    raise ValueError("Bad Binaary Operator " + token)
            else:
                stack.append(inputs[cnt])
                cnt+=1
        
        assert len(stack) == 1, "Bad postfix rule: " + " ".join(postfix_rule)
        return stack[0]

    return eval


def _transform_infix_to_postfix(infix_rule: str) -> Tuple[List[str], List[str]]:
    global _unary_ops
    global _binary_ops
    #  infix to postfix
    input_tensor_names = []
    op_stack = []
    postfix_rule = []
    for token in infix_rule.split():
        if token == "(" :
            op_stack.append(token)
        elif token == ")" :
            while len(op_stack)!=0 and op_stack[-1]!="(" :
                postfix_rule.append(op_stack.pop())
            assert len(op_stack)!=0 and op_stack[-1]=="(", "input rules with unmatch brackets: " + infix_rule
            op_stack.pop()
        elif _is_unary_op(token) or _is_binary_op(token):
            # Because all ops are wrapped in parentheses
            # we don’t need to compare the priority of the top of the stack and the new op
            op_stack.append(token)
        else: # tensor
            postfix_rule.append(token)
            input_tensor_names.append(token)
    while len(op_stack)!=0 :
        postfix_rule.append(op_stack.pop())
    
    return postfix_rule, input_tensor_names

# help func
def _transform(infix_rule: str) -> Tuple[List[str], str, Callable[[List[Tensor]], Tensor]]:
    '''
    transform a infix eval rule to a TensorFolderMap element
    for example, input rule is ( model.conv1.weight ) Broadcast ( Reshape ( ( model.bn1.weight ) BroadcastDiv ( Sqrt ( ScalarAdd ( model.bn1.running_var ) ) ) ) )
    the output are [
        input_names = [model.conv1.weight, model.bn1.weight, model.bn1.running_var]
        ouput_name = infix_rule
        a function to eval rule, its input are tensors which names are input_names, its output is the eval result, which tensor is match ouput_name
    ]
    '''
    postfix_rule, input_tensor_names = _transform_infix_to_postfix(infix_rule)

    return input_tensor_names, infix_rule, _get_eval_func(postfix_rule)


class TensorFolderMap:
    def __init__(self, rules:List[str]):
        super().__init__()
        # include input tensor name list, output tensor name, a function of (List[Tensor]) -> Tensor
        self._map: List[Tuple[List[str], str, Callable[[List[Tensor]], Tensor]]] = []
        for rule in rules:
            self._map.append(_transform(rule))
    
    @property
    def map(self):
        return self._map

    
    
if __name__ == "__main__" :
    infix_rule = "( model.conv1.weight ) BroadcastMul ( Reshape ( ( model.bn1.weight ) BroadcastDiv ( Sqrt ( ScalarAdd ( model.bn1.running_var ) ) ) ) )"
    postfix_rule, input_tensor_names = _transform_infix_to_postfix(infix_rule)
    print(" ".join(postfix_rule))
    print(input_tensor_names)
    
    