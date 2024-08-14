
import torch
import pdb

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)



def smooth_ln_fcs_temporary(ln, fcs, scales,shifts):
    ln.use_temporary_parameter = True
    if not isinstance(fcs, list):
        fcs = [fcs]
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.temp_bias = (ln.bias - shifts) / scales
    else:
        ln.temp_bias = (-1*shifts)/ scales

    ln.temp_weight = ln.weight / scales

    for fc in fcs:
        fc.use_temporary_parameter = True
        if hasattr(fc, 'bias') and fc.bias is not None:
            fc.temp_bias = fc.bias + fc.weight@shifts
        else:
            fc.temp_bias = fc.weight@shifts
        fc.temp_weight = fc.weight * scales.view(1,-1)


def smooth_fc_fc_temporary(fc1, fc2, scales,shifts=None, self_attn=None):
    # only support for v_proj and out_proh now.
    fc1.use_temporary_parameter = True
    fc2.use_temporary_parameter = True
    if hasattr(fc1, 'temp_weight'):
        fc1.temp_bias = fc1.temp_bias - shifts
        fc1.temp_bias = fc1.temp_bias/scales.view(-1)
        fc1.temp_weight = fc1.temp_weight/scales.view(-1,1)
    else:
        fc1.temp_bias = fc1.bias/scales.view(-1)
        fc1.temp_weight = fc1.weight/scales.view(-1,1)

    if fc2.weight.shape[-1] > scales.numel(): # llama GQA
        scales = scales.reshape(self_attn.num_key_value_heads,-1)
        n_head,d = scales.shape
        scales = scales[:,None,:].expand(n_head,self_attn.num_key_value_groups,d).reshape(-1)
        shifts = torch.zeros_like(scales)

    if hasattr(fc2, 'bias') and fc2.bias is not None:
        fc2.temp_bias = fc2.bias + fc2.weight@shifts
    else:
        fc2.temp_bias = fc2.weight@shifts
    fc2.temp_weight = fc2.weight * scales.view(1,-1)


def smooth_q_k_temporary(q_proj, k_proj, scales, self_attn=None):
    q_proj.use_temporary_parameter = True
    k_proj.use_temporary_parameter = True
    k_proj.temp_weight = k_proj.temp_weight*scales.view(-1,1)
    k_proj.temp_bias = k_proj.temp_bias*scales.view(-1)
    
    if q_proj.weight.shape[0] > scales.numel(): # llama GQA
        scales = scales.reshape(self_attn.num_key_value_heads,-1)
        n_head,d = scales.shape
        scales = scales[:,None,:].expand(n_head,self_attn.num_key_value_groups,d).reshape(-1)

    q_proj.temp_weight = q_proj.temp_weight/scales.view(-1,1)
    q_proj.temp_bias = q_proj.temp_bias/scales.view(-1)

def smooth_ln_fcs_inplace(ln, fcs, scales,shifts):
    ln.use_temporary_parameter = False
    if not isinstance(fcs, list):
        fcs = [fcs]
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.sub_(shifts)
        ln.bias.div_(scales)
    else:
        del ln.bias
        ln.register_buffer('bias',(-1*shifts)/scales)

    ln.weight.div_(scales)
    for fc in fcs:
        fc.use_temporary_parameter = False
        if hasattr(fc, 'bias') and fc.bias is not None:
            fc.bias.add_(fc.weight@shifts)
        else:
            del fc.bias
            fc.register_buffer('bias',fc.weight@shifts)
        fc.weight.mul_(scales.view(1,-1))


def smooth_fc_fc_inplace(fc1, fc2, scales,shifts=None, self_attn=None):
    # only support for v_proj and out_proh now.
    fc1.use_temporary_parameter = False
    fc2.use_temporary_parameter = False
    fc1.bias.sub_(shifts)
    fc1.bias.div_(scales.view(-1))
    fc1.weight.div_(scales.view(-1,1))

    if fc2.weight.shape[-1] > scales.numel(): # llama GQA
        scales = scales.reshape(self_attn.num_key_value_heads,-1)
        n_head,d = scales.shape
        scales = scales[:,None,:].expand(n_head,self_attn.num_key_value_groups,d).reshape(-1)
        assert (shifts).abs().sum() == 0,"must"
        shifts = torch.zeros_like(scales)
    
    if hasattr(fc2, 'bias') and fc2.bias is not None:
        fc2.bias.add_(fc2.weight@shifts)
    else:
        del fc2.bias
        fc2.register_buffer('bias',fc2.weight@shifts)
    fc2.weight.mul_(scales.view(1,-1))

def smooth_q_k_inplace(q_proj, k_proj, scales, self_attn=None):
    q_proj.use_temporary_parameter = False
    k_proj.use_temporary_parameter = False
    k_proj.weight.mul_(scales.view(-1,1))
    k_proj.bias.mul_(scales.view(-1))

    if q_proj.weight.shape[0] > scales.numel(): # llama GQA
        scales = scales.reshape(self_attn.num_key_value_heads,-1)
        n_head,d = scales.shape
        scales = scales[:,None,:].expand(n_head,self_attn.num_key_value_groups,d).reshape(-1)

    q_proj.weight.div_(scales.view(-1,1))
    q_proj.bias.div_(scales.view(-1))