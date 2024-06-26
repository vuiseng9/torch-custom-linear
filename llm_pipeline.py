from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from functools import wraps
import custom_linear

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id) 
model.eval()

########################################
# Wrap linear layer with custom forward
########################################
class CustomLinearWrapper(nn.Module):
    def __init__(self, linear_layer, quiet=True):
        super(CustomLinearWrapper, self).__init__()
        self.linear_layer = linear_layer
        self.quiet = quiet

    @property
    def weight(self):
        return self.linear_layer.weight

    @property
    def bias(self):
        return self.linear_layer.bias
    
    def forward(self, input):
        if not self.quiet:
            print("[Info]: Entering custom linear implementation")
        return custom_linear.dense_linear(input, self.weight, self.bias)
    
    def __repr__(self):
        return f"CustomLinear(in_features={self.linear_layer.in_features}, out_features={self.linear_layer.out_features}, bias={self.linear_layer.bias is not None})"


def replace_linear_layers_with_wrapper(model, quiet=True):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name in ['up_proj', 'gate_proj', 'down_proj']: # just replacing MLP as an example
            # Wrap the existing nn.Linear layer with CustomLinearWrapper
            wrapped_linear = CustomLinearWrapper(module, quiet=quiet)
            setattr(model, name, wrapped_linear)
        else:
            # Recursively apply to child modules
            replace_linear_layers_with_wrapper(module, quiet=quiet)
    return model

model = replace_linear_layers_with_wrapper(model, quiet=True)
########################################
print(model)

prompt="Alan Turing theorized that computers would one day become"
input_ids=tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids)

output_ids = model.generate(input_ids, do_sample=False, top_p=None, num_beams=1, max_new_tokens=128)

output = tokenizer.batch_decode(output_ids.cpu())

# output= tokenizer.decode(output_ids[0])
print(output[0])
