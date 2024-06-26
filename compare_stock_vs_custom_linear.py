import torch
from torch import nn

from layer.dense_linear import DenseLinear 

d=2

one_layer_net_torch_stock = nn.Linear(in_features=d, out_features=4*d, bias=True)
print("\n[Info]: Creating one layer neural network with nn.Linear")
print(f"one_layer_net_torch_stock:\n{one_layer_net_torch_stock}")

one_layer_net_torch_extcpp = DenseLinear.from_linear(one_layer_net_torch_stock)
print("\n[Info]: Creating one layer neural network with DenseFC")
print(f"one_layer_net_torch_extcpp:\n{one_layer_net_torch_extcpp}")

N=3
x = torch.rand(N, d)
print(f"\n[Info]: Creating test input x ({N}, {d})")
print(f"x: {x.shape}\n{x}")


with torch.no_grad():
    o_torch_stock = one_layer_net_torch_stock(x)
    o_torch_extcpp = one_layer_net_torch_extcpp(x)

print(f"\none_layer_net_torch_stock(x): {o_torch_stock.shape}\n{o_torch_stock}")
print(f"\none_layer_net_torch_extcpp(x): {o_torch_extcpp.shape}\n{o_torch_extcpp}")

print(f"\none_layer_net_torch_stock(x) ==  one_layer_net_torch_extcpp(x) ???\n{torch.equal(o_torch_stock, o_torch_extcpp)}")
