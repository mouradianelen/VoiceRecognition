import torch
from torchviz import make_dot
from src import Net


model = Net()


dummy_input = torch.zeros(1, 1, 128, 16000)  


out = model(dummy_input)


visual_graph = make_dot(out, params=dict(list(model.named_parameters()) + [('input', dummy_input)]))


visual_graph.view()

