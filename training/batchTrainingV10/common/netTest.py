import torch
from torch.utils.tensorboard import SummaryWriter

from BigEfficientVideoModel import BigEfficientVideoModel
from TinyVideoModel import TinyVideoModel

model = TinyVideoModel()
dummy_input = torch.randn(1, 10, 3, 224, 224)
writer = SummaryWriter()
writer.add_graph(model, dummy_input)
writer.close()
