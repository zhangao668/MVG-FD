import torch.nn as nn

# IntermediateSequential扩展 PyTorch 的代码nn.Sequential。这个定制的顺序模块不仅可以返回模块序列的最终输出，还可以选择返回序列中每个模块的中间输出。
class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=False):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input): 
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs
        
