from typing import Optional, Union, List

import torch
from diffusers import FluxPipeline
from diffusers.models.attention_processor import Attention, FluxAttnProcessor2_0


class AttnProcessorWithHook:
    def __init__(self, hook):
        self.hook = hook

    @staticmethod
    def reshape_seq_to_heads(x: torch.Tensor, attn: Attention):
        assert len(x.shape) == 3
        batch_size, _, inner_dim = x.shape
        head_dim = inner_dim // attn.heads
        x = x.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        return x
    
    @staticmethod
    def reshape_heads_to_seq(x: torch.Tensor):
        assert len(x.shape) == 4
        batch_size, heads, _, head_dim = x.shape
        x = x.transpose(1, 2).reshape(batch_size, -1, heads * head_dim)
        return x


class AttentionHook:
    def __init__(
        self,
        pipe: FluxPipeline, 
        hook_trans_block: bool = False,
        hook_single_trans_block: bool = True,
        num_inference_steps: int = 50,
        max_sequence_length: int = 512,
        in_memory: bool = True,
        cache_dir: Optional[str] = None,
        remove_cache: bool = True
    ):
        assert in_memory or cache_dir is not None, "cache_dir must be specified if in_memory is False."

        self.pipe = pipe
        self.hook_trans_block = hook_trans_block
        self.hook_single_trans_block = hook_single_trans_block
        self.num_inference_steps = num_inference_steps
        self.max_sequence_length = max_sequence_length
        self.in_memory = in_memory
        self.cache_dir = cache_dir
        self.remove_cache = remove_cache
        self.trans_block_counter = None
        self.single_trans_block_counter = None
    
    def set_hook_steps(self, steps: Optional[Union[int, List[int]]] = None, inverse: bool = False):
        if steps is None:
            hook_steps = [True for _ in range(self.num_inference_steps)]
        else:
            hook_steps = []
            if isinstance(steps, int):
                steps = [steps]
            for i in range(self.num_inference_steps):
                if i in steps:
                    hook_steps.append(True)
                else:
                    hook_steps.append(False)
        if inverse:
            hook_steps = reversed(hook_steps)
        
        num_trans_blocks = self.pipe.transformer.config.num_layers
        num_single_trans_blocks = self.pipe.transformer.config.num_single_layers
        self.trans_selected_ids = []
        self.single_trans_selected_ids = []
        for step in range(self.num_inference_steps):
            if hook_steps[step]:
                if self.hook_trans_block:
                    self.trans_selected_ids.extend([i for i in range(step*num_trans_blocks, (step+1)*num_trans_blocks)])
                if self.hook_single_trans_block:
                    self.single_trans_selected_ids.extend([i for i in range(step*num_single_trans_blocks, (step+1)*num_single_trans_blocks)])

    def apply(
        self, 
        trans_attn_processor: AttnProcessorWithHook, 
        single_trans_attn_processor: AttnProcessorWithHook
    ):
        def parse(net, block_type: str):
            if net.__class__.__name__ == "Attention":
                if block_type == "transformer_block":
                    if self.hook_trans_block:
                        net.set_processor(trans_attn_processor(self))
                elif block_type == "single_transformer_block":
                    if self.hook_single_trans_block:
                        net.set_processor(single_trans_attn_processor(self))
            elif hasattr(net, "children"):
                for child in net.children():
                    parse(child, block_type)
        
        for name, module in self.pipe.transformer.named_children():
            if name == "transformer_blocks":
                parse(module, "transformer_block")
            elif name == "single_transformer_blocks":
                parse(module, "single_transformer_block")

    def remove(self):
        def parse(net):
            if net.__class__.__name__ == "Attention":
                net.set_processor(FluxAttnProcessor2_0())
            elif hasattr(net, "children"):
                for child in net.children():
                    parse(child)
        
        for name, module in self.pipe.transformer.named_children():
            if name == "transformer_blocks" or name == "single_transformer_blocks":
                parse(module)

    def reset(self):
        self.trans_block_counter = None
        self.single_trans_block_counter = None
