import os
import os.path as osp
import math
from typing import Optional, Union, List

import torch
from diffusers import FluxPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb

from .prompt_parser import PromptParser
from .attention_utils import AttentionHook, AttnProcessorWithHook


class DDTATransBlockProcessor(AttnProcessorWithHook):
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        query = self.reshape_seq_to_heads(query, attn)
        key = self.reshape_seq_to_heads(key, attn)
        value = self.reshape_seq_to_heads(value, attn)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_query_proj = self.reshape_seq_to_heads(encoder_hidden_states_query_proj, attn)
        encoder_hidden_states_key_proj = self.reshape_seq_to_heads(encoder_hidden_states_key_proj, attn)
        encoder_hidden_states_value_proj = self.reshape_seq_to_heads(encoder_hidden_states_value_proj, attn)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        scale_factor = 1 / math.sqrt(query.shape[-1])
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = torch.softmax(attn_weight, dim=-1)

        if self.hook.store_mode:
            if self.hook.trans_selected_ids[self.hook.step_index][self.hook.trans_block_counter]:
                if self.hook.hook_cc:
                    cc = attn_weight[:, :, :self.hook.max_sequence_length, :self.hook.max_sequence_length]  # N*H*512*512
                    self.hook.store("trans", f"{self.hook.step_index}_{self.hook.trans_block_counter}", "cc", cc)
                
                if self.hook.hook_ci:
                    ci = attn_weight[:, :, :self.hook.max_sequence_length, self.hook.max_sequence_length:]  # N*H*512*4096
                    self.hook.store("trans", f"{self.hook.step_index}_{self.hook.trans_block_counter}", "ci", ci)
                
                if self.hook.hook_ic:
                    ic = attn_weight[:, :, self.hook.max_sequence_length:, :self.hook.max_sequence_length]  # N*H*4096*512
                    self.hook.store("trans", f"{self.hook.step_index}_{self.hook.trans_block_counter}", "ic", ic)
                
                if self.hook.hook_ii:
                    ii = attn_weight[:, :, self.hook.max_sequence_length:, self.hook.max_sequence_length:]  # N*H*4096*4096
                    self.hook.store("trans", f"{self.hook.step_index}_{self.hook.trans_block_counter}", "ii", ii)
                
                if self.hook.hook_v:
                    self.hook.store("trans", f"{self.hook.step_index}_{self.hook.trans_block_counter}", "v", value)  # N*H*4608*128
        else:
            if self.hook.edit_info is not None:
                unchange_mapper = self.hook.edit_info["unchange_mapper"]
                if self.hook.extend_padding:
                    unchange_mapper.extend(self.hook.edit_info["pad_mapper"])
                change_pos = self.hook.edit_info["change_pos"]
                if self.hook.trans_selected_ids[self.hook.step_index][self.hook.trans_block_counter]:
                    if self.hook.hook_cc:
                        cc = attn_weight[:, :, :self.hook.max_sequence_length, :self.hook.max_sequence_length]  # N*H*512*512
                        if self.hook.cc_strategy != "keep":
                            cc_source = self.hook.get("trans", f"{self.hook.step_index}_{self.hook.trans_block_counter}", "cc")
                            uc_map_t = {}
                            for item in unchange_mapper:
                                uc_map_t[item[1]] = item[0]
                            for i in range(cc.shape[2]):
                                for j in range(cc.shape[3]):
                                    if i in change_pos or j in change_pos:
                                        cc[:, :, i, j] *= self.hook.cc_amplify
                                        continue
                                    if i not in uc_map_t.keys() or j not in uc_map_t.keys():
                                        continue
                                    s_i = uc_map_t[i]
                                    s_j = uc_map_t[j]
                                    if self.hook.cc_strategy == "replace":
                                        cc[:, :, i, j] = cc_source[:, :, s_i, s_j]
                                    elif self.hook.cc_strategy == "add":
                                        cc[:, :, i, j] += cc_source[:, :, s_i, s_j]
                                    elif self.hook.cc_strategy == "mean":
                                        cc[:, :, i, j] = (cc[:, :, i, j] + cc_source[:, :, s_i, s_j]) / 2
                        else:
                            cc *= self.hook.cc_amplify
                        attn_weight[:, :, :self.hook.max_sequence_length, :self.hook.max_sequence_length] = cc

                    if self.hook.hook_ci:
                        ci_source = self.hook.get("trans", f"{self.hook.step_index}_{self.hook.trans_block_counter}", "ci")
                        ci = attn_weight[:, :, :self.hook.max_sequence_length, self.hook.max_sequence_length:]  # N*H*512*4096
                        for pos in change_pos:
                            ci[:, :, pos, :] *= self.hook.ci_amplify
                        for item in unchange_mapper:
                            if self.hook.cc_strategy == "replace":
                                ci[:, :, item[1], :] = ci_source[:, :, item[0], :]
                            elif self.hook.cc_strategy == "add":
                                ci[:, :, item[1], :] += ci_source[:, :, item[0], :]
                            elif self.hook.cc_strategy == "mean":
                                ci[:, :, item[1], :] = (ci[:, :, item[1], :] + ci_source[:, :, item[0], :]) / 2
                        attn_weight[:, :, :self.hook.max_sequence_length, self.hook.max_sequence_length:] = ci

                    if self.hook.hook_ic:
                        ic_source = self.hook.get("trans", f"{self.hook.step_index}_{self.hook.trans_block_counter}", "ic")
                        ic = attn_weight[:, :, self.hook.max_sequence_length:, :self.hook.max_sequence_length]  # N*H*4096*512
                        for pos in change_pos:
                            ic[:, :, :, pos] *= self.hook.ic_amplify
                        for item in unchange_mapper:
                            if self.hook.cc_strategy == "replace":
                                ic[:, :, :, item[1]] = ic_source[:, :, :, item[0]]
                            elif self.hook.cc_strategy == "add":
                                ic[:, :, :, item[1]] += ic_source[:, :, :, item[0]]
                            elif self.hook.cc_strategy == "mean":
                                ic[:, :, :, item[1]] = (ic[:, :, :, item[1]] + ic_source[:, :, :, item[0]]) / 2
                        attn_weight[:, :, self.hook.max_sequence_length:, :self.hook.max_sequence_length] = ic
                        
                    if self.hook.hook_ii:
                        ii_source = self.hook.get("trans", f"{self.hook.step_index}_{self.hook.trans_block_counter}", "ii")
                        ii = attn_weight[:, :, self.hook.max_sequence_length:, self.hook.max_sequence_length:]  # N*H*4096*4096
                        if self.hook.ii_strategy == "replace":
                            ii = ii_source
                        elif self.hook.ii_strategy == "add":
                            ii += ii_source
                        elif self.hook.ii_strategy == "mean":
                            ii = (ii + ii_source) / 2
                        attn_weight[:, :, self.hook.max_sequence_length:, self.hook.max_sequence_length:] = ii
                            
                    if self.hook.hook_v:
                        v_source = self.hook.get("trans", f"{self.hook.step_index}_{self.hook.trans_block_counter}", "v")
                        vc_source = v_source[:, :, :self.hook.max_sequence_length, :]  # N*H*512*128
                        vi_source = v_source[:, :, self.hook.max_sequence_length:, :]  # N*H*4096*128
                        vc = value[:, :, :self.hook.max_sequence_length, :]  # N*H*512*128
                        vi = value[:, :, self.hook.max_sequence_length:, :]  # N*H*4096*128

                        if self.hook.vc_strategy != "keep":
                            for pos in change_pos:
                                vc[:, :, pos, :] *= self.hook.vc_amplify
                            for item in unchange_mapper:
                                if self.hook.cc_strategy == "replace":
                                    vc[:, :, item[1], :] = vc_source[:, :, item[0], :]
                                elif self.hook.cc_strategy == "add":
                                    vc[:, :, item[1], :] += vc_source[:, :, item[0], :]
                                elif self.hook.cc_strategy == "mean":
                                    vc[:, :, item[1], :] = (vc[:, :, item[1], :] + vc_source[:, :, item[0], :]) / 2
                        else:
                            vc *= self.hook.vc_amplify
                        value[:, :, :self.hook.max_sequence_length, :] = vc

                        if self.hook.vi_strategy == "replace":
                            vi = vi_source
                        elif self.hook.vi_strategy == "add":
                            vi += vi_source
                        elif self.hook.vi_strategy == "mean":
                            vi = (vi + vi_source) / 2
                        value[:, :, self.hook.max_sequence_length:, :] = vi
        self.hook.trans_block_counter += 1

        hidden_states = attn_weight @ value
        hidden_states = self.reshape_heads_to_seq(hidden_states)
        hidden_states = hidden_states.to(query.dtype)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class DDTASingleTransBlockProcessor(AttnProcessorWithHook):
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        query = self.reshape_seq_to_heads(query, attn)
        key = self.reshape_seq_to_heads(key, attn)
        value = self.reshape_seq_to_heads(value, attn)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        scale_factor = 1 / math.sqrt(query.shape[-1])
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = torch.softmax(attn_weight, dim=-1)

        if self.hook.store_mode:
            if self.hook.single_trans_selected_ids[self.hook.step_index][self.hook.single_trans_block_counter]:
                if self.hook.hook_cc:
                    cc = attn_weight[:, :, :self.hook.max_sequence_length, :self.hook.max_sequence_length]  # N*H*512*512
                    self.hook.store("single_trans", f"{self.hook.step_index}_{self.hook.single_trans_block_counter}", "cc", cc)
                
                if self.hook.hook_ci:
                    ci = attn_weight[:, :, :self.hook.max_sequence_length, self.hook.max_sequence_length:]  # N*H*512*4096
                    self.hook.store("single_trans", f"{self.hook.step_index}_{self.hook.single_trans_block_counter}", "ci", ci)
                
                if self.hook.hook_ic:
                    ic = attn_weight[:, :, self.hook.max_sequence_length:, :self.hook.max_sequence_length]  # N*H*4096*512
                    self.hook.store("single_trans", f"{self.hook.step_index}_{self.hook.single_trans_block_counter}", "ic", ic)
                
                if self.hook.hook_ii:
                    ii = attn_weight[:, :, self.hook.max_sequence_length:, self.hook.max_sequence_length:]  # N*H*4096*4096
                    self.hook.store("single_trans", f"{self.hook.step_index}_{self.hook.single_trans_block_counter}", "ii", ii)
                
                if self.hook.hook_v:
                    self.hook.store("single_trans", f"{self.hook.step_index}_{self.hook.single_trans_block_counter}", "v", value)  # N*H*4608*128
        else:
            if self.hook.edit_info is not None:
                unchange_mapper = self.hook.edit_info["unchange_mapper"]
                if self.hook.extend_padding:
                    unchange_mapper.extend(self.hook.edit_info["pad_mapper"])
                change_pos = self.hook.edit_info["change_pos"]
                if self.hook.single_trans_selected_ids[self.hook.step_index][self.hook.single_trans_block_counter]:
                    if self.hook.hook_cc:
                        cc = attn_weight[:, :, :self.hook.max_sequence_length, :self.hook.max_sequence_length]  # N*H*512*512
                        if self.hook.cc_strategy != "keep":
                            cc_source = self.hook.get("single_trans", f"{self.hook.step_index}_{self.hook.single_trans_block_counter}", "cc")
                            uc_map_t = {}
                            for item in unchange_mapper:
                                uc_map_t[item[1]] = item[0]
                            for i in range(cc.shape[2]):
                                for j in range(cc.shape[3]):
                                    if i in change_pos or j in change_pos:
                                        cc[:, :, i, j] *= self.hook.cc_amplify
                                        continue
                                    if i not in uc_map_t.keys() or j not in uc_map_t.keys():
                                        continue
                                    s_i = uc_map_t[i]
                                    s_j = uc_map_t[j]
                                    if self.hook.cc_strategy == "replace":
                                        cc[:, :, i, j] = cc_source[:, :, s_i, s_j]
                                    elif self.hook.cc_strategy == "add":
                                        cc[:, :, i, j] += cc_source[:, :, s_i, s_j]
                                    elif self.hook.cc_strategy == "mean":
                                        cc[:, :, i, j] = (cc[:, :, i, j] + cc_source[:, :, s_i, s_j]) / 2
                        else:
                            cc *= self.hook.cc_amplify
                        attn_weight[:, :, :self.hook.max_sequence_length, :self.hook.max_sequence_length] = cc
                    
                    if self.hook.hook_ci:
                        ci_source = self.hook.get("single_trans", f"{self.hook.step_index}_{self.hook.single_trans_block_counter}", "ci")
                        ci = attn_weight[:, :, :self.hook.max_sequence_length, self.hook.max_sequence_length:]  # N*H*512*4096
                        for pos in change_pos:
                            ci[:, :, pos, :] *= self.hook.ci_amplify
                        for item in unchange_mapper:
                            if self.hook.cc_strategy == "replace":
                                ci[:, :, item[1], :] = ci_source[:, :, item[0], :]
                            elif self.hook.cc_strategy == "add":
                                ci[:, :, item[1], :] += ci_source[:, :, item[0], :]
                            elif self.hook.cc_strategy == "mean":
                                ci[:, :, item[1], :] = (ci[:, :, item[1], :] + ci_source[:, :, item[0], :]) / 2
                        attn_weight[:, :, :self.hook.max_sequence_length, self.hook.max_sequence_length:] = ci
                    
                    if self.hook.hook_ic:
                        ic_source = self.hook.get("single_trans", f"{self.hook.step_index}_{self.hook.single_trans_block_counter}", "ic")
                        ic = attn_weight[:, :, self.hook.max_sequence_length:, :self.hook.max_sequence_length]  # N*H*4096*512
                        for pos in change_pos:
                            ic[:, :, :, pos] *= self.hook.ic_amplify
                        for item in unchange_mapper:
                            if self.hook.cc_strategy == "replace":
                                ic[:, :, :, item[1]] = ic_source[:, :, :, item[0]]
                            elif self.hook.cc_strategy == "add":
                                ic[:, :, :, item[1]] += ic_source[:, :, :, item[0]]
                            elif self.hook.cc_strategy == "mean":
                                ic[:, :, :, item[1]] = (ic[:, :, :, item[1]] + ic_source[:, :, :, item[0]]) / 2
                        attn_weight[:, :, self.hook.max_sequence_length:, :self.hook.max_sequence_length] = ic
                    
                    if self.hook.hook_ii:
                        ii_source = self.hook.get("single_trans", f"{self.hook.step_index}_{self.hook.single_trans_block_counter}", "ii")
                        ii = attn_weight[:, :, self.hook.max_sequence_length:, self.hook.max_sequence_length:]  # N*H*4096*4096
                        if self.hook.ii_strategy == "replace":
                            ii = ii_source
                        elif self.hook.ii_strategy == "add":
                            ii += ii_source
                        elif self.hook.ii_strategy == "mean":
                            ii = (ii + ii_source) / 2
                        attn_weight[:, :, self.hook.max_sequence_length:, self.hook.max_sequence_length:] = ii
                    
                    if self.hook.hook_v:
                        v_source = self.hook.get("single_trans", f"{self.hook.step_index}_{self.hook.single_trans_block_counter}", "v")
                        vc_source = v_source[:, :, :self.hook.max_sequence_length, :]  # N*H*512*128
                        vi_source = v_source[:, :, self.hook.max_sequence_length:, :]  # N*H*4096*128
                        vc = value[:, :, :self.hook.max_sequence_length, :]  # N*H*512*128
                        vi = value[:, :, self.hook.max_sequence_length:, :]  # N*H*4096*128

                        if self.hook.vc_strategy != "keep":
                            for pos in change_pos:
                                vc[:, :, pos, :] *= self.hook.vc_amplify
                            for item in unchange_mapper:
                                if self.hook.cc_strategy == "replace":
                                    vc[:, :, item[1], :] = vc_source[:, :, item[0], :]
                                elif self.hook.cc_strategy == "add":
                                    vc[:, :, item[1], :] += vc_source[:, :, item[0], :]
                                elif self.hook.cc_strategy == "mean":
                                    vc[:, :, item[1], :] = (vc[:, :, item[1], :] + vc_source[:, :, item[0], :]) / 2
                        else:
                            vc *= self.hook.vc_amplify
                        value[:, :, :self.hook.max_sequence_length, :] = vc

                        if self.hook.vi_strategy == "replace":
                            vi = vi_source
                        elif self.hook.vi_strategy == "add":
                            vi += vi_source
                        elif self.hook.vi_strategy == "mean":
                            vi = (vi + vi_source) / 2
                        value[:, :, self.hook.max_sequence_length:, :] = vi
        self.hook.single_trans_block_counter += 1

        hidden_states = attn_weight @ value
        hidden_states = self.reshape_heads_to_seq(hidden_states)
        hidden_states = hidden_states.to(query.dtype)

        return hidden_states


class DDTA(AttentionHook):
    def __init__(
        self,
        pipe: FluxPipeline, 
        hook_cc: bool = False,
        hook_ci: bool = True,
        hook_ic: bool = True,
        hook_ii: bool = False,
        hook_v: bool = False,
        cc_strategy: str = "replace",
        ci_strategy: str = "replace",
        ic_strategy: str = "replace",
        ii_strategy: str = "replace",
        vc_strategy: str = "replace",
        vi_strategy: str = "replace",
        cc_amplify: float = 1.0,
        ci_amplify: float = 1.0,
        ic_amplify: float = 1.0,
        vc_amplify: float = 1.0,
        extend_padding: bool = False,
        **kwargs
    ):
        assert hook_cc or hook_ci or hook_ic or hook_ii or hook_v, "At least one of hook_cc, hook_ci, hook_ic, hook_ii, or hook_v must be True."
        assert cc_strategy in ["keep", "replace", "add", "mean"]
        assert ci_strategy in ["replace", "add", "mean"]
        assert ic_strategy in ["replace", "add", "mean"]
        assert ii_strategy in ["replace", "add", "mean"]
        assert vc_strategy in ["keep", "replace", "add", "mean"]
        assert vi_strategy in ["keep", "replace", "add", "mean"]
        super().__init__(pipe, **kwargs)
        self.hook_cc = hook_cc
        self.hook_ci = hook_ci
        self.hook_ic = hook_ic
        self.hook_ii = hook_ii
        self.hook_v = hook_v
        self.cc_strategy = cc_strategy
        self.ci_strategy = ci_strategy
        self.ic_strategy = ic_strategy
        self.ii_strategy = ii_strategy
        self.vc_strategy = vc_strategy
        self.vi_strategy = vi_strategy
        self.cc_amplify = cc_amplify
        self.ci_amplify = ci_amplify
        self.ic_amplify = ic_amplify
        self.vc_amplify = vc_amplify
        self.extend_padding = extend_padding

        self.store_mode = True
        self.trans_store = {}
        self.single_trans_store = {}
        self.edit_info = None

    def set_hook_steps(
        self, 
        steps: Optional[Union[int, List[int]]] = None, 
        trans_start_layer: Union[int, List[int]] = 0, 
        single_trans_start_layer: Union[int, List[int]] = 0, 
        inverse: bool = False,
        order: int = 1,
        is_fireflow: bool = False
    ):
        if is_fireflow:
            num_inference_steps = self.num_inference_steps + 1
        else:
            num_inference_steps = self.num_inference_steps

        if steps is None:
            hook_steps = [True for _ in range(order*num_inference_steps)]
        else:
            hook_steps = []
            if isinstance(steps, int):
                steps = [steps]
            for i in range(num_inference_steps):
                for _ in range(order):
                    if i in steps:
                        hook_steps.append(True)
                    else:
                        hook_steps.append(False)

        num_trans_blocks = self.pipe.transformer.config.num_layers
        num_single_trans_blocks = self.pipe.transformer.config.num_single_layers
        self.trans_selected_ids = {}
        self.single_trans_selected_ids = {}
        if inverse:
            hook_steps = list(reversed(hook_steps))
            for i in range(order*num_inference_steps):
                step = order*num_inference_steps - i - 1
                if hook_steps[i]:
                    if self.hook_trans_block:
                        self.trans_selected_ids[step] = []
                        for j in range(num_trans_blocks):
                            if j >= trans_start_layer:
                                self.trans_selected_ids[step].append(True)
                            else:
                                self.trans_selected_ids[step].append(False)
                    else:
                        self.trans_selected_ids[step] = [False for _ in range(num_trans_blocks)]
                    if self.hook_single_trans_block:
                        self.single_trans_selected_ids[step] = []
                        for j in range(num_single_trans_blocks):
                            if j >= single_trans_start_layer:
                                self.single_trans_selected_ids[step].append(True)
                            else:
                                self.single_trans_selected_ids[step].append(False)
                    else:
                        self.single_trans_selected_ids[step] = [False for _ in range(num_single_trans_blocks)]
                else:
                    self.trans_selected_ids[step] = [False for _ in range(num_trans_blocks)]
                    self.single_trans_selected_ids[step] = [False for _ in range(num_single_trans_blocks)]
            self.step_index = order*num_inference_steps - 1
        else:
            for i in range(order*num_inference_steps):
                if hook_steps[i]:
                    if self.hook_trans_block:
                        self.trans_selected_ids[i] = []
                        for j in range(num_trans_blocks):
                            if j >= trans_start_layer:
                                self.trans_selected_ids[i].append(True)
                            else:
                                self.trans_selected_ids[i].append(False)
                    else:
                        self.trans_selected_ids[i] = [False for _ in range(num_trans_blocks)]
                    if self.hook_single_trans_block:
                        self.single_trans_selected_ids[i] = []
                        for j in range(num_single_trans_blocks):
                            if j >= single_trans_start_layer:
                                self.single_trans_selected_ids[i].append(True)
                            else:
                                self.single_trans_selected_ids[i].append(False)
                    else:
                        self.single_trans_selected_ids[i] = [False for _ in range(num_single_trans_blocks)]
                else:
                    self.trans_selected_ids[i] = [False for _ in range(num_trans_blocks)]
                    self.single_trans_selected_ids[i] = [False for _ in range(num_single_trans_blocks)]
            self.step_index = 0

        self.trans_block_counter = 0
        self.single_trans_block_counter = 0

    def set_edit_prompt(self, source: str, target: str):
        prompt_parser = PromptParser(self.pipe, self.max_sequence_length)
        self.edit_info = prompt_parser(source, target)
    
    def store(self, block_type: str, block_id: str, name: str, value: torch.Tensor):
        assert block_type in ["trans", "single_trans"], "block_type must be 'trans' or 'single_trans'."
        if self.in_memory:
            if block_type == "trans":
                if block_id not in self.trans_store.keys():
                    self.trans_store[block_id] = {}
                self.trans_store[block_id][name] = value.detach().cpu()
            elif block_type == "single_trans":
                if block_id not in self.single_trans_store.keys():
                    self.single_trans_store[block_id] = {}
                self.single_trans_store[block_id][name] = value.detach().cpu()
        else:
            torch.save(value, osp.join(self.cache_dir, f"{block_type}_{block_id}_{name}.pth"))

    def get(self, block_type: str, block_id: str, name: str):
        assert block_type in ["trans", "single_trans"], "block_type must be 'trans' or 'single_trans'."
        if self.in_memory:
            if block_type == "trans":
                value = self.trans_store[block_id][name].to(self.pipe.device)
            elif block_type == "single_trans":
                value = self.single_trans_store[block_id][name].to(self.pipe.device)
        else:
            value = torch.load(osp.join(self.cache_dir, f"{block_type}_{block_id}_{name}.pth"), map_location=self.pipe.device)
            if self.remove_cache:
                os.remove(osp.join(self.cache_dir, f"{block_type}_{block_id}_{name}.pth"))
        return value
    
    def next_step(self, inverse: bool):
        if inverse:
            self.step_index -= 1
        else:
            self.step_index += 1
        self.trans_block_counter = 0
        self.single_trans_block_counter = 0
    
    def activate(self):
        self.apply(DDTATransBlockProcessor, DDTASingleTransBlockProcessor)
