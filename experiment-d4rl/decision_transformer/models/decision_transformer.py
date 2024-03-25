import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

import sys
import os
# use ../../decision_transformer as decision_transformer when run as main
if __name__=="__main__":
    sys.path.insert(0, os.path.abspath('../..'))
    sys.path.insert(0, os.path.abspath('..'))

from decision_transformer.models.model import TrajectoryModel
from transformers.models.gpt2 import GPT2Tokenizer
from decision_transformer.models.trajectory_gpt2 import GPT2Model, GPT2LMHeadModel
from decision_transformer.models.trajectory_gpt2_LoRA import GPT2Model_LoRA, GPT2LMHeadModel_LoRA
from decision_transformer.models.trajectory_gpt2_LoRA import GPT2Config_LoRA

from decision_transformer.models.utils import ResidualBlock, MLPBlock


class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, attention_dropout# , batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len
        self.batch_first = True

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]

        norm_x = self.norm1(x)

        is_batched = x.dim() == 3
        if self.batch_first and is_batched:
            norm_x = norm_x.transpose(1, 0)
        attention_out = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        if self.batch_first and is_batched:
            attention_out = attention_out.transpose(1, 0)
        # by default pytorch attention does not use dropout
        # after final attention weights projection, while minGPT does:
        # https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L70 # noqa
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    @property
    def transformer(self):
        return self.transformer_model.transformer
      
    def __init__(
        self,
        args,
        state_dim,
        act_dim,
        hidden_size,
        max_length=None,
        max_ep_len=4096,
        action_tanh=True,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        self.use_llm = args["pretrained_lm"] is not None
        
        if args["pretrained_lm"] is not None:
            print("Loading from pretrained "+args["pretrained_lm"]+" model")
            if args['lora']:
                name_or_path = f"/apdcephfs/share_1563664/ztjiaweixu/huggingface/{args['pretrained_lm']}"
                if not os.path.exists(name_or_path):
                    pretrained_name_or_path = args['pretrained_lm']
                else:
                    pretrained_name_or_path = name_or_path
                config = GPT2Config_LoRA.from_pretrained(pretrained_name_or_path)
                self.transformer_model = GPT2LMHeadModel_LoRA.from_pretrained(
                    pretrained_name_or_path, config=config
                )
                if not os.path.exists(name_or_path):
                    config.save_pretrained(name_or_path)
                    self.transformer_model.save_pretrained(name_or_path)
            else:
                config = transformers.GPT2Config.from_pretrained(args["pretrained_lm"])
                config.resid_pdrop = args["dropout"]
                self.transformer_model = GPT2LMHeadModel.from_pretrained(
                    args["pretrained_lm"],
                    config=config,
                )
            hidden_size = config.n_embd
            self.hidden_size = config.n_embd
        else:
            self.emb_drop = nn.Dropout(args["dropout"])
            self.out_norm = nn.LayerNorm(hidden_size)
            self.blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        seq_len=3 * max_ep_len,
                        embedding_dim=hidden_size,
                        num_heads=args["n_head"],
                        attention_dropout=0.1,
                        residual_dropout=0.1,
                    )
                    for _ in range(args["n_layer"])
                ]
            )
            # if args['lora']:
            #     config = GPT2Config_LoRA.from_pretrained("gpt2")
            #     self.transformer_model = GPT2LMHeadModel_LoRA(config)
            # else:
            #     config = transformers.GPT2Config(
            #         n_embd=hidden_size,
            #         **kwargs
            #     )
            #     # config = transformers.GPT2Config.from_pretrained("gpt2")
            #     # config.resid_pdrop = args["dropout"]
            #     # NOTE: If you comment two lines above, then we adopt non-pretrained 3-layer DT; otherwise we use the same config as the pretrained gpt2 model, but with random weights
            #     self.transformer_model = GPT2LMHeadModel(config)
            # hidden_size = config.n_embd
            # self.hidden_size = config.n_embd

        if args["extend_positions"] and max_ep_len > config.n_positions:
            current_max_pos, embed_size = self.transformer.wpe.weight.shape
            new_encoder_pos_embed = self.transformer.wpe.weight.new_empty(
                max_ep_len, embed_size
            )
            # copy position embeddings over and over to initialize the new position embeddings
            orig_k = 2
            k = orig_k
            step = current_max_pos - k
            new_encoder_pos_embed[:k] = self.transformer.wpe.weight[:k]
            while k < max_ep_len - 1:
                new_encoder_pos_embed[k : (k + step)] = self.transformer.wpe.weight[
                    orig_k : min(max_ep_len - k + orig_k, current_max_pos)
                ]
                k += step
            self.transformer.wpe.weight.data = new_encoder_pos_embed

        if args["extend_positions"]:
            self.embed_timestep = self.transformer.wpe
        else:
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
    
        if args["mlp_embedding"]:
            self.embed_return = ResidualBlock(1, hidden_size)
            self.embed_state = ResidualBlock(self.state_dim, hidden_size)
            self.embed_action = ResidualBlock(self.act_dim, hidden_size)
        else:
            self.embed_return = torch.nn.Linear(1, hidden_size)
            self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
            self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        
        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        if args["mlp_embedding"]:
          if args["share_input_output_proj"]: raise ValueError("With MLP in embeddings, you cannot share the projections")
        #   self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
          self.predict_action = MLPBlock(self.hidden_size, self.act_dim, self.hidden_size)
        #   self.predict_return = torch.nn.Linear(hidden_size, 1)
        else:
          if args["share_input_output_proj"]:
            # self.predict_state = lambda x: F.linear(x, self.embed_state.weight.t())
            # self.predict_return = lambda x: F.linear(x, self.embed_return.weight.t())
            self.predict_action = lambda x: F.tanh(
                F.linear(x, self.embed_action.weight.t())
            )
          else:
            # self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
            self.predict_action = nn.Sequential(
                *(
                    [nn.Linear(hidden_size, self.act_dim)]
                    + ([nn.Tanh()] if action_tanh else [])
                )
            )
            # self.predict_return = torch.nn.Linear(hidden_size, 1)
        
        self.past_key_values = None
        print(self)
        if not self.use_llm:
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)


    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        attention_mask=None,
        past_key_values=None,
    ):

        batch_size, seq_length = states.shape[0], states.shape[1]
        
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        # embed each modality with a different head
        
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
       
        stacked_inputs = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        all_embs = self.embed_ln(stacked_inputs)

        stacked_inputs = all_embs + time_embeddings.repeat_interleave(3, dim=1)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        if self.use_llm:
            transformer_outputs = self.transformer(
                inputs_embeds=stacked_inputs,
                attention_mask=stacked_attention_mask,
                past_key_values=None,  # self.past_key_values,
                use_cache=True,
            )
            x = transformer_outputs["last_hidden_state"]
            self.past_key_values = transformer_outputs["past_key_values"]
        else:
            # True value indicates that the corresponding key value will be ignored
            stacked_attention_mask = ~stacked_attention_mask.to(torch.bool)
            stacked_inputs = self.emb_drop(stacked_inputs)
            for block in self.blocks:
                stacked_inputs = block(stacked_inputs, padding_mask=stacked_attention_mask)
            stacked_inputs =  self.out_norm(stacked_inputs)
            x = stacked_inputs

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        action_preds = self.predict_action(x[:, 1])  # predict next action given state
        return None, action_preds, None, None

    def get_action(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        past_key_values=None,
        **kwargs
    ):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length :]
            actions = actions[:, -self.max_length :]
            returns_to_go = returns_to_go[:, -self.max_length :]
            timesteps = timesteps[:, -self.max_length :]

            # pad all tokens to sequence length
            attention_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            attention_mask = attention_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            states = torch.cat(
                [
                    torch.zeros(
                        (
                            states.shape[0],
                            self.max_length - states.shape[1],
                            self.state_dim,
                        ),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=states.dtype)
            actions = torch.cat(
                [
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=actions.dtype)
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (
                            returns_to_go.shape[0],
                            self.max_length - returns_to_go.shape[1],
                            1,
                        ),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
                ],
                dim=1,
            ).to(dtype=returns_to_go.dtype)
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds, __ = self.forward(
            states,
            actions,
            None,
            returns_to_go,
            timesteps,
            attention_mask=attention_mask,
            **kwargs,
        )

        return action_preds[0, -1]