import numpy as np
import torch
import torch.nn.functional as F

from decision_transformer.training.trainer import Trainer

class SequenceTrainer(Trainer):
    def train_step(self):
        self.optimizer.zero_grad()
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            attention_mask,
        ) = self.get_batch(self.batch_size)
        
        action_target = torch.clone(actions)
        
        if self.args["fp16"]:
            with torch.cuda.amp.autocast():
                _, action_preds, _, _ = self.model.forward(
                    states,
                    actions,
                    rewards,
                    rtg[:, :-1],
                    timesteps,
                    attention_mask=attention_mask,
                )

                loss = self.loss_fn(
                    None,
                    action_preds,
                    None,
                    None,
                    action_target,
                    None,
                )
        else:
            _, action_preds, _, _ = self.model.forward(
                states,
                actions,
                rewards,
                rtg[:, :-1],
                timesteps,
                attention_mask=attention_mask,
            )
            loss = self.loss_fn(
                None,
                action_preds,
                None,
                None,
                action_target,
                None,
            )

        if self.model.use_llm and self.args["co_training"]:
            batch = next(self.train_nlp_dataset)
            lm_out = self.model.transformer_model(**batch)
            lm_loss = lm_out.loss
            loss += self.args["co_lambda"] * lm_loss
        
        if self.args["fp16"]:
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()

        self.step += 1
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[
            attention_mask.reshape(-1) > 0
        ]
        with torch.no_grad():
            self.diagnostics["training/action_error"] = (
                torch.mean((action_preds - action_target) ** 2).detach().cpu().item()
            )

        return loss.detach().cpu().item(), lm_loss.detach().cpu().item() if self.model.use_llm else 0