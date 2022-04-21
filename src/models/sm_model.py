import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class Model(nn.Module):
    def __init__(self, model_name, labels, loss_type=1, checkpoint=None, debug=False):
        super(Model, self).__init__()
        self.label_num = labels
        self.debug = debug
        self.config = AutoConfig.from_pretrained(model_name)
        self.loss_type = loss_type

        if "deberta" in model_name.lower() or "funnel" in model_name.lower():
            self.pooler = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.Tanh()
            )
        self.classifier = nn.Linear(self.config.hidden_size, labels)
        self.kw_con_classifier = nn.Linear(self.config.hidden_size, 1)
        if "funnel" in model_name.lower():
            self.dropout = nn.Dropout(self.config.hidden_dropout)
        else:
            self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.apply(self._init_weights)

        self.encoder = AutoModel.from_pretrained(model_name)

        if checkpoint is not None:
            cp = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            self.load_state_dict(cp, strict=True)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def apply(self, fn):
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def forward(self, input_ids, token_type_ids, attention_mask, labels,
                keyword_mask, context_mask, special_mask):

        if not self.training and not self.debug:
            output_all = self.encoder(input_ids, attention_mask, token_type_ids, return_dict=True)
            if "pooler_output" in output_all.keys():
                logits_all = self.classifier(self.dropout(output_all.pooler_output))
            else:
                pooler_out_all = self.pooler(output_all.last_hidden_state[:, 0])
                logits_all = self.classifier(self.dropout(pooler_out_all))
            cls_loss = F.cross_entropy(logits_all.view(-1, self.label_num), labels.view(-1))

            return SequenceClassifierOutput(
                loss=cls_loss,
                logits=logits_all
            )

        # build mask
        kw_mask = keyword_mask + special_mask
        con_mask = context_mask + special_mask

        # encoding
        output_all = self.encoder(input_ids, attention_mask, token_type_ids, return_dict=True)
        output_kw = self.encoder(input_ids, kw_mask, token_type_ids, return_dict=True)
        output_con = self.encoder(input_ids, con_mask, token_type_ids, return_dict=True)

        # cls logits
        if "pooler_output" in output_all.keys():
            logits_all = self.classifier(self.dropout(output_all.pooler_output))
            logits_kw = self.classifier(self.dropout(output_kw.pooler_output))
            logits_con = self.classifier(self.dropout(output_con.pooler_output))
        else:
            pooler_out_all = self.pooler(output_all.last_hidden_state[:, 0])
            pooler_out_kw = self.pooler(output_kw.last_hidden_state[:, 0])
            pooler_out_con = self.pooler(output_con.last_hidden_state[:, 0])

            logits_all = self.classifier(self.dropout(pooler_out_all))
            logits_kw = self.classifier(self.dropout(pooler_out_kw))
            logits_con = self.classifier(self.dropout(pooler_out_con))

        # get mean pooling states
        all_kw = (output_all.last_hidden_state * kw_mask.unsqueeze(-1).float())\
            .sum(1).div(kw_mask.float().sum(-1).unsqueeze_(-1))
        all_con = (output_all.last_hidden_state * con_mask.unsqueeze(-1).float())\
            .sum(1).div(con_mask.float().sum(-1).unsqueeze_(-1))
        sep_kw = (output_kw.last_hidden_state * kw_mask.unsqueeze(-1).float())\
            .sum(1).div(kw_mask.float().sum(-1).unsqueeze_(-1))
        sep_con = (output_con.last_hidden_state * con_mask.unsqueeze(-1).float())\
            .sum(1).div(con_mask.float().sum(-1).unsqueeze_(-1))

        # kw_con mean pooling logits
        kw_con_logits = self.kw_con_classifier(
            self.dropout(torch.cat([all_kw, sep_kw, all_con, sep_con], 0))
        )

        # kw_con labels
        kw_con_labels = torch.cat([labels.new_ones(all_kw.size(0) * 2),
                                   labels.new_zeros(all_con.size(0) * 2)], 0).float()

        # joint probability distribution
        prob_all = F.log_softmax(logits_all, -1).view(-1, self.label_num)
        prob_kw = F.log_softmax(logits_kw, -1).view(-1, self.label_num)
        prob_con = F.log_softmax(logits_con, -1).view(-1, self.label_num)
        prob_joint = prob_kw.unsqueeze(-1).expand(-1, self.label_num, self.label_num) + \
            prob_con.unsqueeze(-2).expand(-1, self.label_num, self.label_num)
        prob_joint_list = []
        for idx in range(self.label_num):
            prob_dim = prob_joint[:, idx:, idx:].exp().sum((1, 2)) - prob_joint[:, idx+1:, idx+1:].exp().sum((1, 2))
            prob_joint_list.append(prob_dim.unsqueeze(-1))
        prob_kw_con = (torch.cat(prob_joint_list, -1)+1e-20).log()

        cls_loss = F.cross_entropy(logits_all.view(-1, self.label_num), labels.view(-1))
        kw_con_loss = F.binary_cross_entropy_with_logits(kw_con_logits.view(-1), kw_con_labels.view(-1))
        # kl_loss = F.kl_div(prob_kw_con, prob_all, reduction='batchmean', log_target=True)
        kl_loss = 0.5 * (
                         F.kl_div(prob_kw_con, prob_all, reduction='batchmean', log_target=True) +
                         F.kl_div(prob_all, prob_kw_con, reduction='batchmean', log_target=True)
                        )
        if self.loss_type == 1:
            loss = cls_loss + kw_con_loss + kl_loss
        elif self.loss_type == 2:
            loss = cls_loss + kl_loss
        else:
            loss = cls_loss + kw_con_loss

        if self.training:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits_all
            )
        else:
            return {
                "loss": loss,
                "logits": logits_all,
                "kw_logits": logits_kw,
                "con_logits": logits_con
            }
