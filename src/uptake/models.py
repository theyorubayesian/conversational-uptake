from typing import Dict

import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from transformers import PretrainedConfig


class MultiHeadModel(BertPreTrainedModel):
  """Pre-trained BERT model that uses our loss functions"""

  def __init__(self, config: PretrainedConfig, head2size: Dict[str, int]):
    super(MultiHeadModel, self).__init__(config, head2size)

    config.num_labels = 1
    self.bert = BertModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    module_dict = {}

    for head_name, num_labels in head2size.items():
      module_dict[head_name] = nn.Linear(config.hidden_size, num_labels)
    
    self.heads = nn.ModuleDict(module_dict)

    self.init_weights()

  def forward(self, input_ids, token_type_ids=None, attention_mask=None,
              head2labels=None, return_pooler_output=False, head2mask=None,
              nsp_loss_weights=None) -> Dict[str, torch.Tensor]:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get logits
    output = self.bert(
      input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
      output_attentions=False, output_hidden_states=False, return_dict=True)
    pooled_output = self.dropout(output["pooler_output"]).to(device)

    head2logits = {}
    return_dict = {}
    for head_name, head in self.heads.items():
      head2logits[head_name] = self.heads[head_name](pooled_output)
      head2logits[head_name] = head2logits[head_name].float()
      return_dict[head_name + "_logits"] = head2logits[head_name]


    if head2labels is not None:
      for head_name, labels in head2labels.items():
        num_classes = head2logits[head_name].shape[1]

        # Regression (e.g. for politeness)
        if num_classes == 1:

          # Only consider positive examples
          if head2mask is not None and head_name in head2mask:
            num_positives = head2labels[head2mask[head_name]].sum()  # use certain labels as mask
            if num_positives == 0:
              return_dict[head_name + "_loss"] = torch.tensor([0]).to(device)
            else:
              loss_fct = MSELoss(reduction='none')
              loss = loss_fct(head2logits[head_name].view(-1), labels.float().view(-1))
              return_dict[head_name + "_loss"] = loss.dot(head2labels[head2mask[head_name]].float().view(-1)) / num_positives
          else:
            loss_fct = MSELoss()
            return_dict[head_name + "_loss"] = loss_fct(head2logits[head_name].view(-1), labels.float().view(-1))
        else:
          loss_fct = CrossEntropyLoss(weight=nsp_loss_weights.float())
          return_dict[head_name + "_loss"] = loss_fct(head2logits[head_name], labels.view(-1))

    if return_pooler_output:
      return_dict["pooler_output"] = output["pooler_output"]

    return return_dict
