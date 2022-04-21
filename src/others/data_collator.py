from transformers.data.data_collator import DataCollatorWithPadding
from typing import Any, Dict, List
import torch


class DataCollator(DataCollatorWithPadding):

    def __init__(self, args, tokenizer, padding=True):
        super(DataCollator, self).__init__(tokenizer, padding)
        self.args = args
        self.pad_id = 0

    def _pad(self, data, width=-1, dtype=torch.long):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [self.pad_id] * (width - len(d)) for d in data]
        return torch.tensor(rtn_data, dtype=dtype)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        """
        features:
            input_ids, token_type_ids, attention_mask, labels,
            keyword_mask, context_mask, special_mask,
            origin_str, keywords
        """
        batch = {}

        # process entity-masked sentence pairs
        features_new = list(map(lambda x: {"input_ids": x['input_ids'],
                                           "token_type_ids": x['token_type_ids'],
                                           "labels": x['labels'] if x.get('labels', 'no') != 'no' else x['label']}, features))

        batch = self.tokenizer.pad(
            features_new,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch['attention_mask'] = self._pad([x['attention_mask'] for x in features])
        if "keyword_mask" in features[0].keys():
            batch['keyword_mask'] = self._pad([x['keyword_mask'] for x in features])
        else:
            batch['keyword_mask'] = []
        if "context_mask" in features[0].keys():
            batch['context_mask'] = self._pad([x['context_mask'] for x in features])
        else:
            batch['context_mask'] = []
        if "special_mask" in features[0].keys():
            batch['special_mask'] = self._pad([x['special_mask'] for x in features])
        else:
            batch['special_mask'] = []

        return batch
