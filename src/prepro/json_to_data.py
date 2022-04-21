import datasets
import glob
from os.path import join as pjoin

from others.logging import logger
from transformers import AutoTokenizer


class Processor():
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.num_class = args.num_class

    def handle_special_token(self, token_list):
        return [item.replace("Ġ", "") for item in token_list]

    def process(self, example):

        encoded_dict = {}

        encoded_origin_dict = self.tokenizer(example['origin_a'], example['origin_b'],
                                             truncation=True,
                                             max_length=self.args.max_length,
                                             return_token_type_ids=True)

        origin_a_b = self.tokenizer.convert_ids_to_tokens(encoded_origin_dict['input_ids'])

        if "chinese" not in self.args.tokenizer:
            origin_a_b = self.handle_special_token(origin_a_b)

        sorted_keywords = sorted(example['keyword_a'] + example['keyword_b'],
                                 key=lambda x: len(x['entity']),
                                 reverse=True)

        replaced_a = example['origin_a']
        replaced_b = example['origin_b']
        keyword_mask = [0] * len(origin_a_b)
        special_mask = [1 if token in (self.tokenizer.cls_token,
                                       self.tokenizer.sep_token) else 0
                        for token in origin_a_b]
        context_mask = [0 if token in (self.tokenizer.cls_token,
                                       self.tokenizer.sep_token) else 1
                        for token in origin_a_b]

        keywords = []
        for item in sorted_keywords:

            if not (item['entity'] in replaced_a or item['entity'] in replaced_b):
                continue

            encoded_kw = self.tokenizer.tokenize(item['entity'])
            if "chinese" not in self.args.tokenizer:
                encoded_kw = self.handle_special_token(encoded_kw)
            for idx in range(len(origin_a_b)):
                if origin_a_b[idx] == encoded_kw[0] and \
                   origin_a_b[idx: idx+len(encoded_kw)] == encoded_kw:
                    keyword_mask[idx: idx+len(encoded_kw)] = [1] * len(encoded_kw)
                    context_mask[idx: idx+len(encoded_kw)] = [0] * len(encoded_kw)
            keywords.append(item['entity'])
            replaced_a = replaced_a.replace(item['entity'], '#')
            replaced_b = replaced_b.replace(item['entity'], '#')

        if 'chinese' in self.args.tokenizer:
            origin_a_b = ''.join(origin_a_b)
        else:
            origin_a_b = ' '.join(origin_a_b)

        encoded_dict['input_ids'] = encoded_origin_dict['input_ids']
        encoded_dict['attention_mask'] = encoded_origin_dict['attention_mask']
        encoded_dict['token_type_ids'] = encoded_origin_dict['token_type_ids']
        encoded_dict['keyword_mask'] = keyword_mask
        encoded_dict['context_mask'] = context_mask
        encoded_dict['special_mask'] = special_mask
        encoded_dict['origin_str'] = origin_a_b
        encoded_dict['keywords'] = keywords

        # mask assertion
        for idx in range(len(encoded_dict['input_ids'])):
            assert encoded_dict['keyword_mask'][idx] + \
                   encoded_dict['context_mask'][idx] + \
                   encoded_dict['special_mask'][idx] == \
                   encoded_dict['attention_mask'][idx]

        return encoded_dict


def format_json_to_data(args):

    train_lst = []
    dev_lst = []
    test_lst = []
    for json_f in glob.glob(pjoin(args.raw_path, '*.json')):
        real_name = json_f.split('/')[-1]
        corpus_type = real_name.split('.')[-2]
        if corpus_type == 'train':
            train_lst.append(json_f)
        elif corpus_type == 'test':
            test_lst.append(json_f)
        else:
            dev_lst.append(json_f)

    dataset = datasets.load_dataset(
        'json',
        data_files={'train': train_lst,
                    'validation': dev_lst if len(dev_lst) > 0 else test_lst,
                    'test': test_lst}
    )

    processor = Processor(args)
    encoded_dataset = dataset.map(
        processor.process,
        load_from_cache_file=False,
        num_proc=8
    )
    for corpus_type in ['train', 'validation', 'test']:
        total_statistic = {
            "instances": 0,
            "exceed_length_num": 0,
            "total_length": 0.,
            "src_length_dist": [0] * 11,
        }
        for item in encoded_dataset[corpus_type]:
            total_statistic['instances'] += 1
            if len(item['input_ids']) > args.max_length:
                total_statistic['exceed_length_num'] += 1
            total_statistic['total_length'] += len(item['origin_a']) + len(item['origin_b'])
            total_statistic['src_length_dist'][min(len(item['origin_a']) // 30, 10)] += 1
            total_statistic['src_length_dist'][min(len(item['origin_b']) // 30, 10)] += 1

        dataset[corpus_type]
        if total_statistic["instances"] > 0:
            logger.info("Total %s examples: %d" %
                        (corpus_type, total_statistic["instances"]))
            logger.info("Number of samples that exceed maximum length: %d" %
                        total_statistic["exceed_length_num"])
            logger.info("Average length of src sentence: %f" %
                        (total_statistic["total_length"] / (2. * total_statistic["instances"])))
            for idx, num in enumerate(total_statistic["src_length_dist"]):
                logger.info("token num %d ~ %d: %d, %.2f%%" %
                            (idx * 30, (idx+1) * 30, num, (num / (2. * total_statistic["instances"]))))

    encoded_dataset.save_to_disk(args.save_path + '.save')
