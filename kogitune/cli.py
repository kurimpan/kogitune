import os
from tqdm import tqdm

from .commons import *
from .adhocargs import adhoc_parse_arguments
from .file_utils import parse_url_args, safe_new_file

def main_store(args=None):
    from .stores import split_to_store
    split_to_store(args.files, skip_validation=False, args=args)

def main_head(args):
    from .trainers import DatasetComposer
    with DatasetComposer(args.files) as dc:
        tokenizer = dc.get_tokenizer()
        for i in range(len(dc)):
            example = dc[i]
            if 'input_ids' in example:
                print(f'inputs[{i}]:', tokenizer.decode(example['input_ids']))
            if 'labels' in example:
                print(f'labels[{i}]:', tokenizer.decode(example['labels']))
            print('---')


FREEZE='''
from datasets import load_from_disk
ds = load_from_disk("{}")
'''

def main_freeze(args):
    from tqdm import tqdm
    from datasets import Dataset
    input_ids = []
    attention_mask = []
    labels=[]
    with DataComposer(args.urls, 
                      data_type=args.data_type,
                      max_length=args.max_length,
                      prefetch=0) as dc:
        for i in tqdm(range(len(dc))):
            example=dc[i]
            input_ids.append(example['input_ids'])
            if 'attention_mask' in example:
                attention_mask.append(example['attention_mask'])
            if 'labels' in example:
                labels.append(example['labels'])
    if len(labels) > 0:
        ds_dict = { "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels }
    elif len(attention_mask) > 0:
        ds_dict = { "input_ids": input_ids, "attention_mask": attention_mask}
    else:
        ds_dict = { "input_ids": input_ids}
    ds = Dataset.from_dict(ds_dict).with_format("torch")
    print(ds)
    ds.save_to_disk(args.output_path)
    print(FREEZE.format(args.output_path))

def main_histogram(args):
    import pandas as pd
    from .trainers import DatasetComposer
    url_list = args['files']
    if len(url_list) == 0:
        
    with DatasetComposer(url_list, args=args) as dc:
        tokenizer = dc.get_tokenizer()
        token_ids = list(range(0, tokenizer.vocab_size))
        vocabs = tokenizer.convert_ids_to_tokens(token_ids)
        counts = [0] * tokenizer.vocab_size
        # csv_file = f'{store_path.replace("/", "_")}.csv'

        for i in tqdm(range(len(dc)), desc='counting tokens'):
            example = dc[i]
            for token_id in example['input_ids']:
                counts[token_id] += 1
            if 'labels' in example:
                for token_id in example['labels']:
                    counts[token_id] += 1
    df = pd.DataFrame({'tokens': vocabs, 'counts': counts})
    print(df['counts'].describe())
    output_file = args['output_file|output_path']
    if output_file is None:
        _, _args = parse_url_args(args['files'][0], {})
        _, _, path = _args['url_path'].rpartition('/')
        output_file = safe_new_file(f'histogram_{path}', 'csv')
        verbose_print(f"字句の出現頻度を'{output_file}'に保存しました。")
    df.to_csv(args.output_file)

def conv_txt_to_jsonl(file):
    from .file_utils import zopen, filelines
    import json
    newfile = file.replace('.txt', '.jsonl')
    with zopen(newfile, 'wt') as w:
        for line in filelines(file):
            line = line.replace('<nL>', '\n')
            print(json.dumps({'text': line}, ensure_ascii=False), file=w)
    verbose_print(f'"{newfile}"へ変換しました。')

def main_oldconv(args):
    for file in args.files:
        if file.endswith('.txt') or file.endswith('.txt.zst') or file.endswith('.txt.gz'):
            conv_txt_to_jsonl(file)

def main_linenum(args):
    from file_utils import extract_linenum_from_filename, rename_with_linenum, get_linenum
    for file in args['files']:
        n = extract_linenum_from_filename()
        if n is None:
            n = get_linenum(file)
            file = rename_with_linenum(file, n)


def main_update(args):
    args.verbose_print('pip3 install -U git+https://github.com/kuramitsulab/kogitune.git')
    os.system('pip3 uninstall -y kogitune')
    os.system('pip3 install -U git+https://github.com/kuramitsulab/kogitune.git')


def main():
    # メインのパーサーを作成
    subcommands = args.find_options('main')
    args = adhoc_parse_arguments(subcommands=subcommands)
    main_func = args.find_function(args['subcommand'], prefix='main')
    main_func(args)
    args.check_unused()

if __name__ == '__main__':
    main()
