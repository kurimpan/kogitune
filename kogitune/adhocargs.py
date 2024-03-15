from typing import List, Optional
import os
import sys
import json
import re
import inspect

def parse_argument_value(value):
    try:
        return int(value)
    except:
        pass
    try:
        return float(value)
    except:
        pass
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    return value

_key_pattern = re.compile(r'^[A-Za-z0-9\.\-_]+\=')

def _parse_key_value(key, next_value, args):
    if _key_pattern.match(key):
        if key.startswith('--'):
            key = key[2:]
        key, _, value = key.partition('=')
        return key, parse_argument_value(value)     
    elif key.startswith('--'):
        key = key[2:]
        if next_value.startswith('--'):
            if key.startswith('enable_') or key.startswith('enable-'):
                return key[7:], True
            elif key.startswith('disable_') or key.startswith('disable-'):
                return key[8:], False
            return key, True
        else:
            args['_'] = next_value
            return key, parse_argument_value(next_value)
    else:
        if args.get('_') != key:
            files = args.get('files', [])
            files.append(key)
            args['files'] = files
    return key, None

def format_unit(num: int, scale=1000)->str:
    """
    大きな数をSI単位系に変換して返す
    """
    if scale == 1024:
        if num < scale:
            return str(num)
        elif num < scale**2:
            return f"{num / scale:.2f}K"
        elif num < scale**3:
            return f"{num / scale**2:.2f}M"
        elif num < scale**4:
            return f"{num / scale**3:.2f}G"
        elif num < scale**5:
            return f"{num / scale**4:.2f}T"
        elif num < scale**6:
            return f"{num / scale**5:.2f}P"
        else:
            return f"{num / scale**6:.2f}Exa"
    elif scale == 60:
        if num < 1.0:
            return f"{num * 1000:.2f}ms"
        if num < scale:
            return f"{num:.2f}sec"
        elif num < scale**2:
            return f"{num / scale:.2f}min"
        elif num < (scale**2)*24:
            return f"{num /(scale**2):.2f}h"
        else:
            num2 = num % (scale**2)*24
            return f"{num//(scale**2)*24}d {num2/(scale**2):.1f}h"
    else:
        if num < 1_000:
            return str(num)
        elif num < 1_000_000:
            return f"{num / 1_000:.2}K"
        elif num < 1_000_000_000:
            return f"{num / 1_000_000:.2f}M"
        elif num < 1_000_000_000_000:
            return f"{num / 1_000_000_000:.2f}B"
        else:
            return f"{num / 1_000_000_000_000:.2f}T"

## ファイル

def get_basename_from_filepath(filepath:str)->str:
    """
    ファイルパスからベースの名前を取り出す
    """
    filebase = filepath
    if '/' in filebase:
        _, _, filebase = filebase.rpartition('/')
    if '\\' in filebase:
        _, _, filebase = filebase.rpartition('\\')
    if '_L' in filebase:
        left, _, right = filebase.rpartition('_L')
        if right[0].isdigit():
            filebase = left
    filebase, _, _ = filebase.partition('.')
    return filebase

## コンフィグファイル

def load_yaml(config_file):
    import yaml
    loaded_data = {}
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        for section, settings in config.items():
            if isinstance(settings, dict):
                for key, value in settings.items():
                    loaded_data[key] = value
        return loaded_data

def load_json(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)

def load_config(config_file):
    if config_file.endswith('.json'):
        return load_json(config_file)
    if config_file.endswith('.yaml'):
        return load_yaml(config_file)
    return {}

# main adhoc arguments

main_aargs = None

def main_adhoc_arguments():
    global main_aargs
    if main_aargs is None:
        main_aargs = AdhocArguments({})
    return main_aargs

def verbose_print(*args, **kwargs):
    aargs = main_adhoc_arguments()
    aargs.verbose_print(*args, **kwargs)



# main



class AdhocArguments(object):
    """
    アドホックな引数パラメータ
    """
    def __init__(self, 
                 args:dict, 
                 parent=None,
                 expand_config=None, 
                 use_environ=True,
                 face='🦊'):
        self._args = {}
        self._used_keys = set()
        self._use_environ = use_environ
        self.face = face
        self.parent = parent
        if parent:
            self._use_environ = parent._use_environ
            self.face = parent.face
        for key, value in args.items():
            if key == expand_config:
                self.load_config(value)
            else:
                self._args[key] = value

    def __repr__(self):
        if self.parent:
            return f'{self._args}+{self.parent}'
        return repr(self._args)

    def get(self, key, default_value=None):
        keys = key.split('|')
        for key in keys:
            if key in self._args:
                self._used_keys.add(key)
                return self._args[key]
            if key.startswith('='):
                return parse_argument_value(key[1:])
            if key.startswith('!'):
                return self.warn_unset_key(key[0], parse_argument_value(key[1:]))
            if self.parent and key in self.parent :
                return self.parent[key]
            if self._use_environ:
                environ_key = key.upper()
                if environ_key in os.environ:
                    value = parse_argument_value(os.environ[environ_key])
                    self._used_keys.add(key)
                    self._args[key] = value
                    return value
        return default_value

    def __getitem__(self, key):
        return self.get(key, None)

    def __setitem__(self, key, value):
        self._args[key] = value
        # setattr(self, key, value)
        self._used_keys.add(key)

    def __contains__(self, key):
        return key in self._args or (self._use_environ and key.upper() in os.environ)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            for key, value in self._args.items():
                if key not in self._used_keys:
                    raise TypeError(f'{key} is an unused keyword')


    @classmethod
    def from_main(cls, **kwargs):
        aargs = main_adhoc_arguments()
        aargs = AdhocArguments({}, parent=aargs)
        for key, value in kwargs.items():
            aargs._args[key] = value
        return aargs

    def from_kwargs(self, **kwargs):
        aargs = AdhocArguments({}, parent=self)
        for key, value in kwargs.items():
            aargs._args[key] = value
        return aargs


    def update(self, otherdict:dict, overwrite=True, used=True):
        for key, value in otherdict.items():
            if overwrite or key not in self._args:
                self._args[key] = value
                if used:
                    self._used_keys.add(key)

    def load_config(self, config_file, merge=True, overwrite=True):
        loaded_data = load_config(config_file)
        if merge:
            self.update(loaded_data, overwrite=overwrite)
        return loaded_data

    def subset(self, keys='', prefix=None):
        subargs = {}
        keys = set(keys.split('|'))
        for key, value in self._args.items():
            if key in keys:
                self._used_keys.add(key)
                subargs[key] = value
            elif prefix and key.startswith(prefix):
                self._used_keys.add(key)
                key = key[len(prefix):]
                if key.startswith('_'):
                    key = key[1:]
                subargs[key] = value
        return subargs

    def get_subargs(self, keys, exclude=None):
        subargs = {}
        for key in keys.split('|'):
            if key.endswith('*'):
                prefix = key[:-1]
                for key, value in self._args.items():
                    if key.startswith(prefix):
                        self._used_keys.add(key)
                        key = key[len(prefix):]
                        subargs[key] = value
            elif key in self:
                subargs[key] = self[key]
        if exclude is not None:
            for key in exclude.split('|'):
                if key in subargs:
                    del subargs[key]
        return subargs

    def find_options(self, prefix: str, namespace: dict = None):
        if namespace is None:
            # 呼び出し元のフレームを取得
            caller_frame = inspect.stack()[1].frame
            # 呼び出し元のグローバル変数の名前空間を取得
            namespace = caller_frame.f_globals
        return [s.replace(f'{prefix}_', '') for s in globals() if s.startswith(f'{prefix}_')]

    def find_function(self, option:str, prefix: str, namespace: dict = None):
        if namespace is None:
            # 呼び出し元のフレームを取得
            caller_frame = inspect.stack()[1].frame
            # 呼び出し元のグローバル変数の名前空間を取得
            namespace = caller_frame.f_globals
        func = namespace.get(f'{prefix}_{option}')
        if func is None:
            patterns = [s.replace(f'{prefix}_', '') for s in globals() if s.startswith(f'{prefix}_')]
            raise ValueError(f'{prefix}_{option} is not found. Select pattern from {patterns}')
        return func

    def check_unused(self):
        show_notion = True
        for key, value in self._args.items():
            if key not in self._used_keys:
                if show_notion:
                    self.print(f'未使用のパラメータ一覧//List of unused parameters')
                    show_notion = False
                print(f'{key}: {repr(value)}')
        if not show_notion:
            self.print(f'スペルミスがないか確認してください//Check if typos exist.')

    def save_as_json(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory) and directory != '':
            os.makedirs(directory)

        with open(file_path, 'w', encoding='utf-8') as w:
            print(json.dumps(self._args, ensure_ascii=False, indent=4), file=w)

    def raise_files(self, msg='ファイルの指定が一つ以上必要です。'):
        self.print(msg)
        sys.exit(1)

    def raise_uninstalled_module(self, module_name):
        self.print(f'{module_name}がインストールされていません//Uninstalled {module_name}')
        print(f'pip3 install -U {module_name}')
        sys.exit(1)

    def raise_unset_key(self, key, desc_ja=None, desc_en=None):
        desc_ja = f' ({desc_ja})' if desc_ja else ''
        desc_en = f' ({desc_en})' if desc_en else ''
        self.print(f'{key}{desc_ja}を設定してください//Please set {key}{desc_en}')
        sys.exit(1)

    def warn_unset_key(self, key, value):
        self.print(f'{key}を忘れずに設定してください。とりあえず{value}にしてます。//Please set {key}')
        return value

    def print(self, *args, **kwargs):
        print(self.face, *args, **kwargs)

    def verbose_print(self, *args, **kwargs):
        print(self.face, *args, **kwargs)

    @classmethod
    def to_adhoc(cls, aargs: dict=None, args=None, **kwargs):
        aargs = aargs or args
        if not isinstance(aargs, AdhocArguments):
            if aargs is None:
                aargs = AdhocArguments({})
            elif isinstance(aargs, dict):
                aargs = AdhocArguments(aargs)
        # args = {k:v for k,v in vars(hparams).items() if v is not None}
        aargs.update(dict(kwargs))
        return aargs


def adhoc_parse_arguments(subcommands:Optional[List[str]]=None,
                          requires:Optional[List[str]]=None,
                          use_environ=True, expand_config=None)->AdhocArguments:
    global main_aargs
    if subcommands is not None:
        if isinstance(subcommands,str):
            subcommands=subcommands.split('|')
        if len(sys.argv) == 1 or sys.argv[1] not in subcommands:
            print(f'{sys.argv[0]} requires subcommands: {subcommands}')
            sys.exit(0)
        argv = sys.argv[2:]
        args={'_': '', 'subcommand': sys.argv[1]}
    else:
        argv = sys.argv[1:]
        args={'_': ''}

    for arg, next_value in zip(argv, argv[1:] + ['--']):
        key, value = _parse_key_value(arg, next_value, args)
        if value is not None:
            args[key.replace('-', '_')] = value
    del args['_']

    aargs = AdhocArguments(args, 
                          parent=None,
                          expand_config=expand_config, 
                          use_environ=use_environ)

    if requires:
        if isinstance(requires, str):
            requires = requires.split('|')
        lost_found = False
        for key in requires:
            if key not in aargs:
                aargs.print(f'Option {key} is required.')
                lost_found = True
        if lost_found:
            sys.exit(1)
    main_aargs = aargs
    return aargs

###
###


DEFAULT_TOKENIZER = 'llm-jp/llm-jp-1.3b-v1.0'

def load_tokenizer(tokenizer:str = None, **kwargs):
    from transformers import AutoTokenizer
    with AdhocArguments.from_main(**kwargs) as aargs:
        tokenizer = tokenizer or aargs[f'tokenizer_path|tokenizer|={DEFAULT_TOKENIZER}']
        if isinstance(tokenizer, str):
            local_args = aargs.get_subargs('tokenizer_*|trust_remote_code', exclude='tokenizer_path')
            if 'trust_remote_code' not in local_args:
                local_args['trust_remote_code'] = True
            if 'use_fast' not in local_args:
                local_args['use_fast'] = False
            # AutoTokenizer.from_pretrained(tokenizer, legacy=legacy, trust_remote_code=True, use_fast=False)
            return AutoTokenizer.from_pretrained(tokenizer, **local_args)
        return tokenizer



if __name__ == '__main__':
    aargs = adhoc_parse_arguments()
    with AdhocArguments.from_main(a=False,b=1) as aargs:
        print(aargs)
        print(aargs['a'])