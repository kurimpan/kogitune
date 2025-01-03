import os
import importlib
from .stack import adhoc_print, verbose_print, cli, get_stacked


def pip(module: str, command='install'):
    cmd = f"pip3 {command} {module}"
    adhoc_print(cmd, color='red')
    os.system(cmd)


def safe_import(module: str, pip_install_modules=None):
    try:
        module = importlib.import_module(module)
    except ModuleNotFoundError as e:
        if get_stacked('auto_import', False):
            raise e
        pip(pip_install_modules or module)
        module = importlib.import_module(module)
    if hasattr(module, '__version__'):
        adhoc_print('Modules//モジュール', module.__name__, module.__version__, once=module.__name__, lazy=True)
    return module

@cli
def update_cli(**kwargs):
    adhoc_print("KOGITUNEを最新の安定版に更新します。")
    os.system("pip3 uninstall -y kogitune")
    pip('git+https://github.com/kuramitsulab/kogitune.git', command='install -U -q')

@cli
def update_beta_cli(**kwargs):
    adhoc_print("KOGITUNEを研究室内ベータ版に更新します。")
    os.system("pip3 uninstall -y kogitune")
    pip('git+https://github.com/kkuramitsu/kogitune.git', command='install -U -q')


def adhoc_tqdm(iterable, desc=None, total=None, /, **kwargs):
    use_tqdm = get_stacked('use_tqdm', True)
    if use_tqdm:# and safe_check(total) > 1:
        tqdm = safe_import('tqdm.auto', 'tqdm')
        return tqdm.tqdm(iterable, desc=desc, total=total)
    else:
        return iterable

class _DummyTqdm:
    def update(self, n=1):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

def safe_check(total):
    if isinstance(total, int):
        return total
    return 10

def adhoc_progress_bar(total=None, desc=None):
    """
    with progress_bar(total=10) as pbar:
        for n in range(10):
            pbar.update()
    """
    if total is None:
        return _DummyTqdm()
    use_tqdm = get_stacked('use_tqdm', True)
    if use_tqdm and safe_check(total) > 1:
        tqdm = safe_import('tqdm.auto', 'tqdm')
        return tqdm.tqdm(desc=desc, total=total)
    else:
        return _DummyTqdm()
