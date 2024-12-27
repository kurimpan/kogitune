from ..loads.commons import *
from ..loads import Model, Metric, LeaderBoard
from .templates import guess_template

## base class

TASK_MAP = {}

class TaskLoader(adhoc.AdhocLoader):

    def load_modules(self, path, kwargs):
        from .tasks_textgen import TextGeneration
        from .tasks_selfcheck import SelfCheckGPT
        from .tasks_code import CodeEval
        from .tasks_choice import QAChoice
        from .tasks_mia import MIA

TaskLoader(TASK_MAP).register("task")

def fmt(template:str, sample:dict):
    return template.format(**sample)

class Task(adhoc.AdhocObject):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.template = adhoc.get(kwargs, 'template_config|template')
        if is_config(self.template):
            self.template = load_config(self.template)
        self.shots = adhoc.get(kwargs, 'shots|shot|=0')
        self.heading_messages = adhoc.get(kwargs, 'heading_messages')
        self.extra_prompt = adhoc.get(kwargs, 'extra_prompt|cot_prompt|=')
        if self.extra_prompt != '':
            self.extra_prompt = f'\n{self.extra_prompt}'
        self.init_extractor(**kwargs)
        self.progress_bar = adhoc.progress_bar()
        self.init_kwargs = {**kwargs}
    
    def init_extractor(self, **kwargs):
        self.extractor = self.load('extractor', 'extractor', **kwargs)

    @property
    def tasktag(self):
        name = self.name if hasattr(self, "name") else self.path
        return self.tag if self.tag != '' else name

    def start_progress_bar(self, total:int, desc:str=None):
        self.progress_bar = adhoc.progress_bar(total=total, desc=desc)

    def end_progress_bar(self):
        if self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = adhoc.progress_bar()

    def update_from_template(self, samples: List[dict]):
        if self.template is not None:
            template = self.template
        else:
            template = self.guess_template(samples[0])
            if template:
                adhoc.verbose_print('[テンプレート]', dump=template, once=True,
                                    if_dislike={'template_config': repr("(template.json)")})
                self.template = template
        if template:
            if self.shots > 0 and 'shots' in template and self.heading_messages is None:
                self.heading_messages = template['shots']
                self.set_few_shots()
            if self.extractor is None:
                if 'extract_pattern' in template:
                    self.extractor = adhoc.load('pattern', template['extract_pattern'])
        return template

    def prepare(self, samples: List[dict]):
        template = self.update_from_template(samples)
        for sample in samples:
            if template:
                try:
                    self.apply_template(sample, template)
                except KeyError as e:
                    adhoc.print("テンプレートがデータにマッチしません。")
                    adhoc.print("テンプレート", adhoc.dump(template), face='')
                    adhoc.print("データ", adhoc.dump(sample), face='')
                    adhoc.exit(throw=e)
            else:
                self.transform(sample)

    def guess_template(self, sample):
        return guess_template(sample)
    
    def set_few_shots(self):
        if self.heading_messages and self.shots > 0:
            if self.shots *2 != len(self.heading_messages):
                self.heading_messages = self.heading_messages[:self.shots*2]
                shots = len(self.heading_messages)//2
                if self.shots == shots:
                    return
                # 再調整
                self.shots = shots
                self.name = f'{self.shots}-shot'
                adhoc.verbose_print(f'{self.shots}-shot//ショット', dump=self.heading_messages)


    def apply_template(self, sample:dict, template:dict):
        pass

    def format(self, template, key, sample):
        template_format = template[key]
        return template_format.format(**sample)

    def transform(self, sample):
        adhoc.print(f"{self.name}タスク用のテンプレートがありません。")
        adhoc.print(adhoc.dump(sample), face='')
        adhoc.exit(throw=ValueError(f"{self.name}タスク用のテンプレートがありません。"))


    def eval(self, model, samples: List[dict]):
        pass

    def calc(self, metric: Metric, samples: List[dict]):
        candidates = self.column_values(samples, "_output")
        references = self.column_values(samples, "_reference")
        results = metric.calc(candidates, references)
        self.update_values(samples, results)
        return results

    @property
    def default_metrics(self):
        return []

    def column_values(self, samples: List[dict], key:str):
        return [sample[key] for sample in samples]

    def update_kwargs(self, samples:List[dict], /, **kwargs):
        items = list(kwargs.items())
        for sample in samples:
            for k, v in items:
                sample[k] = v

    def update_values(self, samples: List[dict], results: dict):
        """
        results = {"output": scores}
        """
        for key, outputs in results.items():
            if isinstance(outputs, tuple):
                # ('mean', scores) 形式への対応
                outputs = outputs[1]
            assert len(outputs) >= len(samples), f"{len(outputs)} == {len(samples)}: {outputs}"
            for i, sample in enumerate(samples):
                sample[key] = outputs[i]

    def verbose_samples(self, samples: List[dict]):
        with self.verbose as verbose:
            for sample in samples:
                verbose.print_sample(sample)

    @classmethod
    def register(cls, schemes):
        global TASK_MAP
        for scheme in schemes.split("|"):
            TASK_MAP[scheme] = cls


def task_eval(model_list: List[str], metrics:List[str], /, **kwargs):
    # タスクをロードしておきます。
    from .tasks_textgen import TextGeneration
    from .tasks_choice import QAChoice
    from ..loads import Metric

    board:LeaderBoard = adhoc.load('from_kwargs', 'leaderboard', **kwargs)
    save_step = adhoc.get(kwargs, "save_steps|save_step")
    if save_step:
        kwargs["_resume"] = True # 前の続きから実行する (ただし、resumeが優先)
    task:Task = adhoc.load('task', adhoc.get(kwargs, 'task|='))

    for model_path in model_list:
        model : Model = adhoc.load("model", model_path, _lazy = True, **kwargs)
        for taskdata in list_testdata(model.modeltag, task.tasktag, **kwargs):
            # 未処理のサンプルのみ対象にする
            samples = [sample for sample in taskdata.samples() 
                       if sample.get("_model") != model.modeltag]
            if len(samples) > 0:   
                head = adhoc.get(kwargs, "test_run|head")
                if head:
                    samples = samples[:head]
                
                # save_stepごとに保存する
                save_step = adhoc.get(kwargs, "save_steps|save_step|=4")
                with VerboseCounter(**kwargs) as verbose:
                    task.start_progress_bar(total=len(samples), desc=model.modeltag)
                    for start in range(0, len(samples), save_step):
                        splited_samples = samples[start:start+save_step]
                        task.prepare(splited_samples)
                        task.eval(model, splited_samples)
                        taskdata.save()
                        verbose.print_sample(splited_samples)
                    task.end_progress_bar()
            
            if len(metrics) == 0:
                metrics = task.default_metrics

            # 処理済みのサンプルのみ扱う
            samples = [sample for sample in taskdata.samples() if sample.get("_model") == model.modeltag]
            if len(samples) == 0:
                adhoc.verbose_print("ひとつも生成されてないね")
                break
            task.update_from_template(samples) # extractor の再読み込み
            for metric_path in listfy(metrics):
                if metric_path == "none": 
                    break
                metric = adhoc.load("metric", metric_path, **kwargs)
                if metric.check(samples):
                    results = task.calc(metric, samples)
                    board.pivot_table(samples, results, **kwargs)
                taskdata.save()
        model = None
    board.show()

def list_testdata(modeltag, tasktag, /, **kwargs):
    dataset_list = adhoc.get_list(kwargs, "dataset_list|dataset|!!")
    kwargs['_modeltag'] = modeltag
    kwargs['_tasktag'] = tasktag
    for path in dataset_list:
        if 'dataset_subset' in kwargs:
            path, largs, _ = adhoc.parse_path(path, parent_args=kwargs)
            if kwargs['dataset_subset'] == '*':
                from ..loads.datasets import load_dataset_names
                kwargs['dataset_subset'] = load_dataset_names(path)
            if largs.get('name') != '*':
                for subset_name in adhoc.get_list(kwargs, f"dataset_subset"):
                    taskdata = load_taskdata(path, **(kwargs|{'_name': subset_name}))
                    yield taskdata
                continue
        taskdata = load_taskdata(path, **kwargs)
        yield taskdata

def load_taskdata(path, /, **kwargs):
    from ..loads import RecordData, Transform

    stream = adhoc.load("datastream", path, **kwargs)
    samples = [sample for sample in stream.samples()]
    if path.endswith('.jsonl'):
        sample = samples[0]
        if '_dataset' in sample and '_model' in sample and '_task' in sample:
            record = RecordData(path, samples, **kwargs)
            record.tags = (sample['_model'], sample['_dataset'], sample['_task'])
            return record

    transform: Transform = adhoc.load('from_kwargs', 'Transform', **kwargs)
    if not transform.isNullObject():
        samples = transform.transform(samples)
        adhoc.verbose_print("Transform//簡易的な変形:", adhoc.dump(samples[0]))

    modeltag = kwargs.get('_modeltag', '(model)')
    datatag = stream.datatag
    task = adhoc.get(kwargs, "_tasktag|task|=")
    samples = [(sample | {"_dataset": datatag}) for sample in samples]

    # save_path の決定
    if task == "":
        save_path = f"{modeltag}/{datatag}_x_{modeltag}.jsonl"
    else:
        save_path = f"{modeltag}/{datatag}_{task}_x_{modeltag}.jsonl"
    output_path = adhoc.get(kwargs, "output_path|output_dir")
    if output_path:
        save_path = os.path.join(output_path, save_path)

    record = RecordData(save_path, samples, **kwargs)
    record.tags = (modeltag, datatag, task)
    record.save()
    adhoc.saved(save_path, f'Results//実験結果 {(modeltag, datatag, task)}')
    return record

def calc_leaderboard(files: Union[List[str], str], names:Union[List[str], str] = None, /, **kwargs):
    board: LeaderBoard = adhoc.load('from_kwargs', 'leaderboard', **kwargs)

    for path in listfy(files):
        record = load_taskfile(path, **kwargs)
        samples = record.samples()
        if name is None:
            adhoc.print(f"`names='{example_names(samples)}'`のように指定してください", adhoc.dump([0]))
            return
        for name in listfy(names):
            if ':' in name:
                name, _, aggfunc = name.partition(':')
            else:
                aggfunc = 'mean'      
            board.pivot_table(record.samples(), name, aggfunc, **kwargs)
    board.show()

def load_taskfile(path, /, **kwargs):
    from ..loads import RecordData

    stream = adhoc.load("datastream", path, **kwargs)
    samples = [sample for sample in stream.samples()]
    sample = samples[0]
    if '_dataset' in sample and '_model' in sample and '_task' in sample:
        record = RecordData(path, samples, **kwargs)
        record.tags = (sample['_model'], sample['_dataset'], sample['_task'])
        return record
    adhoc.exit(throw=IOError(f'評価タスクの保存データではありません。{path}'))

def example_names(samples):
    aggfuncs = []
    for key, value in samples[0].items():
        if isinstance(value, float):
            aggfuncs.append(f'{key}:mean')
    return ':'.join(aggfuncs)
