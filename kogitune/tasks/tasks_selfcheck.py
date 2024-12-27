from ..loads.commons import *
from .tasks_textgen import TextGeneration

@adhoc.reg('selfcheck')
class SelfCheckGPT(TextGeneration):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'selfcheck'
        
    def apply_template(self, sample: dict, template: dict):
        sample["_input"] = self.format(template, "prompt", sample)

    def calc(self, metric, samples: List[dict]):
        # 必ず_extractedを使用してメトリクスを計算
        candidates = self.column_values(samples, "_extracted")
        list_samples = self.column_values(samples, "_samples")
        adhoc.verbose_print("Using pre-extracted values", candidates[0], list_samples[0], once='extracted')
        results = metric.calc(candidates, list_samples)
        self.update_values(samples, results)
        return results
