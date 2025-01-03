from ..loads.commons import *
from .tasks_textgen import TextGeneration

@adhoc.reg('selfcheck')
class SelfCheckGPT(TextGeneration):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'selfcheck'
        
    def apply_template(self, sample:dict, template:dict):
        sample["_input"] = self.format(template, "prompt", sample)

    def eval(self, model, samples: List[dict]):
        input_texts = self.column_values(samples, "_input")
        gen_args = model.filter_gen_args(**self.init_kwargs)

        output_texts = model.generate(input_texts, _n=1, _do_sample=False, **gen_args)
        self.update_values(samples, {"_output": output_texts})

        n = adhoc.get(gen_args, "num_return_sequences|n|=6")
        if "temperature" not in gen_args:
            adhoc.verbose_print("temperatureを設定してね. temperature=1.0", once="temperature=1.0")
            gen_args["temperature"] = 1.0
        output_texts = model.generate(input_texts, self.progress_bar, _n=n, _do_sample=True, **gen_args)
        self.update_kwargs(samples, _model=model.modeltag, _task=self.name)
        self.update_values(samples, {"_samples": output_texts})

    def calc(self, metric, samples: List[dict]):
        candidates = self.column_values(samples, "_output")
        list_samples = self.column_values(samples, "_samples")
        if self.extractor:
            candidates = [self.extractor.extract(c)[-1] for c in candidates]
            self.update_values(samples, {"_extracted": candidates})
            _list_samples = []
            for _samples in list_samples:
                _samples = [self.extractor.extract(c)[-1] for c in _samples]
                _list_samples.append(_samples)
            list_samples = _list_samples
            adhoc.verbose_print("extracted", candidates[0], list_samples[0], once='extracted')
        else:
            adhoc.verbose_print("extractorが設定されてないね", once='extracted')
        results = metric.calc(candidates, list_samples)
        self.update_values(samples, results)
        return results
