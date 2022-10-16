import abc
from functools import partial

import datasets
import torch
import tqdm
from easy_transformer import EasyTransformer


class TaskHarness(metaclass=abc.ABCMeta):
    """This is an abstract class that should be subclassed to define how
    to measure (possibly ablated) performance of the specific task of
    interest.
    """
    @abc.abstractmethod
    def get_logit_diff(self, logits, key):
        pass

    def get_name(self):
        return self.__class__.__name__


class ModelAblator:
    def __init__(self, model_name='gpt2', tasks=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = EasyTransformer.from_pretrained(model_name)
        self.model = model.to(device)
        torch.set_grad_enabled(False)

        if tasks is None:
            tasks = []
        self.tasks = tasks

        self.example_cache = None
        self.average_activations = None
        
    def cache_activations(self):
        self.model.cfg.use_attn_result = True
        self.example_cache = {}
        for task in self.tasks:
            task_name = task.get_name()
            self.example_cache[task_name] = {}
            for key, sentence in task.get_examples().items():
                self.example_cache[task_name][key] = {}
                self.model.cache_all(self.example_cache[task_name][key],
                                     remove_batch_dim=True)
                _ = self.model(sentence)
                model.reset_hooks()

                for act_name in self.example_cache[task_name][key]:
                    print(
                        act_name,
                        self.example_cache[task_name][key][act_name].shape)
                
        self.model.cfg.use_attn_result = False

    def calculate_mean_activations(
            self,
            dataset_name='stas/openwebtext-10k'):
        dataset = datasets.load_dataset(dataset_name, split='train')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

        def running_total_hook(act, hook):
            if 'running_total' not in hook.ctx:
                hook.ctx['running_total'] = torch.zeros_like(act[0, 0])
                hook.ctx['count'] = 0

            hook.ctx['running_total'] += einops.reduce(
                act, "batch pos ... -> ...", "sum")
            hook.ctx['count'] += act.shape[0] * act.shape[1]

        # Always reset hooks before setting up new ones, unless you
        # explicitly want to keep old ones
        self.model.reset_hooks()
        for name, hook in self.model.hook_dict.items():
            hook_suffix = name.split('.')[-1]
            if hook_suffix in ['hook_v',
                               'hook_attn_out',
                               'hook_mlp_out']:
                # This adds a hook to count a running total
                hook.add_hook(running_total_hook, dir='fwd')

        max_samples = 1000
        count_samples = 0
        for batch in tqdm.tqdm(dataloader):
            tokens = self.model.tokenizer.encode(
                batch['text'][0],
                truncation=True,
                return_tensors='pt').to(device)
            logits = self.model(tokens)
            count_samples += len(tokens)
            if count_samples > max_samples:
                break

        average_acts = {}
        for name, hook in self.model.hook_dict.items():
            hook_suffix = name.split('.')[-1]
            if hook_suffix in ['hook_v', 'hook_attn_out', 'hook_mlp_out']:
                average_acts[hook.name] = hook.ctx['running_total']/hook.ctx['count']
        print('Average act keys:', average_acts.keys())
        self.model.reset_hooks(clear_contexts=True)

        self.average_activations = average_acts
        
    def zero_ablate_hook(self, result, hook):
        result[:] = 0
        return result

    def zero_ablate_head_hook(self, result, hook, head_idx):
        result[:, :, head_idx] = 0
        return result

    def mean_ablate_hook(self, result, hook):
        result[:] = self.average_activations[hook.name]
        return result

    def mean_ablate_head_hook(self, result, hook, head_idx):
        result[:, :, head_idx] = self.average_activations[hook.name][head_idx]
        return result
        
    def ablate_component(self, component_name, hook):
        logit_diffs = {}
        for task in self.tasks:
            task_name = task.get_name()
            logit_diffs[task_name] = {}
            for key, sentence in task.get_examples().items():
                clean_logits = self.model(sentence)
                ablated_logits = self.model.run_with_hooks(
                    sentence,
                    fwd_hooks=[(component_name,
                                hook)],
                )
                self.model.reset_hooks()
                clean_logit_diff = task.get_logit_diff(
                    logits=logits, key=key)
                ablated_logit_diff = task.get_logit_diff(
                    logits=logits, key=key)
                logit_diffs[task_name][key] = (
                    ablated_logit_diff - clean_logit_diff
                ).item()

    def zero_ablate_attention_layer(self, layer_idx):
        logit_diffs = self.ablate_component(
            component_name=f'blocks.{layer_idx}.hook_attn_out',
            hook=self.zero_ablate_hook,
        )
        return logit_diffs

    def zero_ablate_attention_head(self, layer_idx, head_idx):
        logit_diffs = self.ablate_component(
            component_name=f'blocks.{layer_idx}.hook_v',
            hook=partial(self.zero_ablate_head_hook, head_idx=head_idx),
        )
        return logit_diffs
    
    def zero_ablate_mlp_layer(self, layer_idx):
        logit_diffs = self.ablate_component(
            component_name=f'blocks.{layer_idx}.hook_mlp_out',
            hook=self.zero_ablate_hook,
        )
        return logit_diffs

    def mean_ablate_attention_layer(self, layer_idx):
        assert self.average_activations is not None, 'must calculate average activations before calling this function'
        logit_diffs = self.ablate_component(
            component_name=f'blocks.{layer_idx}.hook_attn_out',
            hook=self.mean_ablate_hook,
        )
        return logit_diffs

    def mean_ablate_attention_head(self, layer_idx, head_idx):
        assert self.average_activations is not None, 'must calculate average activations before calling this function'
        logit_diffs = self.ablate_component(
            component_name=f'blocks.{layer_idx}.hook_v',
            hook=partial(self.mean_ablate_head_hook, head_idx=head_idx),
        )
        return logit_diffs
    
    def mean_ablate_mlp_layer(self, layer_idx):
        assert self.average_activations is not None, 'must calculate average activations before calling this function'
        logit_diffs = self.ablate_component(
            component_name=f'blocks.{layer_idx}.hook_mlp_out',
            hook=self.mean_ablate_hook,
        )
        return logit_diffs

    
