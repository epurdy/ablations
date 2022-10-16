import abc
from collections import defaultdict
from functools import partial


import datasets
import einops
import torch
import tqdm
from easy_transformer import EasyTransformer


def return_zero_on_index_error(fn):
    def wrapped_fn(*args, **kwargs):
        try:
            retval = fn(*args, **kwargs)
        except:
            return 0
    return wrapped_fn

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
    def __init__(self, model_name='gpt2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = EasyTransformer.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        torch.set_grad_enabled(False)

        self.tasks = []

        self.example_cache = None
        self.average_activations = None

    def register_task(self, task):
        self.tasks.append(task)

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

        print('Calculating mean activations...')
        self.model.cfg.use_attn_result = True

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
            if hook_suffix in ['hook_result',
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
                return_tensors='pt').to(self.device)
            logits = self.model(tokens)
            count_samples += len(tokens)
            if count_samples > max_samples:
                break

        average_acts = {}
        for name, hook in self.model.hook_dict.items():
            hook_suffix = name.split('.')[-1]
            if hook_suffix in ['hook_result', 'hook_attn_out', 'hook_mlp_out']:
                average_acts[hook.name] = hook.ctx['running_total']/hook.ctx['count']
        print('Average act keys:', average_acts.keys())
        self.model.reset_hooks(clear_contexts=True)

        self.average_activations = average_acts
        self.model.cfg.use_attn_result = False

    def zero_ablate_hook(self, result, hook):
        result[:] = 0
        return result

    def zero_ablate_head_hook(self, result, hook, *, head_idx):
        result[:, :, head_idx] = 0
        return result

    def zero_ablate_position_hook(self, result, hook, *, posn_idx):
        result[:, posn_idx] = 0

    def zero_ablate_head_position_hook(self, result, hook, *, posn_idx, head_idx):
        result[:, posn_idx, head_idx] = 0

    def mean_ablate_hook(self, result, hook):
        result[:] = self.average_activations[hook.name]
        return result

    def mean_ablate_head_hook(self, result, hook, *, head_idx):
        result[:, :, head_idx] = self.average_activations[hook.name][head_idx]
        return result

    def mean_ablate_position_hook(self, result, hook, *, posn_idx):
        result[:, posn_idx] = self.average_activations[hook.name]
        return result

    def mean_ablate_head_position_hook(self, result, hook, *,
                                       posn_idx, head_idx):
        result[:, posn_idx, head_idx] = self.average_activations[hook.name]
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
                    logits=clean_logits, key=key)
                ablated_logit_diff = task.get_logit_diff(
                    logits=ablated_logits, key=key)
                logit_diffs[task_name][key] = (
                    ablated_logit_diff - clean_logit_diff
                ).item()

        return logit_diffs

    def zero_ablate_attention_layer(self, *, layer_idx):
        logit_diffs = self.ablate_component(
            component_name=f'blocks.{layer_idx}.hook_attn_out',
            hook=self.zero_ablate_hook,
        )
        return logit_diffs

    @return_zero_on_index_error
    def zero_ablate_attention_layer_at_posn(self, *, layer_idx, posn_idx):
        logit_diffs = self.ablate_component(
            component_name=f'blocks.{layer_idx}.hook_attn_out',
            hook=partial(self.zero_ablate_position_hook,
                         posn_idx=posn_idx),
        )
        return logit_diffs

    def zero_ablate_attention_head(self, *, layer_idx, head_idx):
        logit_diffs = self.ablate_component(
            component_name=f'blocks.{layer_idx}.attn.hook_result',
            hook=partial(self.zero_ablate_head_hook, head_idx=head_idx),
        )
        return logit_diffs

    @return_zero_on_index_error
    def zero_ablate_attention_head_at_posn(self, *, layer_idx, head_idx, posn_idx):
        logit_diffs = self.ablate_component(
            component_name=f'blocks.{layer_idx}.attn.hook_result',
            hook=partial(self.zero_ablate_head_position_hook,
                         head_idx=head_idx,
                         posn_idx=posn_idx),
        )
        return logit_diffs
    
    def zero_ablate_mlp_layer(self, *, layer_idx):
        logit_diffs = self.ablate_component(
            component_name=f'blocks.{layer_idx}.hook_mlp_out',
            hook=self.zero_ablate_hook,
        )
        return logit_diffs

    @return_zero_on_index_error
    def zero_ablate_mlp_layer_at_posn(self, *, layer_idx, posn_idx):
        logit_diffs = self.ablate_component(
            component_name=f'blocks.{layer_idx}.hook_mlp_out',
            hook=partial(self.zero_ablate_position_hook,
                         posn_idx=posn_idx),
        )
        return logit_diffs
    
    def mean_ablate_attention_layer(self, *, layer_idx):
        assert self.average_activations is not None, 'must calculate average activations before calling this function'
        logit_diffs = self.ablate_component(
            component_name=f'blocks.{layer_idx}.hook_attn_out',
            hook=self.mean_ablate_hook,
        )
        return logit_diffs

    @return_zero_on_index_error
    def mean_ablate_attention_layer_at_posn(self, *,
                                            layer_idx, posn_idx):
        assert self.average_activations is not None, 'must calculate average activations before calling this function'
        logit_diffs = self.ablate_component(
            component_name=f'blocks.{layer_idx}.hook_attn_out',
            hook=partial(self.mean_ablate_position_hook,
                         posn_idx=posn_idx),
        )
        return logit_diffs
    
    def mean_ablate_attention_head(self, *, layer_idx, head_idx):
        assert self.average_activations is not None, 'must calculate average activations before calling this function'
        logit_diffs = self.ablate_component(
            component_name=f'blocks.{layer_idx}.attn.hook_result',
            hook=partial(self.mean_ablate_head_hook, head_idx=head_idx),
        )
        return logit_diffs

    @return_zero_on_index_error
    def mean_ablate_attention_head_at_posn(self, *,
                                           layer_idx,
                                           head_idx,
                                           posn_idx):
        assert self.average_activations is not None, 'must calculate average activations before calling this function'
        logit_diffs = self.ablate_component(
            component_name=f'blocks.{layer_idx}.attn.hook_result',
            hook=partial(self.mean_ablate_head_position_hook,
                         head_idx=head_idx,
                         posn_idx=posn_idx),
        )
        return logit_diffs
    
    def mean_ablate_mlp_layer(self, *, layer_idx):
        assert self.average_activations is not None, 'must calculate average activations before calling this function'
        logit_diffs = self.ablate_component(
            component_name=f'blocks.{layer_idx}.hook_mlp_out',
            hook=self.mean_ablate_hook,
        )
        return logit_diffs

    @return_zero_on_index_error
    def mean_ablate_mlp_layer_at_posn(self, *, layer_idx, posn_idx):
        assert self.average_activations is not None, 'must calculate average activations before calling this function'
        logit_diffs = self.ablate_component(
            component_name=f'blocks.{layer_idx}.hook_mlp_out',
            hook=partial(self.mean_ablate_position_hook,
                         posn_idx=posn_idx),
        )
        return logit_diffs
    
    def run_mean_ablation_sweep(self):
        assert self.average_activations is not None, 'must calculate average activations before calling this function'

        print('Ablating individual attention heads')
        all_head_logit_diffs = {}
        for layer_idx in tqdm.tqdm(range(self.model.cfg.n_layers)):
            for head_idx in range(self.model.cfg.n_heads):
                logit_diffs = self.mean_ablate_attention_head(
                    layer_idx=layer_idx, head_idx=head_idx
                )
                all_head_logit_diffs[layer_idx, head_idx] = logit_diffs

        print('Ablating attention layers')
        all_attn_logit_diffs = {}
        for layer_idx in tqdm.tqdm(range(self.model.cfg.n_layers)):
            logit_diffs = self.mean_ablate_attention_layer(
                layer_idx=layer_idx,
            )
            all_attn_logit_diffs[layer_idx] = logit_diffs


        print('Ablating MLP layers')
        all_mlp_logit_diffs = {}
        for layer_idx in tqdm.tqdm(range(self.model.cfg.n_layers)):
            logit_diffs = self.mean_ablate_mlp_layer(
                layer_idx=layer_idx,
            )
            all_mlp_logit_diffs[layer_idx] = logit_diffs

        return dict(
            heads=all_head_logit_diffs,
            mlps=all_mlp_logit_diffs,
            attns=all_attn_logit_diffs
        )

    def run_positional_mean_ablation_sweep(self):
        assert self.average_activations is not None, 'must calculate average activations before calling this function'

        max_posn = 0
        for task in self.tasks:
            for example in task.get_examples().values():
                max_posn = max(
                    max_posn,
                    len(self.model.tokenizer.encode(example)))
        print('max_posn', max_posn)
        
        print('Ablating individual attention heads')
        all_head_logit_diffs = {}
        for layer_idx in tqdm.tqdm(range(self.model.cfg.n_layers)):
            for head_idx in range(self.model.cfg.n_heads):
                for posn_idx in range(max_posn):
                    logit_diffs = self.mean_ablate_attention_head_at_posn(
                        layer_idx=layer_idx, head_idx=head_idx,
                        posn_idx=posn_idx,
                    )
                    all_head_logit_diffs[posn_idx, layer_idx, head_idx] = (
                        logit_diffs
                    )

        print('Ablating attention layers')
        all_attn_logit_diffs = {}
        for layer_idx in tqdm.tqdm(range(self.model.cfg.n_layers)):
            for posn_idx in range(max_posn):
                logit_diffs = self.mean_ablate_attention_layer_at_posn(
                    layer_idx=layer_idx,
                    posn_idx=posn_idx,
                )
                all_attn_logit_diffs[posn_idx, layer_idx] = logit_diffs


        print('Ablating MLP layers')
        all_mlp_logit_diffs = {}
        for layer_idx in tqdm.tqdm(range(self.model.cfg.n_layers)):
            for posn_idx in range(max_posn):
                logit_diffs = self.mean_ablate_mlp_layer_at_posn(
                    layer_idx=layer_idx,
                    posn_idx=posn_idx,
                )
                all_mlp_logit_diffs[posn_idx, layer_idx] = logit_diffs

        return dict(
            heads=all_head_logit_diffs,
            mlps=all_mlp_logit_diffs,
            attns=all_attn_logit_diffs
        )
    

    def run_zero_ablation_sweep(self):
        assert self.average_activations is not None, 'must calculate average activations before calling this function'

        print('Ablating individual attention heads')
        all_head_logit_diffs = {}
        for layer_idx in tqdm.tqdm(range(self.model.cfg.n_layers)):
            for head_idx in range(self.model.cfg.n_heads):
                logit_diffs = self.zero_ablate_attention_head(
                    layer_idx=layer_idx, head_idx=head_idx
                )
                all_head_logit_diffs[layer_idx, head_idx] = logit_diffs

        print('Ablating attention layers')
        all_attn_logit_diffs = {}
        for layer_idx in tqdm.tqdm(range(self.model.cfg.n_layers)):
            logit_diffs = self.zero_ablate_attention_layer(
                layer_idx=layer_idx,
            )
            all_attn_logit_diffs[layer_idx] = logit_diffs


        print('Ablating MLP layers')
        all_mlp_logit_diffs = {}
        for layer_idx in tqdm.tqdm(range(self.model.cfg.n_layers)):
            logit_diffs = self.zero_ablate_mlp_layer(
                layer_idx=layer_idx,
            )
            all_mlp_logit_diffs[layer_idx] = logit_diffs

        return dict(
            heads=all_head_logit_diffs,
            mlps=all_mlp_logit_diffs,
            attns=all_attn_logit_diffs
        )

    def run_positional_zero_ablation_sweep(self):
        max_posn = 0
        for task in self.tasks:
            for example in task.get_examples().values():
                max_posn = max(
                    max_posn,
                    len(self.model.tokenizer.encode(example)))
        print('max_posn', max_posn)
        
        print('Ablating individual attention heads')
        all_head_logit_diffs = {}
        for layer_idx in tqdm.tqdm(range(self.model.cfg.n_layers)):
            for head_idx in range(self.model.cfg.n_heads):
                for posn_idx in range(max_posn):
                    logit_diffs = self.zero_ablate_attention_head_at_posn(
                        layer_idx=layer_idx, head_idx=head_idx,
                        posn_idx=posn_idx,
                    )
                    all_head_logit_diffs[posn_idx, layer_idx, head_idx] = (
                        logit_diffs
                    )

        print('Ablating attention layers')
        all_attn_logit_diffs = {}
        for layer_idx in tqdm.tqdm(range(self.model.cfg.n_layers)):
            for posn_idx in range(max_posn):
                logit_diffs = self.zero_ablate_attention_layer_at_posn(
                    layer_idx=layer_idx,
                    posn_idx=posn_idx,
                )
                all_attn_logit_diffs[posn_idx, layer_idx] = logit_diffs

        print('Ablating MLP layers')
        all_mlp_logit_diffs = {}
        for layer_idx in tqdm.tqdm(range(self.model.cfg.n_layers)):
            for posn_idx in range(max_posn):
                logit_diffs = self.zero_ablate_mlp_layer_at_posn(
                    layer_idx=layer_idx,
                    posn_idx=posn_idx,
                )
                all_mlp_logit_diffs[posn_idx, layer_idx] = logit_diffs

        return dict(
            heads=all_head_logit_diffs,
            mlps=all_mlp_logit_diffs,
            attns=all_attn_logit_diffs
        )
    
    def summarize_logit_diffs(self, logit_diffs):
        scored_components = defaultdict(list)
        for component_type in logit_diffs:
            for index in logit_diffs[component_type]:
                for task_name in logit_diffs[component_type][index]:
                    for example_key in logit_diffs[component_type][index][task_name]:
                        diff = logit_diffs[component_type][index][task_name][example_key]
                        scored_components[task_name, example_key].append(
                            (diff, component_type, index)
                        )

        for task_name, example_key in scored_components:
            scored_components[task_name, example_key].sort()
            print('-' * 80)
            print('Task:', task_name)
            print('Example key:', example_key)
            for i in range(10):
                diff, typ, idx = scored_components[task_name, example_key][i]
                print(f'Component: {typ} {idx} {diff:.3f}')
