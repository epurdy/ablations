import abc
from collections import defaultdict
from functools import partial


import datasets
import einops
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from easy_transformer import EasyTransformer
from fancy_einsum import einsum


def return_zero_on_index_error(fn):
    def wrapped_fn(*args, **kwargs):
        try:
            retval = fn(*args, **kwargs)
            return retval
        except IndexError:
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
            dataset_name='stas/openwebtext-10k',
            max_samples=1000,
    ):

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
        result[:, posn_idx, head_idx] = self.average_activations[hook.name][head_idx]
        return result


    def ablate_component(self, component_name, hook):
        self.model.cfg.use_attn_result = True
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

        self.model.cfg.use_attn_result = False
        return logit_diffs

    def p2p_store_mlp_out_hook(self, result, hook, *, posn_idx, cache):
        cache[0] = (self.average_activations[hook.name] -
                    result[:, posn_idx])

    def p2p_store_attn_out_hook(self, result, hook, *,
                                posn_idx, head_idx, cache):
        cache[0] = (self.average_activations[hook.name] -
                    result[:, posn_idx, head_idx])

    def p2p_store_alternate_input_hook(self, result, hook, *,
                                       posn_idx, cache, ln):
        ln_result = ln(result)
        ln_result[:, posn_idx] = ln(result[:, posn_idx] + cache[0])
        cache[1] = ln_result

    def p2p_ablate_head_hook(self, result, hook, *,
                             posn_idx, head_idx,
                             attn, cache):
        new_input = cache[1]
        new_output = self.compute_attn(attn=attn, input_=new_input)
        result[:, posn_idx, head_idx] = new_output[:, posn_idx, head_idx]
        return result

    def p2p_ablate_mlp_hook(self, result, hook, *,
                            posn_idx,
                            mlp, cache):
        new_input = cache[1]
        new_output = self.compute_mlp(mlp=mlp, input_=new_input)
        result[:, posn_idx] = new_output[:, posn_idx]
        return result

    def compute_mlp(self, mlp, input_):
        # Technically, all these einsums could be done with a single matmul, but this is more readable.
        pre_act = einsum("batch pos d_model, d_model d_mlp -> batch pos d_mlp", input_, mlp.W_in) + mlp.b_in
        post_act = mlp.act_fn(pre_act)  # [batch, pos, d_mlp]
        assert not mlp.cfg.act_fn.endswith("_ln")
        mlp_out = (
            einsum("batch pos d_mlp, d_mlp d_model -> batch pos d_model", post_act, mlp.W_out) + mlp.b_out
        )  # [batch, pos, d_model]
        return mlp_out

    def compute_attn(self, attn, input_):
        # [batch, pos, head_index, d_head]
        q = einsum("batch pos d_model, head_index d_model d_head \
                    -> batch pos head_index d_head",
                   input_, attn.W_Q) + attn.b_Q
        # [batch, pos, head_index, d_head]
        k = einsum("batch pos d_model, head_index d_model d_head \
                    -> batch pos head_index d_head",
                   input_, attn.W_K) + attn.b_K
        # [batch, pos, head_index, d_head]
        v = einsum("batch pos d_model, head_index d_model d_head \
                    -> batch pos head_index d_head",
                   input_, attn.W_V) + attn.b_V
        attn_scores = (
            einsum("batch query_pos head_index d_head, \
                batch key_pos head_index d_head \
                -> batch head_index query_pos key_pos",
                   q, k) / attn.attn_scale
        )  # [batch, head_index, query_pos, key_pos]

        if attn.cfg.attention_dir == 'causal':
            # If causal attention, we mask it to only attend
            # backwards. If bidirectional, we don't mask.
            attn_scores = attn.apply_causal_mask(
                attn_scores,
                0
            ) # [batch, head_index, query_pos, key_pos]

        # [batch, head_index, query_pos, key_pos]
        attn_matrix = F.softmax(attn_scores, dim=-1)

        # [batch, pos, head_index, d_head]
        z = einsum("batch key_pos head_index d_head, \
                batch head_index query_pos key_pos -> \
                batch query_pos head_index d_head",
                v, attn_matrix)

        # [batch, pos, head_index, d_model]
        result = einsum("batch pos head_index d_head, \
                        head_index d_head d_model -> \
                        batch pos head_index d_model",
                        z,
                        attn.W_O)

        return result


    def mean_ablate_p2p(self, component_key1, component_key2):
        component1, index1 = component_key1
        component2, index2 = component_key2

        fwd_hooks = []
        p2p_cache = {}

        if component1 == 'heads':
            posn_idx1, layer_idx1, head_idx1 = index1
            fwd_hooks.append(
                (f'blocks.{layer_idx1}.hook_attn_out',
                 partial(self.p2p_store_attn_out_hook, posn_idx=posn_idx1,
                         head_idx=head_idx1,
                         cache=p2p_cache))
            )

        else:
            assert component1 == 'mlps'
            posn_idx1, layer_idx1 = index1
            fwd_hooks.append(
                (f'blocks.{layer_idx1}.hook_mlp_out',
                 partial(self.p2p_store_mlp_out_hook, posn_idx=posn_idx1,
                         cache=p2p_cache))
            )

        if component2 == 'heads':
            posn_idx2, layer_idx2, head_idx2 = index2
            fwd_hooks.append(
                (f'blocks.{layer_idx2}.hook_resid_pre',
                 partial(self.p2p_store_alternate_input_hook,
                         posn_idx=posn_idx1,
                         ln=self.model.blocks[layer_idx2].ln1,
                         cache=p2p_cache))
            )
            fwd_hooks.append(
                (f'blocks.{layer_idx2}.attn.hook_result',
                 partial(self.p2p_ablate_head_hook,
                         posn_idx=posn_idx2,
                         head_idx=head_idx2,
                         attn=self.model.blocks[layer_idx2].attn,
                         cache=p2p_cache))
            )

        else:
            assert component2 == 'mlps'
            posn_idx2, layer_idx2 = index2
            assert posn_idx1 == posn_idx2
            fwd_hooks.append(
                (f'blocks.{layer_idx2}.hook_resid_mid',
                 partial(self.p2p_store_alternate_input_hook,
                         posn_idx=posn_idx1,
                         ln=self.model.blocks[layer_idx2].ln2,
                         cache=p2p_cache))
            )
            fwd_hooks.append(
                (f'blocks.{layer_idx2}.hook_mlp_out',
                 partial(self.p2p_ablate_mlp_hook,
                         posn_idx=posn_idx2,
                         mlp=self.model.blocks[layer_idx2].mlp,
                         cache=p2p_cache))
            )


        self.model.cfg.use_attn_result = True
        logit_diffs = {}
        for task in self.tasks:
            task_name = task.get_name()
            logit_diffs[task_name] = {}
            for key, sentence in task.get_examples().items():
                for p2p_key in list(p2p_cache.keys()):
                    del p2p_cache[p2p_key]
                clean_logits = self.model(sentence)
                ablated_logits = self.model.run_with_hooks(
                    sentence,
                    fwd_hooks=fwd_hooks,
                )
                self.model.reset_hooks()
                clean_logit_diff = task.get_logit_diff(
                    logits=clean_logits, key=key)
                ablated_logit_diff = task.get_logit_diff(
                    logits=ablated_logits, key=key)
                logit_diffs[task_name][key] = (
                    ablated_logit_diff - clean_logit_diff
                ).item()

        self.model.cfg.use_attn_result = False
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
    def zero_ablate_attention_head_at_posn(self, *, layer_idx, head_idx,
                                           posn_idx):
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

        print('Ablating individual attention heads at all positions')
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

        print('Ablating attention layers at all positions')
        all_attn_logit_diffs = {}
        for layer_idx in tqdm.tqdm(range(self.model.cfg.n_layers)):
            for posn_idx in range(max_posn):
                logit_diffs = self.mean_ablate_attention_layer_at_posn(
                    layer_idx=layer_idx,
                    posn_idx=posn_idx,
                )
                all_attn_logit_diffs[posn_idx, layer_idx] = logit_diffs


        print('Ablating MLP layers at all positions')
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

        print('Ablating individual attention heads at all positions')
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

        print('Ablating attention layers at all positions')
        all_attn_logit_diffs = {}
        for layer_idx in tqdm.tqdm(range(self.model.cfg.n_layers)):
            for posn_idx in range(max_posn):
                logit_diffs = self.zero_ablate_attention_layer_at_posn(
                    layer_idx=layer_idx,
                    posn_idx=posn_idx,
                )
                all_attn_logit_diffs[posn_idx, layer_idx] = logit_diffs

        print('Ablating MLP layers at all positions')
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

    def summarize_logit_diffs(self, logit_diffs, depth=10):
        scored_components = defaultdict(list)
        for component_type in logit_diffs:
            for index in logit_diffs[component_type]:
                for task_name in logit_diffs[component_type][index]:
                    for example_key in logit_diffs[component_type][index][task_name]:
                        diff = logit_diffs[component_type][index][task_name][example_key]
                        scored_components[task_name,
                                          example_key,
                                          component_type].append(
                            (diff, index)
                        )

        for task_name, example_key, component_type in scored_components:
            scored_components[task_name,
                              example_key,
                              component_type].sort()
            print('-' * 80)
            print('Task:', task_name)
            print('Example key:', example_key)
            print('Component type:', component_type)
            for i in range(depth):
                diff, idx = scored_components[task_name,
                                              example_key,
                                              component_type][i]
                print(f'Component: {component_type} {idx} {diff:.3f}')

    def pick_interesting_logit_diffs(self, logit_diffs):
        # rearrange the scores into a nicer format
        scored_components = defaultdict(list)
        for component_type in logit_diffs:
            # skip whole attention-layers, since we have head-level
            # data
            if component_type == 'attns':
                continue
            for index in logit_diffs[component_type]:
                for task_name in logit_diffs[component_type][index]:
                    for example_key in logit_diffs[component_type][index][task_name]:
                        diff = logit_diffs[component_type][index][task_name][example_key]
                        scored_components[component_type, index].append(
                            diff
                        )

        # score components as the minimum logit diff seen from
        # ablating them
        component_scores = []
        for component_key in scored_components:
            score = min(scored_components[component_key] + [np.inf])
            component_scores.append((score, component_key))
        component_scores.sort()

        scored_pairs = []
        for i, (score1, component_key1) in enumerate(component_scores):
            for j, (score2, component_key2) in enumerate(component_scores):
                if j <= i:
                    continue

                component1, index1 = component_key1
                component2, index2 = component_key2

                if component1 == 'mlps':
                    if component2 == 'mlps':
                        posn1, layer1 = index1
                        posn2, layer2 = index2

                        # can't do a non-trivial point-to-point
                        # ablation if two mlps are on different tokens
                        if posn1 != posn2:
                            continue

                        assert layer1 != layer2, 'should not happen'

                        if layer1 < layer2:
                            scored_pairs.append(
                                ((score1 + score2),
                                 component_key1,
                                 component_key2))
                        else:
                            scored_pairs.append(
                                ((score1 + score2),
                                 component_key2,
                                 component_key1))

                    elif component2 == 'heads':
                        posn1, layer1 = index1
                        posn2, layer2, head2 = index2

                        # if mlp is after the attention head, then it
                        # must be on the same token for a
                        # point-to-point ablation to be non-trivial
                        if layer1 >= layer2:
                            if posn1 != posn2:
                                continue
                            scored_pairs.append(
                                ((score1 + score2),
                                 component_key2,
                                 component_key1))
                        else:
                            # mlp is before the attention head; it can
                            # potentially be read from, if the
                            # relevant attention value is non-zero
                            scored_pairs.append(
                                ((score1 + score2),
                                 component_key1,
                                 component_key2))

                    else:
                        assert False, 'should not get here'
                elif component1 == 'heads':
                    if component2 == 'mlps':
                        posn1, layer1, head1 = index1
                        posn2, layer2 = index2

                        # if mlp is after the attention head, then it
                        # must be on the same token for a
                        # point-to-point ablation to be non-trivial
                        if layer2 >= layer1:
                            if posn1 != posn2:
                                continue
                            scored_pairs.append(
                                ((score1 + score2),
                                 component_key1,
                                 component_key2))
                        else:
                            # mlp is before the attention head; it can
                            # potentially be read from, if the
                            # relevant attention value is non-zero
                            scored_pairs.append(
                                ((score1 + score2),
                                 component_key2,
                                 component_key1))

                    elif component2 == 'heads':
                        posn1, layer1, head1 = index1
                        posn2, layer2, head2 = index2

                        # can't do a non-trivial point-to-point
                        # ablation if the two heads are on the same
                        # layer
                        if layer1 == layer2:
                            continue

                        # because of causal masking, can only have a
                        # non-trivial point-to-point ablation if
                        # earlier layer is also at an earlier position
                        # (or the same position)
                        if layer1 < layer2 and posn1 <= posn2:
                            scored_pairs.append(
                                ((score1 + score2),
                                 component_key1,
                                 component_key2))

                        # reverse case
                        elif layer2 < layer1 and posn2 <= posn1:
                            scored_pairs.append(
                                ((score1 + score2),
                                 component_key2,
                                 component_key1))

                        # otherwise, no non-trivial point-to-point
                        # ablation is possible
                        else:
                            continue

                    else:
                        assert False, 'should not get here'
                else:
                    assert False, 'should not get here'

        print('All done!')
        scored_pairs.sort()
        for score, component_key1, component_key2  in scored_pairs:
            yield score, component_key1, component_key2
