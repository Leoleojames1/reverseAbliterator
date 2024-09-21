import torch
import torch.nn.functional as F
import functools
import einops
import gc
from itertools import islice
from tqdm import tqdm
from typing import Callable, Dict, List, Set, Tuple
from transformer_lens import HookedTransformer, utils, ActivationCache
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float, Int

class ReverseAbliterator:
    def __init__(
        self,
        model: str,
        dataset: Tuple[List[str], List[str]]|List[Tuple[List[str], List[str]]],
        device: str = 'cuda',
        n_devices: int = None,
        cache_fname: str = None,
        activation_layers: List[str] = ['resid_pre', 'resid_post', 'mlp_out', 'attn_out'],
        chat_template: str = None,
        target_toks: List[int]|Tuple[int]|Set[int]|Int[Tensor, '...'] = None,
    ):
        self.MODEL_PATH = model
        if n_devices is None and torch.cuda.is_available():
            n_devices = torch.cuda.device_count()
        elif n_devices is None:
            n_devices = 1

        torch.set_grad_enabled(False)

        self.model = HookedTransformer.from_pretrained_no_processing(
            model,
            n_devices=n_devices,
            device=device,
            dtype=torch.bfloat16,
            default_padding_side='left'
        )

        self.model.requires_grad_(False)

        self.model.tokenizer.padding_side = 'left'
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        self.chat_template = chat_template or ChatTemplate(self, LLAMA3_CHAT_TEMPLATE)

        self.hidden_size = self.model.cfg.d_model
        self.original_state = {k:v.to('cpu') for k,v in self.model.state_dict().items()}
        self.target = {}
        self.baseline = {}
        self.modified_layers = {'mlp':{}, 'W_O':{}}
        self.checkpoints = []

        if cache_fname is not None:
            outs = torch.load(cache_fname, map_location='cpu')
            self.target, self.baseline, modified_layers, checkpoints = outs[:4]
            self.checkpoints = checkpoints or []
            self.modified_layers = modified_layers

        self.target_inst_train, self.target_inst_test = prepare_dataset(dataset[0])
        self.baseline_inst_train, self.baseline_inst_test = prepare_dataset(dataset[1])

        self.fwd_hooks = []
        self.modified = False
        self.activation_layers = [activation_layers] if isinstance(activation_layers, str) else activation_layers
        self.target_toks = target_toks or {32, 1271, 8586, 96556, 78145}  # Default to some positive tokens
        self._blacklisted = set()

    def reset_state(self):
        self.modified = False
        self.modified_layers = {'mlp':{}, 'W_O':{}}
        self.model.load_state_dict(self.original_state)

    def checkpoint(self):
        self.checkpoints.append(self.modified_layers.copy())

    def save_activations(self, fname: str):
        torch.save([self.target, self.baseline, self.modified_layers if self.modified_layers['mlp'] or self.modified_layers['W_O'] else None, self.checkpoints if len(self.checkpoints) > 0 else None], fname)

    def calculate_enhancement_dirs(self, key: str) -> Dict[str, Float[Tensor, 'd_model']]:
        dirs = {
            'target_mean': torch.mean(self.target[key], dim=0),
            'baseline_mean': torch.mean(self.baseline[key], dim=0)
        }
        dirs['enhancement_dir'] = dirs['target_mean'] - dirs['baseline_mean']
        return dirs

    def enhancement_dirs(self) -> Dict[str, Float[Tensor, 'd_model']]:
        if not self.target:
            raise IndexError("No cache")

        enhancement_dirs = {key: self.calculate_enhancement_dirs(key) for key in self.target if '.0.' not in key}
        return {key: (v['enhancement_dir'] / v['enhancement_dir'].norm()).to('cpu') for key, v in enhancement_dirs.items()}

    def apply_enhancement_dirs(
        self,
        enhancement_dirs: List[Float[Tensor, 'd_model']],
        W_O: bool = True,
        mlp: bool = True,
        layers: List[int] = None,
        strength: float = 1.0
    ):
        if layers is None:
            layers = list(range(1, self.model.cfg.n_layers))
        for enhancement_dir in enhancement_dirs:
            for layer in layers:
                for modifying in [(W_O, self.layer_attn), (mlp, self.layer_mlp)]:
                    if modifying[0]:
                        matrix = modifying[1](layer)
                        if enhancement_dir.device != matrix.device:
                            enhancement_dir = enhancement_dir.to(matrix.device)
                        proj = einops.einsum(matrix, enhancement_dir.view(-1, 1), '... d_model, d_model single -> ... single') * enhancement_dir
                        modifying[1](layer, matrix + strength * proj)

    def layer_attn(self, layer: int, replacement: Float[Tensor, "d_model"] = None) -> Float[Tensor, "d_model"]:
        if replacement is not None and layer not in self._blacklisted:
            self.modified = True
            self.model.blocks[layer].attn.W_O.data = replacement.to(self.model.blocks[layer].attn.W_O.device)
            self.modified_layers['W_O'][layer] = self.modified_layers.get(layer, []) + [(self.model.blocks[layer].attn.W_O.data.to('cpu'), replacement.to('cpu'))]
        return self.model.blocks[layer].attn.W_O.data

    def layer_mlp(self, layer: int, replacement: Float[Tensor, "d_model"] = None) -> Float[Tensor, "d_model"]:
        if replacement is not None and layer not in self._blacklisted:
            self.modified = True
            self.model.blocks[layer].mlp.W_out.data = replacement.to(self.model.blocks[layer].mlp.W_out.device)
            self.modified_layers['mlp'][layer] = self.modified_layers.get(layer, []) + [(self.model.blocks[layer].mlp.W_out.data.to('cpu'), replacement.to('cpu'))]
        return self.model.blocks[layer].mlp.W_out.data

    def cache_activations(
        self,
        N: int = 128,
        batch_size: int = 8,
        last_indices: int = 1,
        reset: bool = True,
        activation_layers: int = -1,
        preserve_baseline: bool = True,
    ):
        if hasattr(self, "current_state"):
            print("WARNING: Caching activations using a context")
        if self.modified:
            print("WARNING: Running modified model")

        if activation_layers == -1:
            activation_layers = self.activation_layers

        baseline_is_set = len(getattr(self, "baseline", {})) > 0
        preserve_baseline = baseline_is_set and preserve_baseline

        if reset or getattr(self, "baseline", None) is None:
            self.target = {}
            if not preserve_baseline:
                self.baseline = {}

        toks = self.tokenize_instructions_fn(instructions=self.target_inst_train[:N] + self.baseline_inst_train[:N])

        splitpos = min(N, len(self.target_inst_train))
        target_toks = toks[:splitpos]
        baseline_toks = toks[splitpos:]

        last_indices = last_indices or 1

        self.target = self.create_activation_cache(target_toks, N=N, batch_size=batch_size, last_indices=last_indices)
        if not preserve_baseline:
            self.baseline = self.create_activation_cache(baseline_toks, N=N, batch_size=batch_size, last_indices=last_indices)

    def create_activation_cache(
        self,
        toks,
        N: int = 128,
        batch_size: int = 8,
        last_indices: int = 1,
    ) -> Dict[str, Float[Tensor, 'batch d_model']]:
        base = {}
        for i in tqdm(range(0, min(N, len(toks)), batch_size)):
            logits, cache = self.run_with_cache(toks[i:min(i+batch_size, len(toks))])
            for key in cache:
                if self.activation_layers is None or any(k in key for k in self.activation_layers):
                    tensor = torch.mean(cache[key][:, -last_indices:, :].to('cpu'), dim=1)
                    if key not in base:
                        base[key] = tensor
                    else:
                        base[key] = torch.cat((base[key], tensor), dim=0)
            del logits, cache
            gc.collect()
            torch.cuda.empty_cache()

        return base

    def measure_enhancement(
        self,
        N: int = 4,
        sampled_token_ct: int = 8,
        measure: str = 'max',
    ) -> Dict[str, Float[Tensor, 'd_model']]:
        toks = self.tokenize_instructions_fn(instructions=self.target_inst_test[:N])
        logits, _ = self.run_with_cache(toks, max_new_tokens=sampled_token_ct)

        enhancement_score = self.measure_enhancement_from_logits(logits, sampled_token_ct, measure=measure)
        return {'enhancement': enhancement_score.to('cpu')}

    def measure_enhancement_from_logits(
        self,
        logits: Float[Tensor, 'batch_size seq_len d_vocab'],
        sequence: int,
        measure: str = 'max'
    ) -> Float[Tensor, 'batch_size']:
        normalized_scores = torch.softmax(logits[:, -sequence:, :].to('cpu'), dim=-1)[:, :, list(self.target_toks)]
        max_score_per_sequence = torch.max(normalized_scores, dim=-1)[0]
        score_per_batch = getattr(torch, measure)(max_score_per_sequence, dim=-1)[0]
        return score_per_batch

    def run_with_cache(
        self,
        *model_args,
        names_filter: Callable[[str], bool] = None,
        max_new_tokens: int = 1,
        **model_kwargs
    ) -> Tuple[Float[Tensor, 'batch_size seq_len d_vocab'], Dict[str, Float[Tensor, 'batch_size seq_len d_model']]]:
        if names_filter is None and self.activation_layers:
            names_filter = lambda namefunc: any(s in namefunc for s in self.activation_layers)

        cache_dict, fwd, _ = self.model.get_caching_hooks(
            names_filter,
            remove_batch_dim=False,
            pos_slice=utils.Slice(None)
        )

        fwd_hooks = fwd + self.fwd_hooks

        with self.model.hooks(fwd_hooks=fwd_hooks):
            model_out, _ = self.generate_logits(*model_args, max_tokens_generated=max_new_tokens, **model_kwargs)

        return model_out, cache_dict

    def generate_logits(
        self,
        toks: Int[Tensor, 'batch_size seq_len'],
        *args,
        max_tokens_generated: int = 1,
        **kwargs
    ) -> Tuple[Float[Tensor, 'batch_size seq_len d_vocab'], Int[Tensor, 'batch_size seq_len']]:
        all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
        all_toks[:, :toks.shape[1]] = toks
        for i in range(max_tokens_generated):
            logits = self.model(all_toks[:, :-max_tokens_generated + i], *args, **kwargs)
            next_tokens = logits[:, -1, :].argmax(dim=-1)
            all_toks[:, -max_tokens_generated + i] = next_tokens
        return logits, all_toks

    def tokenize_instructions_fn(
        self,
        instructions: List[str]
    ) -> Int[Tensor, 'batch_size seq_len']:
        prompts = [self.chat_template.format(instruction=instruction) for instruction in instructions]
        return self.model.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").input_ids

    def enhance_model(
        self,
        layers: List[int] = None,
        W_O: bool = True,
        mlp: bool = True,
        strength: float = 1.0,
    ):
        enhancement_directions = self.enhancement_dirs()
        self.apply_enhancement_dirs(
            list(enhancement_directions.values()),
            W_O=W_O,
            mlp=mlp,
            layers=layers,
            strength=strength
        )

    def test_enhancement(
        self,
        N: int = 16,
        batch_size: int = 4,
        max_tokens_generated: int = 64,
    ):
        for prompts in batch(self.target_inst_test[:min(len(self.target_inst_test), N)], batch_size):
            toks = self.tokenize_instructions_fn(prompts)
            _, all_toks = self.generate_logits(toks, max_tokens_generated=max_tokens_generated)
            responses = self.model.tokenizer.batch_decode(all_toks, skip_special_tokens=True)
            for prompt, response in zip(prompts, responses):
                print(f"Prompt: {prompt}\nResponse: {response}\n")

# Utility functions

def batch(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk

def prepare_dataset(dataset: Tuple[List[str], List[str]]|List[str]) -> Tuple[List[str], List[str]]:
    from sklearn.model_selection import train_test_split
    if len(dataset) != 2:
        train, test = train_test_split(dataset, test_size=0.1, random_state=42)
    else:
        train, test = dataset
    return train, test

if __name__ == "__main__":
    # Example usage of ReverseAbliterator
    
    # Define your model path and datasets
    model_path = "path/to/your/model"
    target_instructions = ["Write a poem about nature", "Explain quantum physics", "Describe the process of photosynthesis"]
    baseline_instructions = ["Hello", "What's the weather like?", "Tell me a joke"]

    # Initialize ReverseAbliterator
    reverse_abliterator = ReverseAbliterator(
        model=model_path,
        dataset=([target_instructions, baseline_instructions]),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Cache activations
    reverse_abliterator.cache_activations(N=len(target_instructions), batch_size=1)

    # Measure initial enhancement
    initial_enhancement = reverse_abliterator.measure_enhancement()
    print("Initial enhancement score:", initial_enhancement)

    # Enhance the model
    reverse_abliterator.enhance_model(strength=0.1)  # Start with a small strength

    # Measure enhancement after modification
    post_enhancement = reverse_abliterator.measure_enhancement()
    print("Post-enhancement score:", post_enhancement)

    # Test the enhanced model
    print("Testing enhanced model responses:")
    reverse_abliterator.test_enhancement(N=3, max_tokens_generated=30)

    # Save the modified model state if desired
    reverse_abliterator.save_activations("enhanced_model_state.pt")

    print("Reverse abliteration process complete.")
