from logging import getLogger
from pprint import pformat
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from ..utils.cache_prefix_sampler import SequenceCache
from .model import Model
from .model_utils import KeyWordsCriteria

if TYPE_CHECKING:
    from ..utils import ModelArguments

logger = getLogger(__name__)

_InputsWithOptionNum = Union[List[Tuple[str, int]], List[Tuple[str, str, int]], List[Tuple[str, str, str, int]]]
_PostfixEncoding = Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[int], List[int]]
"""`tuple(input_ids, attention_mask, input_pos, prefix_lengths, input_lengths)`"""


def load_tokenizer(tokenizer_name_or_path: str, use_fast: bool, max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, use_fast=use_fast, padding_side="left", truncation_side="left", add_eos_token=False
    )

    # TODO: [Important]!!! check for each tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    max_length = min(max_length, getattr(tokenizer, "tokenizer_model_max_length", 1e10))
    tokenizer.model_max_length = max_length
    return tokenizer


def load_hf_model(args: "ModelArguments") -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    logger.info(f"Loading {args.model_name_or_path} using Hugging Face Transformers...")

    model_kwargs = dict(
        torch_dtype=torch.float16,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    if args.prefix_caching:
        model_kwargs["is_decoder"] = True

    if args.flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if hasattr(args, 'bnb_config') and args.bnb_config:
        model_kwargs['quantization_config'] = args.bnb_config

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs).eval()
    except (TypeError, ImportError, ValueError) as e:
        if "attn_implementation" in str(e) or "flash att" in str(e).lower().replace("_", " "):
            logger.warning(
                f"Cannot set `attn_implementation` for {args.model_name_or_path}: {e}. Set `flash_attention` to False."
            )
            args.flash_attention = False
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs).eval()
        else:
            raise e

    # TODO: [Important]!!! check for each tokenizer
    max_length = args.max_length or 1e10
    for key in [
        "max_sequence_length",
        "max_position_embeddings",
        "model_max_length",
        "seq_length",
        "seq_len",
        "n_positions",
        "max_seq_len",
        "max_seq_length",
    ]:
        max_length = min(max_length, getattr(model.config, key, 1e10))
    if not max_length or max_length >= 1e10:
        max_length = 2048
        logger.warning(
            f"Cannot specify model's maximum length according to `args` or model config. Set to 2048 by default."
        )

    slow_tokenizer = load_tokenizer(args.tokenizer_name_or_path, use_fast=False, max_length=max_length)
    logger.debug(f"Model: {model}\nTokenizer: {slow_tokenizer}")
    return model, slow_tokenizer


class HuggingFaceModel(Model):

    model: PreTrainedModel
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

    def __init__(self, args: "ModelArguments"):
        super().__init__(args)
        self.args = args
        self.type = args.model_type
        if self.type not in {"base", "instruction"}:
            raise ValueError(
                f"Invalid model type: {self.type}. Please use `--model_type` to specify the"
                " model type, which can be chosen from `base` and `instruction`."
            )

        self.model, self.tokenizer = load_hf_model(args)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def post_fork_init(self):
        self.tokenizer = load_tokenizer(
            self.args.tokenizer_name_or_path, use_fast=True, max_length=self.tokenizer.model_max_length
        )
        if hasattr(self, "generation_kwargs") and "stop" in self.generation_kwargs:
            self.stop_id_sequences = self._tokenize_postfix(
                self.generation_kwargs.pop("stop"), add_dummy_prefix=True, padding=False
            )
            self.generation_kwargs["stopping_criteria"] = [KeyWordsCriteria(self.stop_id_sequences)]

    def _process_postfix_encodings(
        self,
        attention_mask: torch.Tensor,
        prefix_cache: SequenceCache,
        device: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Concatenate the attention_mask of the prefix with the input attention_mask and calculate the position_ids of the input based on the length of the prefix."""

        _to = dict(dtype=torch.long, device=self.device)
        if device is not None:
            _to["device"] = torch.device(device)
        max_prefix_len = prefix_cache.get_seq_length()
        batch_size, max_input_len = attention_mask.size()

        # prepare attention_mask of prefix, and position_ids of postfix (continue from the last token of prefix)
        prefix_mask = torch.ones((batch_size, max_prefix_len), **_to)
        if prefix_cache.get_seq_num() == 1 and batch_size > 1:
            # same prefix for all inputs
            prefix_cache = prefix_cache.expand_seq(batch_size)
            prefix_lengths = [prefix_cache.get_seq_length()] * batch_size
            input_pos = torch.arange(max_prefix_len, max_prefix_len + max_input_len, **_to).expand(batch_size, -1)
        else:
            # different prefix for each input
            prefix_lengths = []
            input_pos = []
            for seq_idx in range(batch_size):
                prefix_len = prefix_cache.real_seq_length[seq_idx]
                prefix_mask[seq_idx, :-prefix_len] = 0  # prefix is left padded
                prefix_lengths.append(prefix_len)
                input_pos.append(torch.arange(prefix_len, max_input_len + prefix_len))
            input_pos = torch.stack(input_pos).to(**_to)

        # concatenate the prefix and input attention_mask
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1).to(**_to)  # type: ignore
        return attention_mask, input_pos, prefix_lengths

    def _tokenize_postfix(
        self,
        batched_inputs: List[str],
        prefix_cache: Optional[SequenceCache] = None,
        device: Optional[str] = None,
        add_dummy_prefix: bool = False,
        dummy_prefix: str = "text ",
        padding: bool = True,
    ) -> Union[List[List[int]], _PostfixEncoding]:
        """Tokenize the inputs as postfix. If `prefix_cache` is provided, the attention_mask of the prefix will be concatenated with the input attention_mask and the position_ids of the input will be calculated based on the length of the prefix.

        Args:
            batched_inputs (`List[str]`): Batched inputs to be tokenized as postfix.
            prefix_cache (`Optional[SequenceCache]`, optional): The SequenceCache of prefix. Defaults to None.
            device (`Optional[str]`, optional): Target device of returned tensors. Defaults to None.
            add_prefix (`bool`, optional): If no `prefix_cache` is provided, use this to add a dummy prefix and remove it after tokenization. Defaults to False.
            padding (`bool`, optional): Whether to pad the sequence (to right) and return in tensor. Defaults to True.

        Returns:
            - If `padding` is True:
                `_PostfixEncoding`: Encoding of postfix with padding.
            - If `padding` is False:
                `List[List[int]]`: A list of tokenized inputs without padding.
        """

        batch_size = len(batched_inputs)
        _to = dict(dtype=torch.long, device=self.device)
        if device is not None:
            _to["device"] = torch.device(device)

        # tokenize the postfix like a postfix. this is useful to handle tokenizers like llama
        if prefix_cache is not None and len(prefix_cache.last_tokens) == batch_size:
            batched_inputs = [l + p for l, p in zip(prefix_cache.last_tokens, batched_inputs)]
        elif add_dummy_prefix:
            batched_inputs = [dummy_prefix + p for p in batched_inputs]

        # use the same tokenizer, but different padding strategy
        batched_encodings = self.tokenizer(batched_inputs)

        # remove the prefix from the input_ids and get the batched_ids for postfix
        if prefix_cache is not None and prefix_cache.last_tokens is not None:
            ids_slice = [
                slice(batched_encodings.char_to_token(i, len(l)), self.tokenizer.model_max_length)
                for i, l in enumerate(prefix_cache.last_tokens)
            ]
        elif add_dummy_prefix:
            char_index = len(dummy_prefix)
            ids_slice = [
                slice(batched_encodings.char_to_token(i, char_index), self.tokenizer.model_max_length)
                for i in range(batch_size)
            ]
        else:
            ids_slice = [slice(0, self.tokenizer.model_max_length)] * batch_size
        batched_ids = [i[slc] for i, slc in zip(batched_encodings["input_ids"], ids_slice)]
        input_lengths = [len(seq) for seq in batched_ids]
        max_input_len = max(input_lengths)
        if not padding:
            return batched_ids

        # pad the input_ids and attention_mask
        input_ids = torch.full((batch_size, max_input_len), self.tokenizer.pad_token_id, **_to)
        attention_mask = torch.zeros((batch_size, max_input_len), **_to)
        for i, ids in enumerate(batched_ids):
            input_ids[i, :len(ids)] = torch.tensor(ids, **_to)
            attention_mask[i, :len(ids)] = 1

        if prefix_cache is not None:
            attention_mask, input_pos, prefix_lengths = self._process_postfix_encodings(attention_mask, prefix_cache)
        else:
            prefix_lengths = [0] * batch_size
            input_pos = None

        return input_ids, attention_mask, input_pos, prefix_lengths, input_lengths

    def get_cache(
        self,
        batched_inputs: List[str],
        prefix_cache: Optional[SequenceCache] = None,
        return_caches: bool = True,
        save_last_logits: bool = False,
    ) -> Union[List[SequenceCache], Tuple[torch.Tensor, torch.Tensor, List[int]]]:
        """
        Return:
            `return_caches` is True:
                caches (`List[SequenceCache]`): A list of caches for each prefix and input pair without padding. At the same device as `self.device`.
            `return_caches` is False:
                logits (`torch.Tensor`): Logits of batched inputs. At the same device as `self.device`.
                input_ids (`torch.Tensor`): A tensor of input_ids of batched inputs. At the same device as `self.device`.
                input_lengths (`List[int]`): The number of non-padding tokens in each input.
        """
        batch_size = len(batched_inputs)
        if prefix_cache is not None:
            cache_num = prefix_cache.get_seq_num()
            if cache_num != batch_size and cache_num != 1:
                raise RuntimeError(
                    f"The number of sentence in prefix_cache should be one or be equal to the batch size {batch_size}"
                )

        input_ids, attention_mask, input_pos, prefix_lengths, input_lengths = self._tokenize_postfix(
            batched_inputs, prefix_cache
        )
        if prefix_cache is not None:
            prefix_cache = prefix_cache.to_legacy_cache()  # type: ignore

        with torch.no_grad():
            results = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=input_pos,
                past_key_values=prefix_cache,
                use_cache=True,
            )
            logits = results.logits.detach()

        if return_caches:
            # store the non-padding parts of caches to ensure the correct creation of position_ids when using
            # these caches in the future
            max_prefix_len = max(prefix_lengths)
            max_input_len = input_ids.size(1)
            caches = SequenceCache.from_legacy_cache(results.past_key_values).split_by_seq()
            for idx, seq_cache in enumerate(caches):
                seq_cache.remove_paddings(
                    num_l=max_prefix_len - prefix_lengths[idx],
                    num_r=max_input_len - input_lengths[idx],
                )
                if save_last_logits:
                    p = input_lengths[idx]
                    seq_cache.set_last_logits(logits[idx:idx + 1, p - 1:p, :].clone())
                seq_cache.set_last_tokens(batched_inputs[idx].rsplit(" ", 1)[-1])
            return caches
        else:
            return logits, input_ids, input_lengths

    def get_ppl_with_cache(self,
                           batched_targets: List[str],
                           prefix_cache: SequenceCache,
                           exact_match: bool = False) -> List[Tuple[float, int]]:
        logits, labels, input_lengths = self.get_cache(batched_targets, prefix_cache, return_caches=False)
        last_logits = torch.cat(prefix_cache.last_logits, dim=0).to(logits.device)
        shift_logits = torch.cat([last_logits, logits[:, :-1]], dim=-2)
        labels[labels == self.tokenizer.pad_token_id] = -100
        probs = self.loss_fct(shift_logits.view(-1, self.model.config.vocab_size),
                              labels.view(-1)).view(labels.size(0), -1)

        if exact_match:
            greedy_tokens = torch.argmax(shift_logits, dim=-1)
            ppls = []
            for idx, tgt_len in enumerate(input_lengths):
                if greedy_tokens[idx, :tgt_len].eq(labels[idx, :tgt_len]).all():
                    ppl = 0  # exact-match
                else:
                    ppl = probs[idx, :tgt_len].sum().item()
                ppls.append((ppl, tgt_len))
        else:
            ppls = [(prob[:tgt_len].sum().item(), tgt_len) for prob, tgt_len in zip(probs, input_lengths)]
        return ppls

    def set_ppl_args(self, **extra_model_args):
        r"""Set the configurations for PPL score calculation. This is useful because different datasets may have different requirements for ppl calculation."""
        self.loss_fct = CrossEntropyLoss(reduction="none")
        self._ppl_args_set = True
        if len(extra_model_args) > 0:
            logger.warning(f"Unused generation arguments: {extra_model_args}")

    def get_ppl(self,
                batched_inputs: List[Tuple[str, ...]],
                use_cache: bool = True,
                exact_match: bool = False) -> List[Tuple[float, int]]:

        if use_cache and self.cacher is not None:
            # grouped_prefixes: a list of batched substrings without the last group (target text) with shape [GroupNum - 1, BatchSize]
            *grouped_prefixes, targets = list(map(list, zip(*batched_inputs)))
            cache_level = len(grouped_prefixes)

            # if cache is available, get_ppl_with_cache
            prefix_cache, cached_num = self.cacher.get_cache()
            if prefix_cache is not None and cached_num == cache_level:
                self.cacher.step()
                return self.get_ppl_with_cache(targets, prefix_cache, exact_match)

            # pass the input without prefix text to the model
            prefix_cache = self.get_cache(
                grouped_prefixes[cached_num], prefix_cache, save_last_logits=cached_num == cache_level - 1
            )
            self.cacher.set_cache(prefix_cache)
            self.cacher.step()
            return []

        prompt = ["".join(seq_tuple) for seq_tuple in batched_inputs]

        batched_encodings = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        ).to(device=self.device)

        with torch.no_grad():
            logits = self.model(
                input_ids=batched_encodings["input_ids"], attention_mask=batched_encodings["attention_mask"]
            ).logits
            shift_logits = logits.detach()[:, :-1].contiguous()
            shift_labels = batched_encodings["input_ids"][:, 1:].contiguous()
            shift_labels[shift_labels == self.tokenizer.pad_token_id] = -100
            probs = self.loss_fct(shift_logits.view(-1, self.model.config.vocab_size),
                                  shift_labels.view(-1)).view(shift_labels.size(0), -1)

        src_lengths = [len("".join(pg[:-1])) for pg in batched_inputs]
        # tgt_starts = [batched_encodings.char_to_token(i, l) for i, l in enumerate(src_lengths)]
        offsets = [list(map(lambda x: x[0], offset)) for offset in batched_encodings["offset_mapping"]]
        tgt_starts = [
            max(offset.index(src_len),
                attention_mask.nonzero()[0][0].item() + 1)
            for src_len, offset, attention_mask in zip(src_lengths, offsets, batched_encodings["attention_mask"])
        ]
        ed = len(batched_encodings["input_ids"][0])

        if exact_match:
            ppls = []
            greedy_tokens = torch.argmax(shift_logits, dim=-1)
            for idx, st in enumerate(tgt_starts):
                if greedy_tokens[idx, st - 1:].eq(shift_labels[idx, st - 1:]).all():
                    ppl = 0
                else:
                    ppl = probs[idx, st - 1:].sum().item()
                ppls.append((ppl, ed - st))
        else:
            ppls = [(prob[st - 1:].sum().item(), ed - st) for prob, st in zip(probs, tgt_starts)]
        return ppls

    def set_generation_args(self, **extra_model_args):
        generation_kwargs = {}
        for key in [
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "best_of",
            "repetition_penalty",
            "length_penalty",
            "early_stopping",
            "no_repeat_ngram_size",
            "stop",
        ]:
            # ModelArguments > extra_model_args
            value = getattr(self.args, key, None)
            if value is None:
                value = extra_model_args.pop(key, None)

            if key == "max_tokens" and value is None:
                value = 1024
            if value is not None:
                if key == "max_tokens":
                    generation_kwargs["max_new_tokens"] = value
                elif key == "best_of":
                    generation_kwargs["num_beams"] = value
                elif key == "temperature":
                    if value > 0:
                        generation_kwargs["temperature"] = value
                        generation_kwargs["do_sample"] = True
                    else:
                        generation_kwargs["do_sample"] = False
                else:
                    generation_kwargs[key] = value

        generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        self.generation_kwargs = generation_kwargs

        self._generation_args_set = True
        if len(extra_model_args) > 0:
            logger.warning(f"Unused generation arguments: {extra_model_args}")

    def generation_with_cache(
        self,
        batched_inputs: List[str],
        prefix_cache: SequenceCache,
    ) -> List[str]:

        batched_encodings = self.tokenizer(
            batched_inputs,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)

        batch_outputs = self.model.generate(**batched_encodings, **self.generation_kwargs)
        self.generation_kwargs["stopping_criteria"][0].step()

        self._process_generation_results()

    def generation(self, batched_inputs: Union[List[str], List[Tuple[str, ...]]], use_cache: bool = False) -> List[str]:
        """Generate the response of given question for this batch.

        Returns:
            List[str]: The list of generation results.
        """
        if isinstance(batched_inputs[0], str):
            prompts = batched_inputs
        else:
            grouped_prompts = list(map(list, zip(*batched_inputs)))
            prompts = ["".join(pg[i] for pg in grouped_prompts) for i in range(len(grouped_prompts[0]))]
            cache_level = len(grouped_prompts) - 1

        if use_cache and self.cacher is not None:
            # if cache is available, generation_with_cache
            prefix_cache, cached_num = self.cacher.get_cache()
            if prefix_cache is not None and cached_num == cache_level:
                self.cacher.step()
                return self.generation_with_cache(grouped_prompts[-1], prefix_cache)

            # pass the input without prefix text to the model
            prefix_cache = self.get_cache(
                grouped_prompts[cached_num], prefix_cache, save_last_logits=cached_num == cache_level - 1
            )
            self.cacher.set_cache(prefix_cache)
            self.cacher.step()
            return []

        batched_encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        max_input_length = batched_encodings["input_ids"].size(1)

        batch_outputs = self.model.generate(**batched_encodings, **self.generation_kwargs)
        self.generation_kwargs["stopping_criteria"][0].step()

        batch_outputs = batch_outputs[:, max_input_length:]
        answers = self._process_generation_results(batch_outputs)
        return answers

    def _process_generation_results(self, batch_outputs: torch.Tensor) -> List[str]:
        """Remove the sequences after the `stop_id_sequences` and decode to strings."""
        max_output_length = batch_outputs.size(1)
        if getattr(self, "stop_id_sequences", None) is not None:
            for seq_idx in range(batch_outputs.size(0)):
                for token_idx in range(max_output_length):
                    if any(
                        batch_outputs[seq_idx, token_idx:token_idx + len(s)].tolist() == s
                        for s in self.stop_id_sequences
                    ):
                        batch_outputs[seq_idx, token_idx:] = self.tokenizer.pad_token_id
                        break

        answers = self.tokenizer.batch_decode(
            batch_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return answers

    def set_prob_args(self, **extra_model_args):
        self._token_labels = []
        self._word_labels = []
        self._candidate_ids = extra_model_args.pop("candidate_ids", None)
        self.constant_option_num = extra_model_args.pop("constant_option_num", False)

        self._prob_args_set = True
        if len(extra_model_args) > 0:
            logger.warning(f"Unused generation arguments: {extra_model_args}")

    def _get_label_ids(self, option_num: Optional[int]) -> List[int]:
        """Return the tokenized labels of options."""
        if option_num is not None:
            if len(self._token_labels) < option_num:
                labels = [chr(i + 65) for i in range(len(self._token_labels), option_num)]
                self._word_labels.extend([self.tokenizer.encode(l, add_special_tokens=False)[0] for l in labels])
                self._token_labels.extend([self.tokenizer.convert_tokens_to_ids(l) for l in labels])
            return self._word_labels[:option_num] + self._token_labels[:option_num]
        else:
            if self._candidate_ids is None:
                raise ValueError("The candidate_ids must be provided when option_num is None.")
            return self._candidate_ids

    def get_prob_with_cache(
        self,
        batched_inputs: List[str],
        batched_option_nums: List[int],
        prefix_cache: SequenceCache,
    ) -> List[List[float]]:
        logits, _, input_lengths = self.get_cache(batched_inputs, prefix_cache, return_caches=False)
        input_lengths = [i - 1 for i in input_lengths]
        logits = logits[range(len(input_lengths)), input_lengths, :]

        answers = []
        if self.constant_option_num:
            label_ids = self._get_label_ids(batched_option_nums[0])
            answers = torch.softmax(logits[:, label_ids], dim=-1, dtype=torch.float32).tolist()
        else:
            for i, option_num in enumerate(batched_option_nums):
                label_ids = self._get_label_ids(option_num)
                answers.append(torch.softmax(logits[i, label_ids], dim=-1, dtype=torch.float32).tolist())
        return answers

    def get_prob(self, batched_inputs: _InputsWithOptionNum, use_cache: bool = True) -> List[List[float]]:

        if len(batched_inputs[0]) <= 2:
            batched_prompts, batched_option_nums = map(list, zip(*batched_inputs))
        else:
            # batched_groups: a batch of concatenated input strings
            # grouped_prompts: a list of batched substrings with shape [GroupNum, BatchSize]
            *grouped_prompts, batched_option_nums = map(list, zip(*batched_inputs))
            batched_prompts = ["".join(seq_tuple[:-1]) for seq_tuple in batched_inputs]
            cache_level = len(grouped_prompts) - 1

        if self.cacher is not None and use_cache:
            # if cache is available, get_prob_with_cache
            prefix_cache, cached_num = self.cacher.get_cache()
            if prefix_cache is not None and cached_num == cache_level:
                self.cacher.step()
                return self.get_prob_with_cache(grouped_prompts[-1], batched_option_nums, prefix_cache)

            # pass the input without prefix text to the model
            prefix_cache = self.get_cache(grouped_prompts[cached_num], prefix_cache, save_last_logits=False)
            self.cacher.set_cache(prefix_cache)
            self.cacher.step()
            return []

        batched_encodings = self.tokenizer(
            batched_prompts,
            padding="longest",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            batch_logits = self.model(
                input_ids=batched_encodings["input_ids"].to(self.device),
                attention_mask=batched_encodings["attention_mask"].to(self.device),
            ).logits.detach()[:, -1]  # padding_side="left" in tokenizer

        if self.constant_option_num:
            label_ids = self._get_label_ids(batched_option_nums[0])
            answers = torch.softmax(batch_logits[:, label_ids], dim=-1, dtype=torch.float32).tolist()
        else:
            answers = []
            for i, option_num in enumerate(batched_option_nums):
                label_ids = self._get_label_ids(option_num)
                answers.append(torch.softmax(batch_logits[i, label_ids], dim=-1, dtype=torch.float32).tolist())
        return answers
