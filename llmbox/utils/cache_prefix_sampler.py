from logging import getLogger
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from cyac import Trie
from torch.utils.data.sampler import Sampler
from transformers import DynamicCache

_LegacyCache = Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], ...]

logger = getLogger(__name__)


class SequenceCache(DynamicCache):
    """A cache that supports some sequence level operations."""

    def __init__(self) -> None:
        # keeps cache in a list instead of a stacked tensor because the tensor may on different devices
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        self.seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.last_logits: List[torch.Tensor] = []  # used in `get_ppl` to concatenate logits
        self.last_tokens: List[str] = []  # used in `get_cache` to concatenate tokens
        self.real_seq_length: List[int] = []
        self.cache_level = None

    def set_last_tokens(self, last_tokens: Union[str, List[str]]):
        if isinstance(last_tokens, str):
            last_tokens = [last_tokens]

        if len(last_tokens) != self.get_seq_num():
            raise ValueError(
                f"last_tokens ({len(last_tokens)}) should be a list of strings with the same length as the cache ({self.get_seq_num()})"
            )

        self.last_tokens = last_tokens

    def set_last_logits(self, last_logits: Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(last_logits, torch.Tensor):
            last_logits = [last_logits]

        if len(last_logits) != self.get_seq_num():
            raise ValueError(
                f"last_logits ({len(last_logits)}) should be a list of tensors with the same length as the cache ({self.get_seq_num()})"
            )
        self.last_logits = last_logits

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[_LegacyCache] = None) -> "SequenceCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = SequenceCache()
        if past_key_values is not None:
            batch_size, _, seq_len, _ = past_key_values[0][0].shape
            for key_states, value_states in past_key_values:
                cache.key_cache.append(key_states.detach())
                cache.value_cache.append(value_states.detach())
            cache.real_seq_length = [seq_len] * batch_size
        return cache

    def get_seq_num(self) -> int:
        return len(self.real_seq_length)

    def remove_paddings(self, num_l: int = 0, num_r: int = 0):
        if num_l + num_r > 0:
            self.real_seq_length = [l - num_l - num_r for l in self.real_seq_length]
            for layer_idx in range(len(self.key_cache)):
                self.key_cache[layer_idx] = self.key_cache[layer_idx][..., num_l:-num_r, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][..., num_l:-num_r, :]
            self.seen_tokens = self.seen_tokens - num_l - num_r
        # logger.warning(f"Remove paddings: {num_l}, {num_r}, {self.real_seq_length}")

    def split_by_seq(self) -> List["SequenceCache"]:
        results = []
        for seq_idx in range(self.get_seq_num()):
            cache = SequenceCache()
            cache.real_seq_length = [self.real_seq_length[seq_idx]]
            if len(self.last_logits) > seq_idx:
                cache.last_logits = [self.last_logits[seq_idx]]
            if len(self.last_tokens) > seq_idx:
                cache.last_tokens = [self.last_tokens[seq_idx]]
            cache.key_cache, cache.value_cache = self._apply_cache(lambda x: x[seq_idx:seq_idx + 1, ...].clone())
            results.append(cache)
        return results

    def _apply_cache(self, fn) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        applied = [(fn(key), fn(value)) for key, value in zip(self.key_cache, self.value_cache)]
        key_list, value_list = map(list, zip(*applied))
        return key_list, value_list

    def expand_seq(self, repeat_times: int) -> "SequenceCache":
        assert self.get_seq_num() == 1, "SequenceCache can only repeat sequence when it contains only one sequence"

        cache = SequenceCache()
        cache.seen_tokens = self.seen_tokens
        cache.last_logits = self.last_logits * repeat_times
        cache.last_tokens = [self.last_tokens[0]] * repeat_times  # repeat a list of strings
        cache.real_seq_length = self.real_seq_length * repeat_times
        for key, value in zip(self.key_cache, self.value_cache):
            cache.key_cache.append(key.expand(repeat_times, -1, -1, -1))
            cache.value_cache.append(value.expand(repeat_times, -1, -1, -1))
        return cache

    @classmethod
    def pad_and_stack(cls, seq_caches: Sequence["SequenceCache"]) -> "SequenceCache":
        if len(seq_caches) == 1:
            return seq_caches[0]

        cache = SequenceCache()
        for sc in seq_caches:
            cache.last_logits.extend(sc.last_logits)
            cache.last_tokens.extend(sc.last_tokens)
            cache.real_seq_length.extend(sc.real_seq_length)
        max_seq_len = max(cache.real_seq_length)
        max_layer_idx = len(seq_caches[0].key_cache)
        cache.seen_tokens = max_seq_len
        # logger.warning(f"Pad and stack: {max_seq_len}, {max_layer_idx}, {cache.real_seq_length}")

        for layer_idx in range(max_layer_idx):
            key_list = []
            value_list = []
            for sc in seq_caches:
                kv_shape = sc.key_cache[0].shape
                if sc.get_seq_length() < max_seq_len:
                    padding = torch.zeros(
                        kv_shape[:-2] + (max_seq_len - sc.get_seq_length(), kv_shape[-1]),
                        device=sc.key_cache[layer_idx].device,
                        dtype=sc.key_cache[layer_idx].dtype
                    )
                    key_list.append(torch.cat((padding, sc.key_cache[layer_idx]), dim=-2))
                    value_list.append(torch.cat((padding, sc.value_cache[layer_idx]), dim=-2))
                else:
                    key_list.append(sc.key_cache[layer_idx])
                    value_list.append(sc.value_cache[layer_idx])
            cache.key_cache.append(torch.cat(key_list, dim=0))
            cache.value_cache.append(torch.cat(value_list, dim=0))
        return cache

    def __repr__(self) -> str:
        return f"SequenceCache(real_seq_length={self.real_seq_length}, last_tokens={self.last_tokens}, last_logits=[{repr(self.last_logits[0].shape) + '] * ' + str(len(self.last_logits)) if self.last_logits else ']'})"


class Cacher:
    """A base class that supports caching for a list of sources."""

    def get_cache(self) -> Tuple[Optional[SequenceCache], int]:
        raise NotImplementedError

    def set_cache(self, caches: List[SequenceCache]):  # -> Any:# -> Any:
        raise NotImplementedError

    def step(self):
        raise NotImplementedError


class CachePrefixSampler(Sampler[List[int]], Cacher):
    """A sampler that facilitates key-value caching for a list of text segments."""

    def __init__(
        self,
        data: Sequence[Tuple[str, ...]],
        batch_size: int,
    ):
        print("!!")
        self.data = data
        self.data_idx = 0

        # split data into (src,) and (src, tgt)
        self.total_prefix_num = len([1 for i in self.data[0] if isinstance(i, str)])
        self.joined_data = [[] for _ in range(self.total_prefix_num)]
        self.cache_levels = [0] * len(self.data)
        cache_batch_size = (batch_size + 1) // 2
        self.cache_batch_size = [cache_batch_size] * (self.total_prefix_num - 1) + [batch_size]

        self.cache: Dict[Tuple[int, int], SequenceCache] = dict()

        self.next_data_idx = [dict() for _ in range(self.total_prefix_num)]
        last_start_idx = [0 for _ in range(self.total_prefix_num)]
        data_len = len(self.data)
        for s_idx, src in enumerate(self.data):
            for p_idx in range(self.total_prefix_num):
                joined_src = "".join(src[:p_idx + 1])
                self.joined_data[p_idx].append(joined_src)
                if s_idx > 0 and joined_src != self.joined_data[p_idx][s_idx - 1]:
                    if last_start_idx[p_idx] + 1 != s_idx:
                        self.next_data_idx[p_idx][last_start_idx[p_idx]] = s_idx
                    last_start_idx[p_idx] = s_idx
        for p_idx in range(self.total_prefix_num):
            if last_start_idx[p_idx] + 1 != data_len:
                self.next_data_idx[p_idx][last_start_idx[p_idx]] = data_len

        self.data_order_with_cache: List[List[int]] = []
        self.data_cache_level = []
        is_cache_ready = [0 for _ in range(self.total_prefix_num)]
        self.order_idx_by_cache = [None] * self.total_prefix_num
        for data_idx in range(data_len):
            for i in range(self.total_prefix_num):
                if is_cache_ready[i] <= data_idx:
                    if self.order_idx_by_cache[i] is None:
                        self.data_order_with_cache.append([])
                        self.data_cache_level.append(i)

                        self.order_idx_by_cache[i] = len(self.data_order_with_cache) - 1

                    self.data_order_with_cache[self.order_idx_by_cache[i]].append(data_idx)

                    is_cache_ready[i] = self.next_data_idx[i].get(data_idx, data_idx + 1)
                    for j in range(i + 1, self.total_prefix_num):
                        if self.order_idx_by_cache[j] is not None and self.order_idx_by_cache[
                            j] < self.order_idx_by_cache[i]:
                            self.order_idx_by_cache[j] = None
                    if len(self.data_order_with_cache[self.order_idx_by_cache[i]]) == self.cache_batch_size[i]:
                        self.order_idx_by_cache[i] = None

        print(self.data_order_with_cache[:40])
        print(self.data_cache_level, data_len, self.total_prefix_num)
        print(self.next_data_idx)

    def __len__(self):
        return len(self.data_cache_level)

    def get_cache(self) -> Tuple[Optional[SequenceCache], int]:
        """Get cache for a list of sources. Return None if any source is not cached.

        Return:
            cache (`SequenceCache`): The (left padded) cache for the sources.
            cache_level (`int`): The number of prefixes that are matched in the cache.
        """

        cache_level = self.data_cache_level[self.data_idx]
        if cache_level == 0:
            return None, 0

        def lower_bound(i, l) -> Tuple[int, int]:
            max_k = 0
            for k, _ in self.next_data_idx[l].items():
                if k < i:
                    max_k = max(max_k, k)
            return max_k, self.next_data_idx[l].get(max_k, max_k + 1)

        # logger.warning(
        #     f"Get cache: {self.data_idx} {self.data_cache_level[self.data_idx]} {self.data_order_with_cache[self.data_idx]} "
        # )

        caches = []
        last_cache_count = 1
        last_cache_st = -1
        last_cache_ed = 0
        last_cache: Optional[SequenceCache] = None
        for i in self.data_order_with_cache[self.data_idx]:
            # get the `cache_level - 1` cache for the current data
            if last_cache_ed <= i:
                if last_cache is not None:
                    # if i goes out of the range of the last cache, we pop the last cache
                    caches.append(last_cache.expand_seq(last_cache_count))
                    # print("del", cache_level - 1, last_cache_st)
                    del self.cache[(cache_level - 1, last_cache_st)]

                last_cache_st = i
                last_cache_ed = self.next_data_idx[cache_level - 1].get(last_cache_st, None)
                if last_cache_ed is None:
                    if cache_level == self.total_prefix_num:
                        last_cache_ed = i + 1
                    else:
                        last_cache_st, last_cache_ed = lower_bound(i, cache_level - 1)
                last_cache_count = 1
                # print(i, cache_level - 1, last_cache_st, last_cache_ed, last_cache_count)
                last_cache = self.cache[(cache_level - 1, last_cache_st)]
            else:
                last_cache_count += 1
        caches.append(last_cache.expand_seq(last_cache_count))

        # logger.warning(f"{caches}")

        return SequenceCache.pad_and_stack(caches), cache_level

    def set_cache(self, caches: List[SequenceCache]):

        # logger.warning(
        #     f"Set cache: {self.data_idx} {self.data_cache_level[self.data_idx]} {self.data_order_with_cache[self.data_idx]}"
        # )

        cache_level = self.data_cache_level[self.data_idx]
        for i, cache in zip(self.data_order_with_cache[self.data_idx], caches):
            self.cache[(cache_level, i)] = cache
            # logger.warning(f"Set cache: {cache_level} {i}")

    def step(self):
        # print("step", self.data_idx)
        self.data_idx += 1

    def __iter__(self) -> Iterator[List[int]]:
        print("!")
        self.data_idx = 0
        yield from self.data_order_with_cache

    def __repr__(self) -> str:
        return f"CachePrefixSampler(cache_batch_size={self.cache_batch_size}, total_prefix_num={self.total_prefix_num})"
