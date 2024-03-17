import time
from logging import getLogger
from typing import List, Literal, Tuple

import anthropic
import tiktoken

from ..utils import ModelArguments
from .model import Model

logger = getLogger(__name__)


class Anthropic(Model):
    r"""The model for calling Anthropic APIs.

    Please refer to https://docs.anthropic.com/claude/reference.

    We now support Claude (`claude-2.1`) and Claude Instant (`claude-instant-1.2`).
    """

    def __init__(self, args: ModelArguments):
        super().__init__(args)
        if not args.anthropic_api_key:
            raise ValueError(
                "Anthropic API key is required. Please set it by passing a `--anthropic_api_key` or through environment variable `ANTHROPIC_API_KEY`."
            )
        private_key = args.anthropic_api_key[:8] + "*" * 39 + args.anthropic_api_key[-4:]
        logger.info(f"Trying to load Anthropic model with api_key='{private_key}'")
        self.api_key = args.anthropic_api_key

        self.args = args
        self.name = args.model_name_or_path
        self.type = "instruction"
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_try_times = 5

    def set_generation_args(self, **extra_model_args):
        r"""Set the configurations for open-ended generation. This is useful because different datasets may have different requirements for generation."""
        generation_kwargs = {}
        for key in ["temperature", "top_p", "max_tokens", "best_of", "stop"]:
            # ModelArguments > extra_model_args
            value = getattr(self.args, key, None)
            if value is None:
                value = extra_model_args.get(key, None)

            if key == "max_tokens" and value is None:
                value = 4096
            if key == "stop":
                key = "stop_sequences"
            if value is not None:
                generation_kwargs[key] = value
        self.generation_kwargs = generation_kwargs

    def generation(self, batched_inputs: List[str]):
        results = self.request(batched_inputs, self.generation_kwargs)
        answers = []
        for result in results:
            answer = result[0].content[0].text
            answers.append(answer)
        return answers

    def set_prob_args(self, **extra_model_args):

        self._word_label_texts = []
        self._token_label_texts = []
        if extra_model_args.pop("candidate_ids", None) is not None:
            logger.warning(f"Anthropic does not support candidate_ids currently, so it will be ignored.")

        self.prob_kwargs = {"max_tokens": 1, "temperature": 0.0}

        if len(extra_model_args) > 0:
            logger.warning(f"Unused generation arguments: {extra_model_args}")

    def _get_label_texts(self, option_num: int) -> List[str]:
        """Return the tokenized labels of options and labels themselves."""
        if len(self._word_label_texts) < option_num:
            for i in range(len(self._word_label_texts), option_num):
                self._word_label_texts.append(chr(i + 65))
                self._token_label_texts.append(" " + chr(i + 65))
        return self._word_label_texts[:option_num] + self._token_label_texts[:option_num]

    def get_prob(
        self,
        batched_inputs: List[Tuple[str, int]],
        use_logit_bias: Literal[False] = False,
        return_real_logprobs: Literal[False] = False
    ) -> List[List[int]]:

        if use_logit_bias or return_real_logprobs:
            logger.warning("Anthropic does not support logit_bias and logprobs currently, so it will be ignored.")

        *batched_prompts, batched_option_nums = map(list, zip(*batched_inputs))
        batch_size = len(batched_prompts[0])
        batched_prompts = ["".join(group[idx] for group in batched_prompts) for idx in range(batch_size)]
        label_texts = [self._get_label_texts(option_num) for option_num in batched_option_nums]

        results = self.request(batched_prompts, self.prob_kwargs)

        answers = []
        for result, option_num, label in zip(results, batched_option_nums, label_texts):
            probs = [-9999.] * (option_num * 2)
            text = result[0].content[0].text
            if text in label:
                probs[label.index(text)] = 20.0
            answers.append(probs)
        return answers

    def request(self, prompt, kwargs) -> List[List[anthropic.types.Message]]:
        r"""Call the Anthropic API.

        Args:
            prompt (List[str]): The list of input prompts.
            model_args (dict): The additional calling configurations.

        Returns:
            List[dict]: The responsed JSON results.
        """
        client = anthropic.Anthropic(api_key=self.api_key)
        for _ in range(self.max_try_times):
            try:
                message = [{"role": "user", "content": prompt[0]}]
                response = client.messages.create(model=self.name, messages=message, **kwargs)
                return [[response]]
            except anthropic.RateLimitError:
                logger.warning("Receive anthropic.RateLimitError, retrying...")
                time.sleep(10)
            except anthropic.APIStatusError as e:
                logger.warning("Another non-200-range status code was received")
                raise e
            except anthropic.APIConnectionError as e:
                logger.warning("The server could not be reached")
                raise e
            except Exception as e:
                logger.warning(f"Receive {e.__class__.__name__}: {str(e)}")
                logger.warning("retrying...")
                time.sleep(1)
        raise ConnectionError("Anthropic API error")
