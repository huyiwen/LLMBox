from logging import getLogger
from typing import Optional, Type

import dashscope
from dashscope.api_entities.dashscope_response import GenerationResponse

from ..utils import ModelArguments
from .model import ApiModel

logger = getLogger(__name__)


class SkipError(Exception):
    pass


class RetryError(Exception):
    pass


class RaiseError(Exception):
    pass


class Dashscope(ApiModel):
    r"""The model for calling Dashscope APIs. (Aliyun)

    Please refer to https://help.aliyun.com/zh/dashscope/.

    We now support `qwen-turbo`, `qwen-plus`, `qwen-max`, `qwen-max-1201`, `qwen-max-longcontext`, `qwen1.5-72b-chat`,
                   `qwen1.5-14b-chat`, `qwen1.5-7b-chat`, `qwen-72b-chat`, `qwen-14b-chat`, `qwen-7b-chat`,
                   `qwen-1.8b-longcontext-chat`, `qwen-1.8b-chat`.
    """

    model_backend = "dashscope"
    model: Type[dashscope.Generation]

    _raise_errors = (RaiseError,)
    _retry_errors = (RetryError,)
    _skip_errors = (SkipError,)

    _repr = ["model_type", "model_backend", "multi_turn"]

    def __init__(self, args: ModelArguments):
        super().__init__(args)
        if not args.dashscope_api_key:
            raise ValueError(
                "Dashscope API key is required. Please set it by passing a `--dashscope_api_key` or through environment variable `DASHSCOPE_API_KEY`."
            )
        logger.info(f"Trying to load Dashscope model")
        self.api_key = args.dashscope_api_key
        self.model = dashscope.Generation

        self.tokenizer = dashscope.get_tokenizer(args.tokenizer_name_or_path)

    def _chat_completions(self, **kwargs) -> GenerationResponse:
        return self.model.call(**kwargs)

    @staticmethod
    def _get_error_type(response: GenerationResponse) -> Optional[Exception]:
        if response.status_code == 200:
            return None
        elif response.status_code == 400:
            # Input or output data may contain inappropriate content.
            if "inappropriate" in response.message:
                return SkipError
            elif "in good standing" in response.message:
                return RetryError
        elif response.status_code in {408, 429, 500}:
            return RetryError
        else:
            return RaiseError

    @staticmethod
    def _get_assistant(msg: GenerationResponse) -> str:
        return msg.output.choices[0].message.content
