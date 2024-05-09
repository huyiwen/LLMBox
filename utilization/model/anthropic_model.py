import os
from logging import getLogger

import anthropic
from anthropic.types import Message

from ..utils import ModelArguments
from .enum import ANTHROPIC_CHAT_COMPLETIONS_ARGS
from .model import ApiModel

logger = getLogger(__name__)

CHAT_COMPLETIONS_ARGS_ALIASES = {"stop": "stop_sequences"}


class Anthropic(ApiModel):
    r"""The model for calling Anthropic APIs.

    Please refer to https://docs.anthropic.com/claude/reference.

    We now support Claude (`claude-2.1`) and Claude Instant (`claude-instant-1.2`).
    """

    model_backend = "anthropic"
    model: anthropic.Anthropic

    _retry_errors = (anthropic.APITimeoutError, anthropic.InternalServerError, anthropic.RateLimitError)
    _raise_errors = (
        anthropic.APIConnectionError, anthropic.AuthenticationError, anthropic.BadRequestError, anthropic.ConflictError,
        anthropic.NotFoundError, anthropic.PermissionDeniedError, anthropic.UnprocessableEntityError
    )

    _repr = ["model_type", "model_backend", "multi_turn"]

    def __init__(self, args: ModelArguments):
        super().__init__(args)

        base_url = os.getenv("ANTHROPIC_BASE_URL", None)
        logger.info(f"Trying to load Anthropic model with ANTHROPIC_BASE_URL='{base_url}'")

        self.model = anthropic.Anthropic(api_key=args.anthropic_api_key, base_url=base_url)

    def _chat_completions(self, **kwargs):
        return self.model.messages.create(**kwargs)

    @staticmethod
    def _get_assistant(msg: Message) -> str:
        return msg.content[0].text
