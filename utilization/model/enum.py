ANTHROPIC_CHAT_COMPLETIONS_ARGS = {"max_tokens", "stop", "system", "temperature", "top_k", "top_p"}
ANTHROPIC_CHAT_COMPLETIONS_ALIASES = {
    "stop": "stop_sequences",
}
ANTHROPIC_CHAT_COMPLETIONS_DEFAULTS = {
    "max_tokens": 4096,
}

DASHSCOPE_CHAT_COMPLETIONS_ARGS = {
    "temperature", "top_p", "top_k", "max_tokens", "repetition_penalty", "enable_search", "stop"
}
DASHSCOPE_CHAT_COMPLETIONS_DEFAULTS = {
    "max_tokens": 1024,
    "temperature": 0.0001,  # can't be 0
}

OPENAI_CHAT_COMPLETIONS_ARGS = {
    "frequency_penalty", "logit_bias", "logprobs", "top_logprobs", "max_tokens", "n", "presence_penalty", "seed",
    "stop", "temperature", "top_p", "best_of"
}
OPENAI_CHAT_COMPLETIONS_DEFAULTS = {
    "max_tokens": 4096,
}

OPENAI_COMPLETIONS_ARGS = {
    "best_of", "echo", "frequency_penalty", "logit_bias", "logprobs", "max_tokens", "n", "presence_penalty", "seed",
    "stop", "temperature", "top_p"
}
OPENAI_COMPLETIONS_DEFAULTS = {
    "max_tokens": 1024,
}

QIANFAN_CHAT_COMPLETIONS_ARGS = {
    "temperature", "top_p", "top_k", "penalty_score", "stop", "disable_search", "enable_citation", "max_tokens"
}
QIANFAN_CHAT_COMPLETIONS_ALIASES = {
    "max_tokens": "max_output_tokens",
}
QIANFAN_CHAT_COMPLETIONS_DEFAULTS = {
    "max_tokens": 1024,
    "temperature": 0.0001,  # can't be 0
}

ENDPOINT_ARGS = {
    "dashscope/chat/completions": (
        DASHSCOPE_CHAT_COMPLETIONS_ARGS,
        {},
        DASHSCOPE_CHAT_COMPLETIONS_DEFAULTS,
    ),
    "anthropic/chat/completions": (
        ANTHROPIC_CHAT_COMPLETIONS_ARGS,
        ANTHROPIC_CHAT_COMPLETIONS_ALIASES,
        ANTHROPIC_CHAT_COMPLETIONS_DEFAULTS,
    ),
    "openai/chat/completions": (
        OPENAI_CHAT_COMPLETIONS_ARGS,
        {},
        OPENAI_CHAT_COMPLETIONS_DEFAULTS,
    ),
    "openai/completions": (
        OPENAI_COMPLETIONS_ARGS,
        {},
        OPENAI_COMPLETIONS_DEFAULTS,
    ),
    "qianfan/chat/completions": (
        QIANFAN_CHAT_COMPLETIONS_ARGS,
        QIANFAN_CHAT_COMPLETIONS_ALIASES,
        QIANFAN_CHAT_COMPLETIONS_DEFAULTS,
    ),
}

API_MODELS = {
    "babbage-002": {
        "endpoint": "completions",
        "model_type": "base",
        "model_backend": "openai"
    },
    "davinci-002": {
        "endpoint": "completions",
        "model_type": "base",
        "model_backend": "openai"
    },
    "gpt-3.5-turbo-instruct": {
        "endpoint": "completions",
        "model_type": "instruction",
        "model_backend": "openai"
    },
    "gpt-3.5-turbo-0125": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "openai"
    },
    "gpt-3.5-turbo": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "openai"
    },
    "gpt-3.5-turbo-1106": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "openai"
    },
    "gpt-3.5-turbo-16k": {
        "endpoint": "completions",
        "model_type": "instruction",
        "model_backend": "openai"
    },
    "gpt-3.5-turbo-0613": {
        "endpoint": "completions",
        "model_type": "instruction",
        "model_backend": "openai"
    },
    "gpt-3.5-turbo-16k-0613": {
        "endpoint": "completions",
        "model_type": "instruction",
        "model_backend": "openai"
    },
    "gpt-4-turbo": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "openai"
    },
    "gpt-4-turbo-2024-04-09": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "openai"
    },
    "gpt-4-turbo-preview": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "openai"
    },
    "gpt-4-0125-preview": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "openai"
    },
    "gpt-4-1106-preview": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "openai"
    },
    "gpt-4-vision-preview": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "openai"
    },
    "gpt-4-1106-vision-preview": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "openai"
    },
    "gpt-4": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "openai"
    },
    "gpt-4-0613": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "openai"
    },
    "gpt-4-32k": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "openai"
    },
    "gpt-4-32k-0613": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "openai"
    },
    "claude-3-opus-20240229": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "anthropic"
    },
    "claude-3-sonnet-20240229": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "anthropic"
    },
    "claude-3-haiku-20240307": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "anthropic"
    },
    "claude-2.1": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "anthropic"
    },
    "claude-2.0": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "anthropic"
    },
    "claude-instant-1.2": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "anthropic"
    },
    "qwen-turbo": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "qwen-plus": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "qwen-max": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "qwen-max-0403": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "qwen-max-0107": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "qwen-max-longcontext": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "qwen-max-0428": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "qwen1.5-110b-chat": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "qwen1.5-72b-chat": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "qwen1.5-32b-chat": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "qwen1.5-14b-chat": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "qwen1.5-7b-chat": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "qwen1.5-1.8b-chat": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "qwen1.5-0.5b-chat": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "codeqwen1.5-7b-chat": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "qwen-72b-chat": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "qwen-14b-chat": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "qwen-7b-chat": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "qwen-1.8b-longcontext-chat": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "qwen-1.8b-chat": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "ernie-4.0-8k": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "ernie-4.0-8k-preview": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "ernie-4.0-8k-0329": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "ernie-4.0-8k-0104": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "ernie-3.5-8k": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "ernie-3.5-8k-0205": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "ernie-3.5-8k-1222": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "ernie-3.5-4k-0205": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "ernie-3.5-8k-preview": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "ernie-3.5-8k-0329": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "ernie-speed-8k": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "ernie-speed-128k": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "ernie-lite-8k-0922": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "ernie-lite-8k-0308": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "ernie-tiny-8k": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "ernie-char-8k": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "ernie-func-8k": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
    "ai_apaas": {
        "endpoint": "chat/completions",
        "model_type": "instruction",
        "model_backend": "dashscope"
    },
}
