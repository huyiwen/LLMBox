OPENAI_COMPLETION_MODELS = ["babbage-002", "davinci-002", "gpt-3.5-turbo-instruct"]
OPENAI_CHAT_MODELS = ["gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-16k", "gpt-4"]
OPENAI_MODELS = OPENAI_COMPLETION_MODELS + OPENAI_CHAT_MODELS
OPENAI_INSTRUCTION_MODELS = ["gpt-3.5-turbo-instruct"] + OPENAI_CHAT_MODELS

# https://docs.anthropic.com/claude/docs/models-overview
ANTHROPIC_CLAUDE3_MODELS = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-opus-20240229"]
ANTHROPIC_LEGACY_MODELS = ["claude-2.1", "claude-instant-1.2"]
ANTHROPIC_MODELS = ANTHROPIC_CLAUDE3_MODELS + ANTHROPIC_LEGACY_MODELS
