"""The chat model. Any OpenAI-compatible API that supports tool calling works."""
from jobagent import config


def get_llm(model: str | None = None):
    from langchain_openai import ChatOpenAI

    if not config.LLM_API_KEY:
        raise RuntimeError("Set LLM_API_KEY in your .env file (see .env.example).")
    extra = {}
    if "deepseek" in config.LLM_BASE_URL:
        # DeepSeek models think by default. With tools, thinking mode requires sending the
        # model's reasoning back on every request, which LangChain's OpenAI client does not do
        # (-> HTTP 400). Non-thinking mode is also faster and cheaper for this workload.
        extra["extra_body"] = {"thinking": {"type": config.LLM_THINKING}}
    return ChatOpenAI(
        model=model or config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=0,  # stable outputs: easier to test and evaluate
        **extra,
    )


def structured(llm, schema):
    """LLM that returns a validated Pydantic object.

    method="function_calling" works with DeepSeek and Qwen; the newer "json_schema"
    mode is not supported by every OpenAI-compatible API.
    """
    return _Retrying(llm.with_structured_output(schema, method="function_calling"), schema.__name__)


class _Retrying:
    """Sometimes the model answers in plain text instead of calling the output function, and LangChain
    then returns None. Ask again (up to 3 tries) and fail with a clear message instead of a crash later."""

    def __init__(self, runnable, name: str, tries: int = 3):
        self.runnable, self.name, self.tries = runnable, name, tries

    def invoke(self, prompt):
        for _ in range(self.tries):
            out = self.runnable.invoke(prompt)
            if out is not None:
                return out
        raise RuntimeError(f"The model did not return a valid {self.name} after {self.tries} tries. "
                           f"Try again, or use a stronger model for this step (e.g. CV_MODEL in .env).")
