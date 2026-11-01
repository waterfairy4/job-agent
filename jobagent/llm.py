"""The chat model. Any OpenAI-compatible API that supports tool calling works."""
from jobagent import config


def get_llm():
    from langchain_openai import ChatOpenAI

    if not config.LLM_API_KEY:
        raise RuntimeError("Set LLM_API_KEY in your .env file (see .env.example).")
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=0,  # stable outputs: easier to test and evaluate
    )


def structured(llm, schema):
    """LLM that returns a validated Pydantic object.

    method="function_calling" works with DeepSeek and Qwen; the newer "json_schema"
    mode is not supported by every OpenAI-compatible API.
    """
    return llm.with_structured_output(schema, method="function_calling")
