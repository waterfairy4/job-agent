"""Fake models so tests run offline, fast and free."""
from langchain_core.messages import AIMessage


class FakeLLM:
    """Plays a scripted list of agent replies, and answers structured-output calls
    with a function per schema: handlers[SchemaClass](prompt) -> SchemaClass instance."""

    def __init__(self, script: list, handlers: dict):
        self.script = list(script)
        self.handlers = handlers
        self.prompts: list[str] = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        nxt = self.script.pop(0)
        return nxt(messages) if callable(nxt) else nxt

    def with_structured_output(self, schema, **kwargs):
        outer = self

        class _Structured:
            def invoke(self, prompt):
                outer.prompts.append(prompt)
                return outer.handlers[schema](prompt)

        return _Structured()


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])
