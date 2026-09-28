"""Put a schema variant in front of your own agent loop.

The model sees the variant tools. Each call it makes is decoded into native calls,
and those are what your environment executes and what your scorer sees. Protocol
calls (transaction begin/set, resolvers) are answered by the decoder, not the env.

    REMOTE_OPENAI_BASE_URL=http://localhost:8000/v1 MODEL=Qwen/Qwen3-4B-Instruct-2507 \
        python examples/agent_loop.py
"""
import json
import os

from openai import OpenAI

from toolschema.operators import compile_spec
from toolschema.simenv import SimEnv

from quickstart import TOOLS            # the two-tool catalog from quickstart.py

SPEC = {"ops": [{"op": "curry", "tool": "set_window"}]}          # any operator spec
comp = compile_spec(TOOLS, SPEC)
env = SimEnv(comp.call_map, control="hard")    # decodes, answers protocol calls, rejects off-schema calls

client = OpenAI(base_url=os.environ.get("REMOTE_OPENAI_BASE_URL", "http://localhost:8000/v1"),
                api_key=os.environ.get("REMOTE_OPENAI_API_KEY", "EMPTY"))
messages = [{"role": "user", "content": "Vent the front-left window."}]
tools = [{"type": "function", "function": t} for t in comp.tools]

for _ in range(8):
    msg = client.chat.completions.create(model=os.environ.get("MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
                                         messages=messages, tools=tools, temperature=0).choices[0].message
    if not msg.tool_calls:
        break
    messages.append({"role": "assistant", "content": msg.content or "",
                     "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
    for tc in msg.tool_calls:
        result = env.call(tc.function.name, json.loads(tc.function.arguments or "{}"))
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

print("native actions executed:", env.native_actions)
print("protocol flags:", env.finish())
