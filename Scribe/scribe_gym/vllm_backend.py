"""vLLM OpenAI-server rollout backend for SCRIBE.

Instead of embedding ``vllm.AsyncLLMEngine`` in-process, this backend connects
to a standalone vLLM OpenAI-compatible server via HTTP. This is the recommended
setup for online RL:

1. Start the server (in a separate shell / process):

    vllm serve /home/zane/exp-scribe/qwen2.5-0.5b-instruct \
        --enable-lora \
        --lora-modules scribe_adapter=outputs/scribe_sft/final_lora \
        --gpu-memory-utilization 0.45 \
        --max-model-len 4096 \
        --port 8000

2. Pass ``VLLMBackend(base_url="http://localhost:8000/v1")`` to
   ``GymBackedAgent`` as ``llm_backend``.

3. After each GRPO update, call ``backend.update_weights(adapter_path)`` to load
   the new LoRA adapter into the running server.

The backend inherits ``AsyncOpenAI`` and is used exactly like the OpenAI client
inside ``GymBackedAgent.chat``:

    response = await backend.chat.completions.create(
        model=backend.lora_name,
        messages=[...],
        tools=[...],
        tool_choice="auto",
    )
    message = response.choices[0].message.model_dump()
"""

from __future__ import annotations

from openai import AsyncOpenAI


class VLLMBackend(AsyncOpenAI):
    """AsyncOpenAI client pointing at a local vLLM OpenAI server.

    Args:
        base_url: vLLM server URL, e.g. ``http://localhost:8000/v1``.
        api_key: Optional API key (vLLM defaults ignore this).
        lora_name: Name of the LoRA module registered on the server.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "vllm",
        lora_name: str = "scribe_adapter",
    ):
        super().__init__(base_url=base_url, api_key=api_key)
        self.lora_name = lora_name
        self.base_url = base_url

    def update_weights(self, lora_path: str) -> None:
        """Load a new LoRA adapter into the running vLLM server.

        Uses vLLM's ``/v1/load_lora_adapter`` admin endpoint. If the call fails
        (e.g. endpoint disabled), the user can fall back to restarting the
        server with the new ``--lora-modules`` argument.
        """
        import httpx

        url = f"{str(self.base_url).rstrip('/')}/load_lora_adapter"
        payload = {
            "lora_name": self.lora_name,
            "lora_path": lora_path,
            "load_inplace": True,
        }
        try:
            resp = httpx.post(url, json=payload, timeout=60.0)
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load LoRA adapter at {lora_path} into vLLM server. "
                f"Ensure the server was started with --enable-lora and that the "
                f"load_lora_adapter admin endpoint is enabled. Original error: {exc}"
            ) from exc


__all__ = ["VLLMBackend"]
