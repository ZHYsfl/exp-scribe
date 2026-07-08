import asyncio
import inspect
import json
from typing import TYPE_CHECKING, Any, Callable, Dict

from openai import AsyncOpenAI

from ._models import LLMConfig, Tool

if TYPE_CHECKING:
    from .observation_compressor import ObservationCompressor


def merge_reasoning_content(message: Dict[str, Any]) -> Dict[str, Any]:
    """Convert DeepSeek-style ``reasoning_content`` into a SCRIBE ``<think>`` block.

    Some APIs return chain-of-thought in a separate ``reasoning_content`` field
    instead of inside ``content``. SCRIBE expects reasoning as
    ``<think>...</think>`` in the assistant's content so the parser can treat it
    as a THINK block. This helper normalizes such messages once, right after
    receiving them from the API.
    """
    msg = dict(message)
    reasoning = msg.get("reasoning_content")
    if isinstance(reasoning, str):
        reasoning = reasoning.strip()
        if reasoning:
            content = (msg.get("content") or "").strip()
            if content:
                msg["content"] = f"<think>\n{reasoning}\n</think>\n{content}"
            else:
                msg["content"] = f"<think>\n{reasoning}\n</think>"
            msg.pop("reasoning_content", None)
    return msg


class Agent:
    def __init__(
        self,
        config: LLMConfig,
        max_tool_retries: int = 3,
        debug: bool = False,
        compressor: "ObservationCompressor | None" = None,
    ):
        self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        self.tools = []
        self.config = config
        self.debug = debug
        self.max_tool_retries = max_tool_retries
        self.compressor = compressor

    def add_tool(self, tool: Tool):
        self.tools.append(tool)

    def _get_tools(self) -> list[dict]:
        """Convert to the OpenAI tools format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools
        ]

    def remove_tool(self, tool: Tool):
        tools = [t for t in self.tools if t.name != tool.name]
        self.tools = tools

    async def _execute_tool_call(
        self, tool_call, available_functions: dict[str, Callable]
    ) -> dict:
        """Execute a single tool call. Return a tool message with structured error markers."""
        function_name = tool_call.function.name
        raw_arguments = tool_call.function.arguments
        function_args = {}

        # Parse argument errors
        if raw_arguments and raw_arguments.strip() and raw_arguments != "{}":
            try:
                function_args = json.loads(raw_arguments)
            except json.JSONDecodeError as e:
                error_msg = (
                    f"[PARSE_ERROR] Failed to parse argument JSON: {e}. "
                    f"Raw arguments: '{raw_arguments}'"
                )
                if self.debug:
                    print(f"[Error] {error_msg}")
                return {
                    "role": "tool",
                    "content": error_msg,
                    "tool_call_id": tool_call.id,
                    "_tool_status": "error",
                    "_error_type": "parse_error",
                }

        # Function not found
        if function_name not in available_functions:
            error_msg = f"[NOT_FOUND] No function named '{function_name}' was found"
            if self.debug:
                print(f"[Error] {error_msg}")
            return {
                "role": "tool",
                "content": error_msg,
                "tool_call_id": tool_call.id,
                "_tool_status": "error",
                "_error_type": "not_found",
            }

        function_to_call = available_functions[function_name]

        try:
            # Check whether the function needs the g argument
            func_params = function_to_call.__code__.co_varnames[
                0 : function_to_call.__code__.co_argcount
            ]
            if "g" in func_params:
                function_args["g"] = globals()

            if self.debug:
                print(
                    f"[Debug] Executing function {function_name} "
                    f"with arguments: {list(function_args.keys())}"
                )

            ORANGE = "\033[38;5;214m"
            RESET = "\033[0m"
            print(f"{ORANGE}{'-'*20}{RESET}")
            print(f"{ORANGE}execute tool: {function_name}{RESET}")
            for key in list(function_args.keys()):
                print(f"{ORANGE}{key}: {function_args[key]}{RESET}")
            print(f"{ORANGE}{'-'*20}{RESET}")
            if self.debug:
                print("\033[94mWaiting...\033[0m")

            # Execute the function
            if inspect.iscoroutinefunction(function_to_call):
                result = await function_to_call(**function_args)
            else:
                result = await asyncio.to_thread(function_to_call, **function_args)
                if inspect.isawaitable(result):
                    result = await result

            if self.debug:
                print("-" * 20)
                print("execute result:")
                result_str = str(result)
                print(result_str[:100])
                print("-" * 20)

            return {
                "role": "tool",
                "content": str(result),
                "tool_call_id": tool_call.id,
                "_tool_status": "success",
            }

        except TypeError as e:
            expected_args = function_to_call.__code__.co_varnames[
                0 : function_to_call.__code__.co_argcount
            ]
            error_msg = (
                f"[ARG_ERROR] Argument mismatch: {e}. "
                f"Expected: {expected_args}, actual: {list(function_args.keys())}"
            )
            if self.debug:
                print(f"[Error] {error_msg}")
            return {
                "role": "tool",
                "content": error_msg,
                "tool_call_id": tool_call.id,
                "_tool_status": "error",
                "_error_type": "arg_error",
            }

        except Exception as e:
            import traceback

            error_detail = traceback.format_exc()
            error_msg = (
                f"[EXEC_ERROR] Execution failed: {e}\n\nError details:\n{error_detail}"
            )
            if self.debug:
                print(f"[Error] {error_msg}")
            return {
                "role": "tool",
                "content": error_msg,
                "tool_call_id": tool_call.id,
                "_tool_status": "error",
                "_error_type": "exec_error",
            }

    async def _get_tool_response_observations(self, response) -> list[dict]:
        """Execute tool calls and return the response list.

        Note: this method only executes tools and does not modify observations;
        the returned list should be extended by the caller.
        """
        tool_calls = response.choices[0].message.tool_calls
        return await self._execute_tool_calls(tool_calls)

    async def _execute_tool_calls(self, tool_calls) -> list[dict]:
        """Execute a list of tool calls concurrently.

        Subclasses can override this to plug in a different execution backend
        (e.g. a gym environment) while reusing the chat loop.
        """
        available_functions = {tool.name: tool.function for tool in self.tools}
        tool_tasks = [
            self._execute_tool_call(tool_call, available_functions)
            for tool_call in tool_calls
        ]
        return await asyncio.gather(*tool_tasks)

    def _should_stop_after_tool_response(self, tool_responses: list[dict]) -> bool:
        """Hook for subclasses to break the tool-call loop early."""
        return False

    def _has_tool_errors(self, tool_responses: list[dict]) -> bool:
        """Check whether tool responses contain errors via structured markers."""
        for resp in tool_responses:
            status = resp.get("_tool_status")
            if status == "error":
                return True
            content = resp.get("content", "")
            if isinstance(content, str) and content.startswith("[") and "_ERROR" in content:
                return True
        return False

    def _get_error_summary(self, tool_responses: list[dict]) -> str:
        """Get an error summary for prompting the model."""
        errors = []
        for resp in tool_responses:
            if resp.get("_tool_status") == "error":
                error_type = resp.get("_error_type", "unknown")
                content = resp.get("content", "")
                errors.append(f"- {error_type}: {content[:200]}")
        return "\n".join(errors) if errors else "Unknown error"

    async def _compress_if_needed(self, observations: list[dict]) -> list[dict]:
        """Apply compressor when the observation list exceeds the token limit."""
        if self.compressor is None:
            return observations
        return await self.compressor.compress(observations)

    async def _on_final_response(self, response) -> None:
        """Hook called after the tool-call loop ends and before the final
        assistant message is added to the observation list.
        """
        pass

    # Tloop -> Tloop -> Tloop -> ... -> OLoop -> observations_final
    # Tloop : observations -> Agent -> tool_action -> Environment -> ...
    # OLoop : observations -> Agent -> output_action -> Environment -> ...
    async def chat(self, observations: list[dict]) -> list[dict]:
        observations = await self._compress_if_needed(observations)

        print("\033[94mThinking...\033[0m")

        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=observations,
            tools=self._get_tools(),
            tool_choice="auto",
        )

        print(response.choices[0].message.content)

        observations_next = observations.copy()
        retry_count = 0

        # Handle the tool-call loop with automatic retries and mid-loop compression.
        while response.choices[0].finish_reason == "tool_calls":
            # Add the assistant tool_calls message.
            observations_next.append(
                merge_reasoning_content(response.choices[0].message.model_dump())
            )

            # Execute all tool calls concurrently.
            tool_responses = await self._get_tool_response_observations(response)
            observations_next.extend(tool_responses)

            # Mid-loop compression: shrink context before the next LLM call.
            observations_next = await self._compress_if_needed(observations_next)

            # If there are tool errors and retries remain, ask the model to correct them.
            if (
                self._has_tool_errors(tool_responses)
                and retry_count < self.max_tool_retries
            ):
                retry_count += 1
                error_summary = self._get_error_summary(tool_responses)
                if self.debug:
                    print(
                        f"[Retry {retry_count}/{self.max_tool_retries}] "
                        f"Detected tool execution errors:\n{error_summary}"
                    )
                observations_next.append(
                    {
                        "role": "user",
                        "content": (
                            f"[System Notice] Your previous tool call failed with "
                            f"the following errors:\n\n{error_summary}\n\n"
                            f"Please analyze the cause, correct it, and call the tool "
                            f"again. Remaining retries: "
                            f"{self.max_tool_retries - retry_count}"
                        ),
                    }
                )

            if self._should_stop_after_tool_response(tool_responses):
                break

            print("\033[94mThinking...\033[0m")

            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=observations_next,
                tools=self._get_tools(),
                tool_choice="auto",
            )

            print(response.choices[0].message.content)

        observations_final = observations_next
        # Allow subclasses to react to the final non-tool response before it is
        # added to the conversation (e.g. to step a gym env one last time).
        await self._on_final_response(response)
        # Add the final reply to observations.
        observations_final.append(
            merge_reasoning_content(response.choices[0].message.model_dump())
        )
        # Compress the final list so the next turn starts within budget.
        observations_final = await self._compress_if_needed(observations_final)
        return observations_final


# Re-export models for backward-compatible imports via tool_calling.
__all__ = ["Agent", "LLMConfig", "Tool"]
