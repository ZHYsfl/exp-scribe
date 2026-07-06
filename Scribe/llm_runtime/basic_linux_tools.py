"""Built-in Linux workspace tools for the Agent.

Every tool is a standalone function that returns a string; the Agent
calls them during the tool-calling loop.  The schema helpers are in
``_tool_helpers.py`` to keep this file under 250 lines.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ._models import Tool
from ._tool_helpers import MAX_TOOL_OUTPUT_CHARS, GREP_TIMEOUT_SECONDS, resolve, schema, truncate


def create_basic_linux_tools(workspace_root: str | Path = ".") -> list[Tool]:
    root = Path(workspace_root).resolve()

    def bash(command: str, cwd: str | None = None, timeout: int = 30) -> str:
        """Run a bash command in the workspace."""
        try:
            workdir = resolve(root, cwd)
        except ValueError as exc:
            return truncate(f"Cannot run the command because {exc}.")
        if not workdir.exists():
            try:
                workdir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                return truncate(
                    f"Cannot run the command because the working directory "
                    f"{workdir} does not exist and could not be created: {exc}."
                )
        if not workdir.is_dir():
            return truncate(
                f"Cannot run the command because the working directory "
                f"is not a directory: {workdir} (cwd={cwd!r})."
            )

        try:
            result = subprocess.run(
                ["/bin/bash", "-lc", command],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return truncate(f"The command timed out after {timeout} seconds.")

        # Return stdout/stderr VERBATIM. Do not strip or normalize newlines —
        # the model relies on the exact command output (e.g. `ls` line breaks,
        # trailing blank lines) to reason about results. We only guarantee a
        # separator between sections: if stdout doesn't already end with a
        # newline (e.g. `printf` without \n), add one so the `stderr:` label
        # never glues onto the output.
        stdout_text = result.stdout or "(empty)"
        stderr_text = result.stderr or "(empty)"
        if stdout_text and not stdout_text.endswith("\n"):
            stdout_text += "\n"
        return truncate(
            f"exit_code: {result.returncode}\n"
            f"stdout:\n{stdout_text}"
            f"stderr:\n{stderr_text}"
        )

    def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a text file from the workspace with line numbers."""
        try:
            file_path = resolve(root, path)
        except ValueError as exc:
            return truncate(f"Cannot read {path} because {exc}.")
        if not file_path.is_file():
            return truncate(f"Cannot read {path} because it is not a file.")

        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        total_lines = len(lines)
        if total_lines == 0:
            return truncate(f"{path} is empty and has 0 lines.")
        if start_line < 1:
            return truncate(
                "Cannot read the file because start_line must be at least 1."
            )
        if end_line is not None and end_line < start_line:
            return truncate(
                "Cannot read the file because end_line must be "
                "greater than or equal to start_line."
            )
        if start_line > total_lines:
            return truncate(
                f"Cannot read from line {start_line} because the file "
                f"only has {total_lines} lines."
            )

        start_index = start_line - 1
        end_index = (
            min(end_line, total_lines) if end_line is not None else total_lines
        )
        selected = lines[start_index:end_index]

        return truncate(
            "\n".join(
                f"{line_number}: {line}"
                for line_number, line in enumerate(selected, start=start_line)
            )
        )

    def write_file(path: str, content: str) -> str:
        """Overwrite a text file in the workspace."""
        try:
            file_path = resolve(root, path)
        except ValueError as exc:
            return truncate(f"Cannot write {path} because {exc}.")
        if file_path.exists() and file_path.is_dir():
            return truncate(f"Cannot write {path} because it is a directory.")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return truncate(f"Successfully overwrote {path}.")

    def append_file(path: str, content: str) -> str:
        """Append text to a file in the workspace."""
        try:
            file_path = resolve(root, path)
        except ValueError as exc:
            return truncate(f"Cannot append to {path} because {exc}.")
        if file_path.exists() and file_path.is_dir():
            return truncate(f"Cannot append to {path} because it is a directory.")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("a", encoding="utf-8") as f:
            f.write(content)
        return truncate(f"Successfully appended to {path}.")

    def replace_in_file(path: str, old_text: str, new_text: str) -> str:
        """Replace exact text in a workspace file."""
        if old_text == "":
            return truncate("Cannot replace text because old_text is empty.")
        try:
            file_path = resolve(root, path)
        except ValueError as exc:
            return truncate(f"Cannot edit {path} because {exc}.")
        if not file_path.is_file():
            return truncate(f"Cannot edit {path} because it is not a file.")

        content = file_path.read_text(encoding="utf-8", errors="replace")
        count = content.count(old_text)
        if count == 0:
            return truncate(
                "No changes were made because old_text was not found."
            )
        if count > 1:
            return truncate(
                f"No changes were made because old_text matched {count} times."
            )

        file_path.write_text(content.replace(old_text, new_text), encoding="utf-8")
        return truncate(f"Successfully replaced 1 occurrence in {path}.")

    def grep(pattern: str, path: str = ".") -> str:
        """Search files with grep."""
        try:
            target = resolve(root, path)
        except ValueError as exc:
            return truncate(f"Cannot search {path} because {exc}.")
        try:
            result = subprocess.run(
                ["grep", "-RIn", "--", pattern, str(target)],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=GREP_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return truncate(
                f"The grep command timed out after {GREP_TIMEOUT_SECONDS} seconds."
            )
        if result.returncode == 1:
            return truncate("(no matches)")
        if result.returncode != 0:
            return truncate(f"The grep command failed: {result.stderr}")
        return truncate(result.stdout)

    def list_dir(path: str = ".") -> str:
        """List a workspace directory."""
        try:
            dir_path = resolve(root, path)
        except ValueError as exc:
            return truncate(f"Cannot list {path} because {exc}.")
        if not dir_path.is_dir():
            return truncate(
                f"Cannot list {path} because it is not a directory."
            )
        entries = sorted(
            child.name + ("/" if child.is_dir() else "")
            for child in dir_path.iterdir()
        )
        if not entries:
            return truncate("The directory is empty.")
        return truncate("\n".join(entries))

    return [
        Tool(
            name="bash",
            description=(
                "Run a bash command in the workspace and return the exit code, "
                "stdout, and stderr. "
                f"Use this for tests, builds, git commands, and shell inspection. "
                f"Output is truncated to {MAX_TOOL_OUTPUT_CHARS} characters."
            ),
            function=bash,
            parameters=schema(
                {
                    "command": {
                        "type": "string",
                        "description": "The bash command to run.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": (
                            "Optional working directory relative to the workspace "
                            "root. Defaults to '.' (the workspace root)."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Optional timeout in seconds. Defaults to 30.",
                    },
                },
                ["command"],
            ),
        ),
        Tool(
            name="read_file",
            description=(
                "Read a UTF-8 text file from the workspace with 1-indexed line "
                "numbers. Use start_line and end_line to inspect part of a large "
                f"file. Output is truncated to {MAX_TOOL_OUTPUT_CHARS} characters."
            ),
            function=read_file,
            parameters=schema(
                {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace root.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional first line to read. Defaults to 1.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": (
                            "Optional last line to read. Defaults to the end "
                            "of the file."
                        ),
                    },
                },
                ["path"],
            ),
        ),
        Tool(
            name="write_file",
            description=(
                "Create or overwrite a UTF-8 text file in the workspace. "
                "This replaces the entire file content."
            ),
            function=write_file,
            parameters=schema(
                {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete new file content.",
                    },
                },
                ["path", "content"],
            ),
        ),
        Tool(
            name="append_file",
            description=(
                "Append UTF-8 text to the end of a workspace file. "
                "Creates the file and parent directories if they do not exist."
            ),
            function=append_file,
            parameters=schema(
                {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text to append to the file.",
                    },
                },
                ["path", "content"],
            ),
        ),
        Tool(
            name="replace_in_file",
            description=(
                "Replace one exact text block in a workspace file. "
                "The edit is applied only when old_text appears exactly once."
            ),
            function=replace_in_file,
            parameters=schema(
                {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace root.",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to find.",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                ["path", "old_text", "new_text"],
            ),
        ),
        Tool(
            name="grep",
            description=(
                "Search files recursively with grep -RIn and return path, "
                "line number, and matching text. "
                "Use this to locate symbols, strings, or code patterns. "
                f"Output is truncated to {MAX_TOOL_OUTPUT_CHARS} characters."
            ),
            function=grep,
            parameters=schema(
                {
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Text or grep pattern to search for."
                        ),
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Optional file or directory path relative to "
                            "the workspace root. Defaults to '.' "
                            "(the workspace root)."
                        ),
                    },
                },
                ["pattern"],
            ),
        ),
        Tool(
            name="list_dir",
            description=(
                "List the immediate contents of a workspace directory. "
                "Directory names are returned with a trailing slash. "
                f"Output is truncated to {MAX_TOOL_OUTPUT_CHARS} characters."
            ),
            function=list_dir,
            parameters=schema(
                {
                    "path": {
                        "type": "string",
                        "description": (
                            "Optional directory path relative to the "
                            "workspace root. Defaults to '.' "
                            "(the workspace root)."
                        ),
                    }
                }
            ),
        ),
    ]


def register_basic_linux_tools(
    agent: Any, workspace_root: str | Path = "."
) -> list[Tool]:
    tools = create_basic_linux_tools(workspace_root)
    for tool in tools:
        agent.add_tool(tool)
    return tools
