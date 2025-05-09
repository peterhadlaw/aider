import glob
import os
import re
import subprocess
import sys
import tempfile
from collections import OrderedDict
from os.path import expanduser
from pathlib import Path

import pyperclip
from PIL import Image, ImageGrab
from prompt_toolkit.completion import Completion, PathCompleter
from prompt_toolkit.document import Document

from aider import models, prompts, voice
from aider.editor import pipe_editor
from aider.format_settings import format_settings
from aider.help import Help, install_help_extra
from aider.io import CommandCompletionException
from aider.llm import litellm
from aider.repo import ANY_GIT_ERROR
from aider.run_cmd import run_cmd
from aider.scrape import Scraper, install_playwright
from aider.utils import is_image_file

from .dump import dump  # noqa: F401


class SwitchCoder(Exception):
    def __init__(self, placeholder=None, **kwargs):
        self.kwargs = kwargs
        self.placeholder = placeholder


class Commands:
    voice = None
    scraper = None

    def clone(self):
        return Commands(
            self.io,
            None,
            voice_language=self.voice_language,
            verify_ssl=self.verify_ssl,
            args=self.args,
            parser=self.parser,
            verbose=self.verbose,
            editor=self.editor,
        )

    def __init__(
        self,
        io,
        coder,
        voice_language=None,
        voice_input_device=None,
        voice_format=None,
        verify_ssl=True,
        args=None,
        parser=None,
        verbose=False,
        editor=None,
        original_read_only_fnames=None,
    ):
        self.io = io
        self.coder = coder
        self.parser = parser
        self.args = args
        self.verbose = verbose

        self.verify_ssl = verify_ssl
        if voice_language == "auto":
            voice_language = None

        self.voice_language = voice_language
        self.voice_format = voice_format
        self.voice_input_device = voice_input_device

        self.help = None
        self.editor = editor

        # Store the original read-only filenames provided via args.read
        self.original_read_only_fnames = set(original_read_only_fnames or [])

    def cmd_model(self, args):
        "Switch the Main Model to a new LLM"

        model_name = args.strip()
        model = models.Model(
            model_name,
            editor_model=self.coder.main_model.editor_model.name,
            weak_model=self.coder.main_model.weak_model.name,
        )
        models.sanity_check_models(self.io, model)

        # Check if the current edit format is the default for the old model
        old_model_edit_format = self.coder.main_model.edit_format
        current_edit_format = self.coder.edit_format

        new_edit_format = current_edit_format
        if current_edit_format == old_model_edit_format:
            # If the user was using the old model's default, switch to the new model's default
            new_edit_format = model.edit_format

        raise SwitchCoder(main_model=model, edit_format=new_edit_format)

    def cmd_editor_model(self, args):
        "Switch the Editor Model to a new LLM"

        model_name = args.strip()
        model = models.Model(
            self.coder.main_model.name,
            editor_model=model_name,
            weak_model=self.coder.main_model.weak_model.name,
        )
        models.sanity_check_models(self.io, model)
        raise SwitchCoder(main_model=model)

    def cmd_weak_model(self, args):
        "Switch the Weak Model to a new LLM"

        model_name = args.strip()
        model = models.Model(
            self.coder.main_model.name,
            editor_model=self.coder.main_model.editor_model.name,
            weak_model=model_name,
        )
        models.sanity_check_models(self.io, model)
        raise SwitchCoder(main_model=model)

    def cmd_chat_mode(self, args):
        "Switch to a new chat mode"

        from aider import coders

        ef = args.strip()
        valid_formats = OrderedDict(
            sorted(
                (
                    coder.edit_format,
                    coder.__doc__.strip().split("\n")[0] if coder.__doc__ else "No description",
                )
                for coder in coders.__all__
                if getattr(coder, "edit_format", None)
            )
        )

        show_formats = OrderedDict(
            [
                ("help", "Get help about using aider (usage, config, troubleshoot)."),
                ("ask", "Ask questions about your code without making any changes."),
                ("code", "Ask for changes to your code (using the best edit format)."),
                (
                    "architect",
                    (
                        "Work with an architect model to design code changes, and an editor to make"
                        " them."
                    ),
                ),
                (
                    "context",
                    "Automatically identify which files will need to be edited.",
                ),
            ]
        )

        if ef not in valid_formats and ef not in show_formats:
            if ef:
                self.io.tool_error(f'Chat mode "{ef}" should be one of these:\n')
            else:
                self.io.tool_output("Chat mode should be one of these:\n")

            max_format_length = max(len(format) for format in valid_formats.keys())
            for format, description in show_formats.items():
                self.io.tool_output(f"- {format:<{max_format_length}} : {description}")

            self.io.tool_output("\nOr a valid edit format:\n")
            for format, description in valid_formats.items():
                if format not in show_formats:
                    self.io.tool_output(f"- {format:<{max_format_length}} : {description}")

            return

        summarize_from_coder = True
        edit_format = ef

        if ef == "code":
            edit_format = self.coder.main_model.edit_format
            summarize_from_coder = False
        elif ef == "ask":
            summarize_from_coder = False

        raise SwitchCoder(
            edit_format=edit_format,
            summarize_from_coder=summarize_from_coder,
        )

    def completions_model(self):
        models = litellm.model_cost.keys()
        return models

    def cmd_models(self, args):
        "Search the list of available models"

        args = args.strip()

        if args:
            models.print_matching_models(self.io, args)
        else:
            self.io.tool_output("Please provide a partial model name to search for.")

    def cmd_web(self, args, return_content=False):
        "Scrape a webpage, convert to markdown and send in a message"

        url = args.strip()
        if not url:
            self.io.tool_error("Please provide a URL to scrape.")
            return

        self.io.tool_output(f"Scraping {url}...")
        if not self.scraper:
            disable_playwright = getattr(self.args, "disable_playwright", False)
            if disable_playwright:
                res = False
            else:
                res = install_playwright(self.io)
                if not res:
                    self.io.tool_warning("Unable to initialize playwright.")

            self.scraper = Scraper(
                print_error=self.io.tool_error,
                playwright_available=res,
                verify_ssl=self.verify_ssl,
            )

        content = self.scraper.scrape(url) or ""
        content = f"Here is the content of {url}:\n\n" + content
        if return_content:
            return content

        self.io.tool_output("... added to chat.")

        self.coder.cur_messages += [
            dict(role="user", content=content),
            dict(role="assistant", content="Ok."),
        ]

    def is_command(self, inp):
        return inp[0] in "/!"

    def get_raw_completions(self, cmd):
        assert cmd.startswith("/")
        cmd = cmd[1:]
        cmd = cmd.replace("-", "_")

        raw_completer = getattr(self, f"completions_raw_{cmd}", None)
        return raw_completer

    def get_completions(self, cmd):
        assert cmd.startswith("/")
        cmd = cmd[1:]

        cmd = cmd.replace("-", "_")
        fun = getattr(self, f"completions_{cmd}", None)
        if not fun:
            return
        return sorted(fun())

    def get_commands(self):
        commands = []
        for attr in dir(self):
            if not attr.startswith("cmd_"):
                continue
            cmd = attr[4:]
            cmd = cmd.replace("_", "-")
            commands.append("/" + cmd)

        return commands

    def do_run(self, cmd_name, args):
        cmd_name = cmd_name.replace("-", "_")
        cmd_method_name = f"cmd_{cmd_name}"
        cmd_method = getattr(self, cmd_method_name, None)
        if not cmd_method:
            self.io.tool_output(f"Error: Command {cmd_name} not found.")
            return

        try:
            return cmd_method(args)
        except ANY_GIT_ERROR as err:
            self.io.tool_error(f"Unable to complete {cmd_name}: {err}")

    def matching_commands(self, inp):
        words = inp.strip().split()
        if not words:
            return

        first_word = words[0]
        rest_inp = inp[len(words[0]) :].strip()

        all_commands = self.get_commands()
        matching_commands = [cmd for cmd in all_commands if cmd.startswith(first_word)]
        return matching_commands, first_word, rest_inp

    def run(self, inp):
        if inp.startswith("!"):
            self.coder.event("command_run")
            return self.do_run("run", inp[1:])

        res = self.matching_commands(inp)
        if res is None:
            return
        matching_commands, first_word, rest_inp = res
        if len(matching_commands) == 1:
            command = matching_commands[0][1:]
            self.coder.event(f"command_{command}")
            return self.do_run(command, rest_inp)
        elif first_word in matching_commands:
            command = first_word[1:]
            self.coder.event(f"command_{command}")
            return self.do_run(command, rest_inp)
        elif len(matching_commands) > 1:
            self.io.tool_error(f"Ambiguous command: {', '.join(matching_commands)}")
        else:
            self.io.tool_error(f"Invalid command: {first_word}")

    # any method called cmd_xxx becomes a command automatically.
    # each one must take an args param.

    def cmd_commit(self, args=None):
        "Commit edits to the repo made outside the chat (commit message optional)"
        try:
            self.raw_cmd_commit(args)
        except ANY_GIT_ERROR as err:
            self.io.tool_error(f"Unable to complete commit: {err}")

    def raw_cmd_commit(self, args=None):
        if not self.coder.repo:
            self.io.tool_error("No git repository found.")
            return

        if not self.coder.repo.is_dirty():
            self.io.tool_warning("No more changes to commit.")
            return

        commit_message = args.strip() if args else None
        self.coder.repo.commit(message=commit_message)

    def cmd_lint(self, args="", fnames=None):
        "Lint and fix in-chat files or all dirty files if none in chat"

        if not self.coder.repo:
            self.io.tool_error("No git repository found.")
            return

        if not fnames:
            fnames = self.coder.get_inchat_relative_files()

        # If still no files, get all dirty files in the repo
        if not fnames and self.coder.repo:
            fnames = self.coder.repo.get_dirty_files()

        if not fnames:
            self.io.tool_warning("No dirty files to lint.")
            return

        fnames = [self.coder.abs_root_path(fname) for fname in fnames]

        lint_coder = None
        for fname in fnames:
            try:
                errors = self.coder.linter.lint(fname)
            except FileNotFoundError as err:
                self.io.tool_error(f"Unable to lint {fname}")
                self.io.tool_output(str(err))
                continue

            if not errors:
                continue

            self.io.tool_output(errors)
            if not self.io.confirm_ask(f"Fix lint errors in {fname}?", default="y"):
                continue

            # Commit everything before we start fixing lint errors
            if self.coder.repo.is_dirty() and self.coder.dirty_commits:
                self.cmd_commit("")

            if not lint_coder:
                lint_coder = self.coder.clone(
                    # Clear the chat history, fnames
                    cur_messages=[],
                    done_messages=[],
                    fnames=None,
                )

            lint_coder.add_rel_fname(fname)
            lint_coder.run(errors)
            lint_coder.abs_fnames = set()

        if lint_coder and self.coder.repo.is_dirty() and self.coder.auto_commits:
            self.cmd_commit("")

    def cmd_clear(self, args):
        "Clear the chat history"

        self._clear_chat_history()

    def _drop_all_files(self):
        self.coder.abs_fnames = set()

        # When dropping all files, keep those that were originally provided via args.read
        if self.original_read_only_fnames:
            # Keep only the original read-only files
            to_keep = set()
            for abs_fname in self.coder.abs_read_only_fnames:
                rel_fname = self.coder.get_rel_fname(abs_fname)
                if (
                    abs_fname in self.original_read_only_fnames
                    or rel_fname in self.original_read_only_fnames
                ):
                    to_keep.add(abs_fname)
            self.coder.abs_read_only_fnames = to_keep
        else:
            self.coder.abs_read_only_fnames = set()

    def _clear_chat_history(self):
        self.coder.done_messages = []
        self.coder.cur_messages = []

    def cmd_reset(self, args):
        "Drop all files and clear the chat history"
        self._drop_all_files()
        self._clear_chat_history()
        self.io.tool_output("All files dropped and chat history cleared.")

    def cmd_tokens(self, args):
        "Report on the number of tokens used by the current chat context"

        res = []

        self.coder.choose_fence()

        # system messages
        main_sys = self.coder.fmt_system_prompt(self.coder.gpt_prompts.main_system)
        main_sys += "\n" + self.coder.fmt_system_prompt(self.coder.gpt_prompts.system_reminder)
        msgs = [
            dict(role="system", content=main_sys),
            dict(
                role="system",
                content=self.coder.fmt_system_prompt(self.coder.gpt_prompts.system_reminder),
            ),
        ]

        tokens = self.coder.main_model.token_count(msgs)
        res.append((tokens, "system messages", ""))

        # chat history
        msgs = self.coder.done_messages + self.coder.cur_messages
        if msgs:
            tokens = self.coder.main_model.token_count(msgs)
            res.append((tokens, "chat history", "use /clear to clear"))

        # repo map
        other_files = set(self.coder.get_all_abs_files()) - set(self.coder.abs_fnames)
        if self.coder.repo_map:
            repo_content = self.coder.repo_map.get_repo_map(self.coder.abs_fnames, other_files)
            if repo_content:
                tokens = self.coder.main_model.token_count(repo_content)
                res.append((tokens, "repository map", "use --map-tokens to resize"))

        fence = "`" * 3

        file_res = []
        # files
        for fname in self.coder.abs_fnames:
            relative_fname = self.coder.get_rel_fname(fname)
            content = self.io.read_text(fname)
            if is_image_file(relative_fname):
                tokens = self.coder.main_model.token_count_for_image(fname)
            else:
                # approximate
                content = f"{relative_fname}\n{fence}\n" + content + "{fence}\n"
                tokens = self.coder.main_model.token_count(content)
            file_res.append((tokens, f"{relative_fname}", "/drop to remove"))

        # read-only files
        for fname in self.coder.abs_read_only_fnames:
            relative_fname = self.coder.get_rel_fname(fname)
            content = self.io.read_text(fname)
            if content is not None and not is_image_file(relative_fname):
                # approximate
                content = f"{relative_fname}\n{fence}\n" + content + "{fence}\n"
                tokens = self.coder.main_model.token_count(content)
                file_res.append((tokens, f"{relative_fname} (read-only)", "/drop to remove"))

        file_res.sort()
        res.extend(file_res)

        self.io.tool_output(
            f"Approximate context window usage for {self.coder.main_model.name}, in tokens:"
        )
        self.io.tool_output()

        width = 8
        cost_width = 9

        def fmt(v):
            return format(int(v), ",").rjust(width)

        col_width = max(len(row[1]) for row in res)

        cost_pad = " " * cost_width
        total = 0
        total_cost = 0.0
        for tk, msg, tip in res:
            total += tk
            cost = tk * (self.coder.main_model.info.get("input_cost_per_token") or 0)
            total_cost += cost
            msg = msg.ljust(col_width)
            self.io.tool_output(f"${cost:7.4f} {fmt(tk)} {msg} {tip}")  # noqa: E231

        self.io.tool_output("=" * (width + cost_width + 1))
        self.io.tool_output(f"${total_cost:7.4f} {fmt(total)} tokens total")  # noqa: E231

        limit = self.coder.main_model.info.get("max_input_tokens") or 0
        if not limit:
            return

        remaining = limit - total
        if remaining > 1024:
            self.io.tool_output(f"{remaining:,} tokens remaining")

    def cmd_reasoning_effort(self, args):
        "Set the reasoning effort parameter (for models that support it)"
        args = args.strip()

        if not args:
            # Display current value if no argument provided
            current_effort = self.coder.main_model.get_reasoning_effort()
            if current_effort is not None:
                self.io.tool_output(f"Current reasoning effort: {current_effort}")
            else:
                self.io.tool_output("No reasoning effort currently set")
            return

        # Set the reasoning effort
        self.coder.main_model.set_reasoning_effort(args)
        # Also set it for the editor model if applicable
        self.coder.main_model.set_editor_reasoning_effort(args)
        self.io.tool_output(f"Set reasoning effort to {args}")

    def cmd_editor_reasoning_effort(self, args):
        "Set the reasoning effort parameter for the editor model (for models that support it)"
        args = args.strip()

        # Check if there's a separate editor model
        if (
            self.coder.main_model.editor_model is None
            or self.coder.main_model.editor_model is self.coder.main_model
        ):
            self.io.tool_warning("No separate editor model configured. Use /reasoning-effort instead.")
            return

        if not args:
            # Display current value if no argument provided
            current_effort = self.coder.main_model.editor_model.get_reasoning_effort()
            if current_effort is not None:
                self.io.tool_output(f"Current editor model reasoning effort: {current_effort}")
            else:
                self.io.tool_output("No reasoning effort currently set for editor model")
            return

        # Set the reasoning effort for the editor model
        self.coder.main_model.editor_model.set_reasoning_effort(args)
        self.io.tool_output(f"Set editor model reasoning effort to {args}")

    def cmd_think_tokens(self, args):
        "Set the thinking token budget (for models that support it)"
        args = args.strip()

        if not args:
            # Display current value if no argument provided
            current_budget = self.coder.main_model.get_thinking_tokens()
            if current_budget is not None:
                self.io.tool_output(f"Current thinking token budget: {current_budget}")
            else:
                self.io.tool_output("No thinking token budget currently set")
            return

        # Set the thinking tokens
        raw_value = args
        tokens = self.coder.main_model.parse_token_value(raw_value)
        self.coder.main_model.set_thinking_tokens(tokens)
        self.io.tool_output(f"Set thinking token budget to {tokens:,} tokens ({raw_value}).")

    def cmd_run(self, cmd, add_on_nonzero_exit=False, add_to_chat=True):
        "Run a command in the shell"
        if not cmd:
            self.io.tool_error("Usage: /run <command>")
            return

        cmd = cmd.strip()
        self.io.tool_output(f"Running: {cmd}")

        exit_code, output = run_cmd(cmd)

        if output:
            self.io.tool_output(output)

        if exit_code:
            self.io.tool_output(f"Command exited with status: {exit_code}")
            if not add_on_nonzero_exit:
                return

        if add_to_chat:
            # Add the command and its output to the chat
            self.coder.cur_messages += [
                dict(
                    role="user",
                    content=f"I ran the command: {cmd}\n\n```\n{output}\n```",
                ),
                dict(role="assistant", content="Thanks for sharing the output."),
            ]

        return output

    def cmd_test(self, test_cmd):
        "Run tests and fix any issues found"
        if not test_cmd:
            if not self.coder.test_cmd:
                self.io.tool_error("No test command specified.")
                return None
            test_cmd = self.coder.test_cmd

        output = self.cmd_run(test_cmd, add_on_nonzero_exit=True)
        if not output:
            self.io.tool_output("Tests passed!")
            return None

        return output

    def cmd_undo(self, args):
        "Undo the last aider commit"
        if not self.coder.repo:
            self.io.tool_error("No git repository found.")
            return

        # Get the current commit hash
        commit = self.coder.repo.repo.head.commit

        # Check if this is the first commit in the repo
        if not commit.parents:
            self.io.tool_error("Cannot undo the first commit in the repository.")
            return

        # Check if this is an aider commit
        if commit.hexsha[:7] not in self.coder.aider_commit_hashes:
            self.io.tool_error("Last commit was not made by aider.")
            return

        # Check if the commit had new files created
        parent = commit.parents[0]
        files = commit.stats.files.keys()
        if any(parent.tree.find_missing_objects(files)):
            self.io.tool_error(
                "Cannot undo a commit that created new files. This would leave files untracked."
            )
            return

        # Ensure there are no unstaged changes
        if self.coder.repo.is_dirty():
            dirty_files = self.coder.repo.get_dirty_files()
            # Check if any dirty files are part of the commit to be reverted
            if any(file in dirty_files for file in files):
                self.io.tool_error(
                    "Cannot undo last commit: working directory has unsaved changes to files from"
                    " that commit."
                )
                return
        try:
            # Reset to the parent commit
            self.coder.repo.repo.git.reset("--hard", "HEAD~1")
            self.io.tool_output(
                f"Successfully undid commit {commit.hexsha[:7]}: {commit.message.splitlines()[0]}"
            )
            return True
        except Exception as e:
            self.io.tool_error(f"Error undoing commit: {e}")
            return False

    def cmd_diff(self, args):
        "Show changes in the git repository"
        if not self.coder.repo:
            self.io.tool_error("No git repository found.")
            return

        args = args.strip()

        if args:
            try:
                # Try to parse the arguments as commit references
                if ".." in args:
                    start, end = args.split("..", 1)
                    diff = self.coder.repo.diff_commits(True, start, end)
                else:
                    diff = self.coder.repo.diff_commits(True, args)
            except ANY_GIT_ERROR as err:
                self.io.tool_error(f"Error getting diff: {err}")
                return
        else:
            # Default behavior: show changes between last two commits
            diff = self.coder.repo.get_diffs(show_diff=True, context_lines=10)

        if diff:
            print(diff)
        else:
            self.io.tool_output("No changes found.")

    def cmd_save(self, filename):
        "Save the current session to a file"
        if not filename:
            filename = "aider-session.txt"
        filename = filename.strip()

        try:
            commands = []

            # Add commands for files in the session
            for fname in self.coder.abs_fnames:
                rel_fname = self.coder.get_rel_fname(fname)
                commands.append(f"/add {rel_fname}")

            # Add commands for read-only files
            for fname in self.coder.abs_read_only_fnames:
                rel_fname = self.coder.get_rel_fname(fname)
                commands.append(f"/read-only {rel_fname}")

            # Write commands to the file
            with open(filename, "w", encoding=self.io.encoding) as f:
                f.write("\n".join(commands))

            self.io.tool_output(f"Session saved to {filename}")
        except Exception as e:
            self.io.tool_error(f"Error saving session: {e}")

    def cmd_load(self, filename):
        "Load a session from a file"
        if not filename:
            self.io.tool_error("Please provide a filename to load")
            return
        filename = filename.strip()

        try:
            with open(filename, "r", encoding=self.io.encoding) as f:
                commands = f.read().splitlines()

            for cmd in commands:
                cmd = cmd.strip()
                if not cmd or not cmd.startswith("/"):
                    continue

                # Parse the command
                cmd_parts = cmd.split(None, 1)
                cmd_name = cmd_parts[0][1:]  # Remove the leading slash
                cmd_args = cmd_parts[1] if len(cmd_parts) > 1 else ""

                # Skip commands that are only supported in interactive mode
                if cmd_name in ["ask", "model", "editor-model", "weak-model", "chat-mode"]:
                    self.io.tool_error(
                        f"Command '{cmd}' is only supported in interactive mode, skipping."
                    )
                    continue

                try:
                    # Run the command
                    self.do_run(cmd_name, cmd_args)
                except SwitchCoder:
                    # These commands require restarting the session, which we can't do in load
                    self.io.tool_error(
                        f"Command '{cmd}' is only supported in interactive mode, skipping."
                    )
                except Exception as e:
                    self.io.tool_error(f"Error executing command '{cmd}': {e}")

            self.io.tool_output(f"Session loaded from {filename}")
        except Exception as e:
            self.io.tool_error(f"Error loading session: {e}")

    def cmd_editor(self, args):
        "Edit all active files in an external editor"
        if not self.coder.abs_fnames:
            self.io.tool_output("No files to edit. Use /add to add files to the chat.")
            return

        # Make a list of files to edit
        files_to_edit = list(self.coder.abs_fnames)
        read_only_warning = ""
        if self.coder.abs_read_only_fnames:
            read_only_warning = (
                "\nNote: Read-only files are not included. Use /add to convert them to editable."
            )

        self.io.tool_output(f"Opening editor with {len(files_to_edit)} files.{read_only_warning}")

        # Find the editor to use
        editor = self.editor
        if not editor:
            # Fall back to environment variables
            editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
            if not editor:
                self.io.tool_error(
                    "No editor specified. Set one with --editor or the VISUAL/EDITOR environment"
                    " variables."
                )
                return

        # Prepare the command
        cmd = f"{editor} {' '.join(map(repr, files_to_edit))}"

        # Run the editor
        exit_code, output = run_cmd(cmd)
        if exit_code != 0:
            self.io.tool_error(f"Editor exited with status {exit_code}")
            if output:
                self.io.tool_output(output)

    def cmd_paste(self, args):
        "Apply content from clipboard to edit files"
        try:
            content = pyperclip.paste()
        except pyperclip.PyperclipException as e:
            self.io.tool_error(f"Unable to access clipboard: {e}")
            return

        if not content:
            self.io.tool_error("Clipboard is empty")
            return

        self.coder.partial_response_content = content
        self.coder.apply_updates()

    def cmd_copy(self, args):
        "Copy the last assistant message to the clipboard"
        all_messages = self.coder.done_messages + self.coder.cur_messages
        assistant_messages = [msg for msg in all_messages if msg["role"] == "assistant"]

        if not assistant_messages:
            self.io.tool_error("No assistant messages found to copy.")
            return

        last_message = assistant_messages[-1]["content"]
        try:
            pyperclip.copy(last_message)
            # Show a preview of the copied content (first 50 chars)
            preview = last_message[:50] + ("..." if len(last_message) > 50 else "")
            self.io.tool_output(f"Copied last assistant message to clipboard. Preview: {preview}")
        except pyperclip.PyperclipException as e:
            self.io.tool_error(f"Failed to copy to clipboard: {e}")

    def cmd_edit(self, args):
        "Edit the last prompt using an external editor"
        all_messages = self.coder.done_messages + self.coder.cur_messages
        user_messages = [msg for msg in all_messages if msg["role"] == "user"]

        if not user_messages:
            self.io.tool_error("No user messages found to edit.")
            return

        last_message = user_messages[-1]["content"]
        try:
            edited_content = pipe_editor(last_message, editor=self.editor)
            if edited_content != last_message:
                self.io.tool_output("Message edited. Sending to assistant...")
                self.coder.run(with_message=edited_content)
            else:
                self.io.tool_output("No changes made to the message.")
        except Exception as e:
            self.io.tool_error(f"Error editing message: {e}")

    # Completions for command arguments

    def completions_raw_add(self, document, complete_event, remaining_text, words):
        if len(words) > 1:
            # Get completions for subsequent arguments (multiple files)
            # Exclude files that are already in abs_fnames to avoid duplicating them
            existing_files = set([self.coder.get_rel_fname(f) for f in self.coder.abs_fnames])
            addable_files = set(self.coder.get_addable_relative_files()) - existing_files
            
            # Also check if files from abs_read_only_fnames are in the repository,
            # so they can be promoted to editable
            read_only_files = [
                self.coder.get_rel_fname(f) for f in self.coder.abs_read_only_fnames
            ]
            addable_files.update(set(read_only_files))
            
            # Get the current word being completed
            current_word = words[-1]
            completions = []
            
            for fname in addable_files:
                if fname.startswith(current_word):
                    display = f"{fname}"
                    completions.append(Completion(fname, start_position=-len(current_word), display=display))
            
            return completions

        # First argument handling (path completion)
        completer = PathCompleter(
            only_directories=False,
            expanduser=True,
        )
        return completer.get_completions(document, complete_event)

    def cmd_help(self, args):
        "Show help about using aider"
        if not self.help:
            self.help = Help()
        return self.help.full_help(args)

    def cmd_ask(self, question):
        "Switch to Ask Mode and ask the question"
        if not question:
            self.io.tool_error("Usage: /ask <question about your code>")
            return

        # Switch to Ask mode
        raise SwitchCoder(
            edit_format="ask",
            summarize_from_coder=False,
            placeholder=question,
        )

    def cmd_voice(self, args):
        "Send a message from voice input"
        language = self.voice_language
        device = self.voice_input_device
        audio_format = self.voice_format

        if not self.voice:
            try:
                # Check if we have the required dependencies
                import speech_recognition  # noqa
                import sounddevice  # noqa

                if audio_format in ["mp3", "webm"]:
                    # Try to load ffmpeg for mp3/webm support
                    import ffmpeg  # noqa
            except ImportError as e:
                missing_package = str(e).split("'")[1]
                self.io.tool_error(f"Missing required package: {missing_package}")
                self.io.tool_output(
                    "Please install voice dependencies with:"
                    " pip install aider-chat[voice]"
                    f" or pip install {missing_package}"
                )
                return

            # Initialize the voice module
            try:
                self.voice = voice.Voice(device_name=device, audio_format=audio_format)
            except Exception as e:
                self.io.tool_error(f"Error initializing voice module: {e}")
                return

        try:
            # Record and transcribe voice
            self.io.tool_output("🎙️ Recording... (Press Ctrl+C to stop)")
            transcription = self.voice.record_and_transcribe(language=language)
            if transcription:
                self.io.tool_output(f"🔊 Transcribed: {transcription}")
                self.coder.run(with_message=transcription)
            else:
                self.io.tool_output("❌ No speech detected or transcription failed.")
        except KeyboardInterrupt:
            self.io.tool_output("Recording stopped by user.")
        except Exception as e:
            self.io.tool_error(f"Error during voice recording: {e}")

    def cmd_browse(self, args):
        "Open a URL in the browser"
        url = args.strip()
        if not url:
            self.io.tool_error("Please provide a URL to open")
            return

        # Add http:// prefix if missing
        if not url.startswith(("http://", "https://", "file://")):
            url = "https://" + url

        self.io.offer_url(url, f"Open {url}?", allow_never=False)

    def cmd_add(self, args):
        "Add new or existing files to the chat"

        if not args:
            self.io.tool_output("Usage: /add <file> [<file> ...]")
            return

        # Allow quoting paths with spaces
        # Need to handle both 'file.txt' and "file.txt"
        def split_args(s):
            inside_quotes = False
            quote_char = None
            parts = []
            current = []

            for c in s:
                if c in "\"'":
                    if not inside_quotes:
                        inside_quotes = True
                        quote_char = c
                    elif c == quote_char:
                        inside_quotes = False
                        quote_char = None
                    else:
                        current.append(c)
                elif c.isspace() and not inside_quotes:
                    if current:
                        parts.append("".join(current))
                        current = []
                else:
                    current.append(c)

            if current:
                parts.append("".join(current))

            # Strip quotes from parts
            return [p.strip("\"'") for p in parts]

        filenames = split_args(args)

        fnames_to_abs_fnames = {}
        for filename in filenames:
            # Handle glob patterns
            import glob

            # Verify path is not trying to escape root directory
            abs_path = os.path.abspath(filename)
            root_path = (
                os.path.abspath(self.coder.root) if self.coder.root else os.path.abspath(os.curdir)
            )
            if not abs_path.startswith(root_path):
                self.io.tool_error(f"Path {filename} is outside the root directory {root_path}")
                continue

            match_paths = glob.glob(filename, recursive=True)
            if match_paths:
                for match_path in match_paths:
                    try:
                        if os.path.isdir(match_path):
                            # Add all files in the directory
                            for root, _, files in os.walk(match_path):
                                for fname in files:
                                    file_path = os.path.join(root, fname)
                                    fnames_to_abs_fnames[file_path] = self.coder.abs_root_path(
                                        file_path
                                    )
                        else:
                            # Add individual file
                            fnames_to_abs_fnames[match_path] = self.coder.abs_root_path(match_path)
                    except OSError as e:
                        # This handles path-specific errors like permission issues
                        self.io.tool_error(f"Error accessing {match_path}: {e}")
            else:
                # If the file doesn't exist, create it if confirmed
                abs_fname = self.coder.abs_root_path(filename)
                if os.path.exists(abs_fname):
                    self.io.tool_error(f"File not found: {filename}")
                    continue

                if self.coder.repo and self.coder.repo.ignored_file(filename):
                    self.io.tool_error(f"File {filename} is ignored by git/aider")
                    continue

                if not self.io.confirm_ask(f"Create new file {filename}?", default="y"):
                    continue

                try:
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(abs_fname) or ".", exist_ok=True)
                    # Create the file
                    Path(abs_fname).touch()
                    fnames_to_abs_fnames[filename] = abs_fname
                except OSError as e:
                    self.io.tool_error(f"Error creating {filename}: {e}")

        for fname, abs_fname in fnames_to_abs_fnames.items():
            try:
                if self.coder.repo and self.coder.repo.ignored_file(fname):
                    self.io.tool_error(f"File {fname} is ignored by git/aider")
                    continue

                # Check if the file is in the read-only set
                if any(
                    os.path.samefile(abs_fname, ro_fname)
                    for ro_fname in self.coder.abs_read_only_fnames
                ):
                    # If the file is in the repository, we can add it
                    if self.coder.repo and self.coder.repo.file_in_repo(fname):
                        # Remove from read-only set
                        self.coder.abs_read_only_fnames.remove(abs_fname)
                    else:
                        # Skip read-only files that aren't in the repo
                        self.io.tool_warning(
                            f"File {fname} is currently read-only and not in the repository. Cannot"
                            " convert to editable."
                        )
                        continue

                encoding_errors = False
                if not is_image_file(fname):
                    # Test if we can read the file with the expected encoding
                    try:
                        content = self.io.read_text(abs_fname)
                        if content is None:
                            encoding_errors = True
                    except (UnicodeDecodeError, IsADirectoryError):
                        encoding_errors = True

                if encoding_errors:
                    self.io.tool_error(
                        f"Unable to read {fname} with encoding {self.io.encoding}. Skipping."
                    )
                    continue

                # Actually add the file
                if self.coder.allowed_to_edit(fname):
                    self.io.tool_output(f"Added {fname}")
                else:
                    self.io.tool_output(f"Skipped {fname}")

            except OSError as e:
                # Handle specific OSError subtypes
                if isinstance(e, IsADirectoryError):
                    self.io.tool_error(f"Cannot add {fname}: It is a directory")
                else:
                    self.io.tool_error(f"Error adding {fname}: {e}")

    def cmd_drop(self, args):
        "Drop files from the chat session"

        if not args:
            if self.original_read_only_fnames:
                self.io.tool_output(
                    "Dropping all files from the chat session except originally read-only files."
                )
                # Keep only the original read-only files
                to_keep = set()
                for abs_fname in self.coder.abs_read_only_fnames:
                    rel_fname = self.coder.get_rel_fname(abs_fname)
                    if (
                        abs_fname in self.original_read_only_fnames
                        or rel_fname in self.original_read_only_fnames
                    ):
                        to_keep.add(abs_fname)
                self.coder.abs_read_only_fnames = to_keep
                self.coder.abs_fnames = set()
            else:
                self.io.tool_output("Dropping all files from the chat session.")
                self.coder.abs_fnames = set()
                self.coder.abs_read_only_fnames = set()
            return

        # Allow quoting paths with spaces
        def split_args(s):
            inside_quotes = False
            quote_char = None
            parts = []
            current = []

            for c in s:
                if c in "\"'":
                    if not inside_quotes:
                        inside_quotes = True
                        quote_char = c
                    elif c == quote_char:
                        inside_quotes = False
                        quote_char = None
                    else:
                        current.append(c)
                elif c.isspace() and not inside_quotes:
                    if current:
                        parts.append("".join(current))
                        current = []
                else:
                    current.append(c)

            if current:
                parts.append("".join(current))

            # Strip quotes from parts
            return [p.strip("\"'") for p in parts]

        filenames = split_args(args)

        for filename in filenames:
            # Handle glob patterns
            import glob

            match_paths = glob.glob(filename, recursive=True)
            if match_paths:
                for match_path in match_paths:
                    if os.path.isdir(match_path):
                        # Drop all files in the directory
                        dir_path = os.path.abspath(match_path)
                        self.coder.abs_fnames = set(
                            f
                            for f in self.coder.abs_fnames
                            if not os.path.abspath(f).startswith(dir_path)
                        )
                        self.coder.abs_read_only_fnames = set(
                            f
                            for f in self.coder.abs_read_only_fnames
                            if not os.path.abspath(f).startswith(dir_path)
                        )
                        self.io.tool_output(f"Dropped all files in directory {match_path}")
                    else:
                        # Drop individual file
                        abs_path = os.path.abspath(match_path)
                        # Handle paths that need resolution
                        dropped = False
                        for f in list(self.coder.abs_fnames):
                            try:
                                if os.path.samefile(f, abs_path):
                                    self.coder.abs_fnames.remove(f)
                                    dropped = True
                                    self.io.tool_output(f"Dropped {match_path}")
                                    break
                            except OSError:
                                # Handle case where file doesn't exist anymore
                                continue

                        for f in list(self.coder.abs_read_only_fnames):
                            try:
                                if os.path.samefile(f, abs_path):
                                    self.coder.abs_read_only_fnames.remove(f)
                                    dropped = True
                                    self.io.tool_output(f"Dropped read-only file {match_path}")
                                    break
                            except OSError:
                                continue

                        if not dropped:
                            # Try with relative paths
                            abs_fname = self.coder.abs_root_path(match_path)
                            if abs_fname in self.coder.abs_fnames:
                                self.coder.abs_fnames.remove(abs_fname)
                                dropped = True
                                self.io.tool_output(f"Dropped {match_path}")
                            elif abs_fname in self.coder.abs_read_only_fnames:
                                self.coder.abs_read_only_fnames.remove(abs_fname)
                                dropped = True
                                self.io.tool_output(f"Dropped read-only file {match_path}")

                        if not dropped:
                            # One more check with short filenames
                            for f in list(self.coder.abs_fnames):
                                rel_f = self.coder.get_rel_fname(f)
                                if rel_f == match_path or os.path.basename(f) == match_path:
                                    self.coder.abs_fnames.remove(f)
                                    dropped = True
                                    self.io.tool_output(f"Dropped {match_path}")
                                    break

                            for f in list(self.coder.abs_read_only_fnames):
                                rel_f = self.coder.get_rel_fname(f)
                                if rel_f == match_path or os.path.basename(f) == match_path:
                                    self.coder.abs_read_only_fnames.remove(f)
                                    dropped = True
                                    self.io.tool_output(f"Dropped read-only file {match_path}")
                                    break

                        if not dropped:
                            self.io.tool_error(f"File {match_path} not found in chat")
            else:
                # Try to find the file using various strategies
                abs_fname = self.coder.abs_root_path(filename)
                dropped = False

                # First try direct match
                if abs_fname in self.coder.abs_fnames:
                    self.coder.abs_fnames.remove(abs_fname)
                    dropped = True
                    self.io.tool_output(f"Dropped {filename}")
                elif abs_fname in self.coder.abs_read_only_fnames:
                    self.coder.abs_read_only_fnames.remove(abs_fname)
                    dropped = True
                    self.io.tool_output(f"Dropped read-only file {filename}")

                if not dropped:
                    # Try by filename only
                    for f in list(self.coder.abs_fnames):
                        if os.path.basename(f) == filename:
                            self.coder.abs_fnames.remove(f)
                            dropped = True
                            self.io.tool_output(f"Dropped {filename}")
                            break

                    for f in list(self.coder.abs_read_only_fnames):
                        if os.path.basename(f) == filename:
                            self.coder.abs_read_only_fnames.remove(f)
                            dropped = True
                            self.io.tool_output(f"Dropped read-only file {filename}")
                            break

                if not dropped:
                    # Try by relative filename
                    for f in list(self.coder.abs_fnames):
                        rel_f = self.coder.get_rel_fname(f)
                        if rel_f == filename:
                            self.coder.abs_fnames.remove(f)
                            dropped = True
                            self.io.tool_output(f"Dropped {filename}")
                            break

                    for f in list(self.coder.abs_read_only_fnames):
                        rel_f = self.coder.get_rel_fname(f)
                        if rel_f == filename:
                            self.coder.abs_read_only_fnames.remove(f)
                            dropped = True
                            self.io.tool_output(f"Dropped read-only file {filename}")
                            break

                if not dropped:
                    try:
                        # Last attempt using samefile for path normalization
                        abs_path = os.path.abspath(filename)
                        for f in list(self.coder.abs_fnames):
                            try:
                                if os.path.samefile(f, abs_path):
                                    self.coder.abs_fnames.remove(f)
                                    dropped = True
                                    self.io.tool_output(f"Dropped {filename}")
                                    break
                            except OSError:
                                continue

                        for f in list(self.coder.abs_read_only_fnames):
                            try:
                                if os.path.samefile(f, abs_path):
                                    self.coder.abs_read_only_fnames.remove(f)
                                    dropped = True
                                    self.io.tool_output(f"Dropped read-only file {filename}")
                                    break
                            except OSError:
                                continue
                    except OSError:
                        pass

                if not dropped:
                    self.io.tool_error(f"File {filename} not found in chat")

    def cmd_read_only(self, args):
        "Add read-only files to the chat (won't be edited)"
        
        if not args:
            # If no arguments, convert all editable files to read-only
            if self.coder.abs_fnames:
                to_convert = list(self.coder.abs_fnames)
                self.coder.abs_read_only_fnames.update(to_convert)
                self.coder.abs_fnames.clear()
                self.io.tool_output("Converted all editable files to read-only mode.")
            else:
                self.io.tool_output("No editable files to convert. Use /read-only <file> to add read-only files.")
            return

        # Allow quoting paths with spaces
        def split_args(s):
            inside_quotes = False
            quote_char = None
            parts = []
            current = []

            for c in s:
                if c in "\"'":
                    if not inside_quotes:
                        inside_quotes = True
                        quote_char = c
                    elif c == quote_char:
                        inside_quotes = False
                        quote_char = None
                    else:
                        current.append(c)
                elif c.isspace() and not inside_quotes:
                    if current:
                        parts.append("".join(current))
                        current = []
                else:
                    current.append(c)

            if current:
                parts.append("".join(current))

            # Strip quotes from parts
            return [p.strip("\"'") for p in parts]

        filenames = split_args(args)
        added_files = []

        for filename in filenames:
            # Handle glob patterns
            import glob

            # Handle home directory expansion
            filename = os.path.expanduser(filename)

            match_paths = glob.glob(filename, recursive=True)
            if match_paths:
                for match_path in match_paths:
                    try:
                        if os.path.isdir(match_path):
                            # Add all files in the directory
                            for root, _, files in os.walk(match_path):
                                for fname in files:
                                    file_path = os.path.join(root, fname)
                                    abs_fname = os.path.abspath(file_path)
                                    if self._add_read_only_file(abs_fname):
                                        added_files.append(file_path)
                        else:
                            # Add individual file
                            abs_fname = os.path.abspath(match_path)
                            if self._add_read_only_file(abs_fname):
                                added_files.append(match_path)
                    except OSError as e:
                        self.io.tool_error(f"Error accessing {match_path}: {e}")
            else:
                self.io.tool_error(f"No matches found for: {filename}")

        if added_files:
            self.io.tool_output(f"Added {len(added_files)} files as read-only")

    def _add_read_only_file(self, abs_fname):
        """Helper to add a single read-only file."""
        # Check if file can be read with the current encoding
        try:
            # Special case for images with vision models
            if is_image_file(abs_fname):
                if hasattr(self.coder.main_model, "name") and "vision" in self.coder.main_model.name:
                    self.coder.abs_read_only_fnames.add(abs_fname)
                    return True
                else:
                    self.io.tool_error(
                        f"Image files like {os.path.basename(abs_fname)} require a vision model."
                    )
                    return False
            
            # Try to read the file content
            content = self.io.read_text(abs_fname)
            if content is None:
                return False
            
            # Check if already in editable files
            for existing in self.coder.abs_fnames:
                try:
                    if os.path.samefile(existing, abs_fname):
                        # Remove from editable and add to read-only
                        self.coder.abs_fnames.remove(existing)
                        self.coder.abs_read_only_fnames.add(abs_fname)
                        self.io.tool_output(
                            f"Converted {os.path.basename(abs_fname)} from editable to read-only."
                        )
                        return True
                except OSError:
                    continue
            
            # Add as new read-only file if not already present
            for existing in self.coder.abs_read_only_fnames:
                try:
                    if os.path.samefile(existing, abs_fname):
                        # Already in read-only files
                        return False
                except OSError:
                    continue
            
            # Add new read-only file
            self.coder.abs_read_only_fnames.add(abs_fname)
            return True
            
        except (UnicodeDecodeError, IsADirectoryError) as e:
            self.io.tool_error(f"Error reading {os.path.basename(abs_fname)}: {e}")
            return False
        except Exception as e:
            self.io.tool_error(f"Unexpected error with {os.path.basename(abs_fname)}: {e}")
            return False

    def cmd_files(self, args):
        "List all files in the chat session"
        editable_files = [self.coder.get_rel_fname(f) for f in self.coder.abs_fnames]
        read_only_files = [self.coder.get_rel_fname(f) for f in self.coder.abs_read_only_fnames]
        
        # Sort files for better readability
        editable_files.sort()
        read_only_files.sort()
        
        if not editable_files and not read_only_files:
            self.io.tool_output("No files in the chat session. Use /add or /read-only to add files.")
            return
        
        if editable_files:
            self.io.tool_output("\nEditable files:")
            for f in editable_files:
                self.io.tool_output(f"  - {f}")
        
        if read_only_files:
            self.io.tool_output("\nRead-only files:")
            for f in read_only_files:
                self.io.tool_output(f"  - {f}")

    def cmd_capture(self, args):
        "Capture and share a screenshot"
        try:
            # Try to get a screenshot from clipboard
            image = None
            try:
                image = ImageGrab.grabclipboard()
                if image:
                    self.io.tool_output("Found image in clipboard.")
            except Exception:
                pass
            
            # If no clipboard image, take a screenshot
            if not image:
                self.io.tool_output("Taking screenshot... (click to capture)")
                image = ImageGrab.grab()
                
            if not image:
                self.io.tool_error("Failed to capture screenshot.")
                return
            
            # Save the image to a temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            image.save(temp_path)
            self.io.tool_output(f"Screenshot saved temporarily as {temp_path}")
            
            # Check if the model supports images
            if not hasattr(self.coder.main_model, "name") or "vision" not in self.coder.main_model.name:
                self.io.tool_warning(
                    "Your current model doesn't support images. The screenshot will be referenced but not processed."
                )
            
            # Add to chat with a message
            message = f"Here's a screenshot I captured:\n\n![Screenshot]({temp_path})"
            self.coder.run(with_message=message)
            
        except Exception as e:
            self.io.tool_error(f"Error capturing screenshot: {e}")
