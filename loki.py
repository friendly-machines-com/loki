#!/usr/bin/env python3

# TODO: For each guest file, remember what the agent has seen, and what it has written.  If it writes file X, it has to read all of X again before it can write again.# TODO: chat (and file) history, resuming, rewinding
# TODO: Provide command to set effort level
# TODO: /goal
# TODO: paste support ? maybe not; automatic; weird 4096 Byte length limit ?  It's especially good so pasting something doesnt send 237 requests in a row
# TODO: mouse support; but what for?
# TODO: input with readline support (just print the text you have so far--up to the cursor)
# TODO: maybe sixel bitmap support; but what for?
# TODO: background tasks and job control, maybe
# TODO: also support anthropic protocol (in addition to the old openai chat protocol we do support)
# TODO: make this an actual shell; pipeable and so on like always
# TODO: use getopt getopt to support the options "resume <uuid>" and "--resume <uuid>" with either just the uuid or a filename, then uses that as the chat log (LOADING it into MESSAGES) and prints that out like it just happened.

import sys
import re
import os
import json
import time
import urllib.request
import urllib.error
import urllib.parse
import subprocess
import signal
import socket
import uuid
import getopt
from pprint import pprint

url = "https://opencode.ai/zen/go/v1/chat/completions" # "https://api.openai.com/v1/chat/completions"
model = 'glm-5.2'
models = ['hy3-preview', 'glm-5.2', 'glm-5.1', 'kimi-k2.7', 'kimi-k2.6', 'deepseek-v4-pro', 'deepseek-v4-flash', 'mimo-v2.5', 'mimo-v2.5-pro'] # TODO: use request url to find out: <https://opencode.ai/zen/go/v1/models> => json data id

computer = socket.gethostname()

ERROR_COLOR = 1
STATUS_COLOR = 4
INPUT_COLOR = 7
TOOL_CALL_COLOR = 5

MAX_LOOP_LIMIT = 30

netloc = urllib.parse.urlparse(url).netloc

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if OPENAI_API_KEY:
    api_key = OPENAI_API_KEY
    del os.environ['OPENAI_API_KEY']
else:
    res = subprocess.run(['secret-tool', 'lookup', 'domain', netloc], shell=False, capture_output=True, text=True)
    api_key = res.stdout.strip()

if not api_key:
    raise ValueError('API key missing.  Please run secret-tool store --label="opencode API key" domain {}'.format(netloc))

# TODO: make this terminal a dummy if not os.isatty(sys.stdout)
class terminal:
    def clear_screen():
        print('\033[2J', end='')

    def clear_to_end_of_screen(): # always, not relative.
        print('\033[J', end='')

    def goto_position(row, column):
        print('\033[{};{}H'.format(row, column), end='')

    def set_clipping_region(first_row, last_row): # note: after that, cursor position is (1,1) ABSOLUTE OR RELATIVE DEPENDING ON origin_mode
        assert last_row - first_row >= 2 # otherwise not supported.
        print('\033[{};{}r'.format(first_row, last_row - 1), end='')
		#goto_position(1, 1)

    def disable_clipping_regions(): # note: after that, cursor position is (1,1) either absolute or relative dependig on origin_mode.
        print('\033[r', end='')
		#goto_position(1, 1)

    def save_cursor_position():
        print('\033[s', end='')

    def restore_cursor_position():
        print('\033[u', end='')

    def flush():
        sys.stdout.flush()

    def enable_origin_mode(): # relative coordinates
        print('\033[?6h', end='')

    def disable_origin_mode():
        print('\033[?6l', end='')

    def set_foreground_color(index):
        print('\033[{}m'.format(30 + index), end='')

    def set_background_color(index):
        print('\033[{}m'.format(40 + index), end='')

    def reset_colors_and_flags():
        print('\033[m', end='')

    def enable_bracketed_paste_mode(): # \e[200~ ... \e[201~
        print('\033[?2004h', end='')

    def disable_bracketed_paste_mode():
        print('\033[?2004l', end='')

	# TODO: set raw

# TODO: make this terminal a dummy if not os.isatty(sys.stdout)
def markdown_to_ansi(text: str) -> str:
    # Standard ANSI escape codes
    BOLD = '\033[1m'
    CYAN = '\033[36m'  # Using Cyan to make inline code pop
    RESET = '\033[0m'
    # We split the text by inline code blocks.
    # Using a capture group `(`.*?`)` ensures the code segments are kept in the resulting list.
    parts = re.split(r'(`.*?`)', text)
    for i, part in enumerate(parts):
        # Check if the current part is a code block
        if part.startswith('`') and part.endswith('`') and len(part) >= 2:
            inner_text = part[1:-1] # Strip the backticks
            parts[i] = f"{CYAN}{inner_text}{RESET}"
        else: # normal text
            # .*? is non-greedy so it correctly matches isolated **bold** pairs
            part = re.sub(r'\*\*(.*?)\*\*', f'{BOLD}\\1{RESET}', part) # bold; TODO: also __foo__ also bold!
            part = re.sub(r'(?<!\*)\*(.*?)\*(?!\*)', f'\033[3m\\1{RESET}', part) # italics
            # TODO: maybe underline.
            # TODO: maybe strikethrough text: ~
            # TODO: # headline, ## headline, ### headline; also extra line ===== or ----
            # TODO: handle >blockquote blocks
            # TODO: ![Tux, the Linux mascot](/assets/images/tux.png)
            # TODO: \* Without the backslash, this would be a bullet in an unordered list.
            parts[i] = part

    return "".join(parts)

try:
    terminal_size = os.get_terminal_size()
    terminal_lines = terminal_size.lines
except OSError:
    terminal_lines = 25

output_area = 1, terminal_lines - 4
input_area = terminal_lines - 4, terminal_lines - 2
status_area = terminal_lines - 2, terminal_lines # too big, but that's the minimum supported height of set_clipping_region

'''
    output area
    question area
        regular question
        diff viewer (maybe huge with actual scrolling need!)
    input area
    tab to switch to next area
    status line
'''

def update_status_bar():
    terminal.set_clipping_region(*status_area)
    terminal.goto_position(1, 1)
    terminal.set_background_color(STATUS_COLOR)
    terminal.clear_to_end_of_screen()
    print('Model: {}; Hint: Use /quit to quit, /model to switch model, !foo to execute shell command foo'.format(model), end='')

def get_input(prompt=None):
    terminal.set_clipping_region(*input_area)
    terminal.goto_position(1, 1)
    terminal.set_background_color(INPUT_COLOR)
    terminal.clear_to_end_of_screen() # also clears status bar
    update_status_bar()
    terminal.set_clipping_region(*input_area)
    terminal.goto_position(1, 1)
    terminal.set_background_color(INPUT_COLOR)
    terminal.flush()
    # TODO: read raw stuff
    text = input(prompt or 'User: ')
    #terminal.save_cursor_position()
    terminal.set_clipping_region(*output_area)
    terminal.restore_cursor_position()
    return text

def run_menu(items): # TODO: use.
    # Menus can be large--so show it in the output area.
    terminal.save_cursor_position()
    while True:
        terminal.restore_cursor_position()
        for i, item in enumerate(items):
            print(i + 1, '.', item)
        #terminal.save_cursor_position()
        selection = input('User choice: ') # TODO: just use get_input; and do terminal.save_cursor_position() before here.
        selection = selection.strip()
        if selection.endswith('.'):
            selection = selection.rstrip('.')

        if selection:
            terminal.save_cursor_position()
            try:
                return items[int(selection) - 1]
            except:
                return selection

    terminal.save_cursor_position()
    return 'None of the above'

def run_input_loop():
  while True:
    input_text = get_input()

    terminal.reset_colors_and_flags()
    terminal.set_clipping_region(*output_area)
    terminal.restore_cursor_position()

    yield input_text
    #print('blahblah')
    # Update cursor position for next output
    terminal.save_cursor_position()

    #terminal.disable_origin_mode()
    #break

def run_bash(command: str) -> str:
    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True)
        return res.stdout + res.stderr or "Success (No output)"
    except Exception as e:
        return f"Error: {e}"

def run_read(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error: {e}"

def run_patch(path: str, old_str: str, new_str: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = f.read()
        if old_str not in data:
            return "Error: old_str not found in file."
        # FIXME: what if needle is there more than once?
        with open(path, 'w', encoding='utf-8') as f:
            f.write(data.replace(old_str, new_str, 1))
        return f"Successfully patched {path}"
    except Exception as e:
        return f"Error: {e}"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command on the local machine",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write text content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "patch",
            "description": "Replace a specific string in an existing file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"}
                },
                "required": ["path", "old_str", "new_str"]
            }
        }
    }
]

def chat_completion(messages: list) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "TinyAgent/1.0",
    }
    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except OSError as e:
        terminal.set_background_color(ERROR_COLOR)
        print(f"API Error for <{url}>: {e}")
        terminal.reset_colors_and_flags()
        return None

    elapsed = time.perf_counter() - start
    print(f"\n[T]  [LLM Response Time: {elapsed:.3f}s]")
    return data

messages = []

def new_chat_log(filename):
    global chat_log
    chat_log = open(filename, 'w')

def save_chat_log():
    chat_log.seek(0)
    json.dump(messages, chat_log, indent=4)
    chat_log.truncate()
    chat_log.flush()
    print('Note: Saved chat log to {}'.format(chat_log.name), file=sys.stderr)
    sys.stderr.flush()

def load_chat_log(filename):
    global chat_log
    global messages
    chat_log = open(filename, 'r')
    try:
        messages = json.load(chat_log)
        for message in messages:
            for k, v in message.items():
                pprint((k, v)) # TODO: nicer

            print()

        print('----')
    finally:
        chat_log.close()

    chat_log = open(filename, 'w')
    save_chat_log()

def main():
    global model
    global messages
    options, args = getopt.getopt(sys.argv[1:], 'r:', ['resume='])
    try:
        log_filename = None
        for option_name, option_value in options:
            print(option_name)
            time.sleep(2)
            if option_name == '--resume' or option_name == '-r':
                log_filename = option_value

        if args[0:1] == ['resume']:
            log_filename = args[1]

        if log_filename:
            if not os.path.exists(log_filename):
                log_filename = 'chat-{}.json'.format(log_filename)
            load_chat_log(log_filename)
        else:
            log_filename = 'chat-{}.json'.format(str(uuid.uuid4()))
            new_chat_log(log_filename)

    except IndexError: # TODO: remove
        log_filename = 'chat-{}.json'.format(str(uuid.uuid4()))
        new_chat_log(log_filename)

    messages = [{
        "role": "system",
        "content": "You are a helpful system agent capable of running bash commands, reading, writing, and patching files."
    }]
    for user_in in run_input_loop():
        if not user_in:
            continue

        print('User:', user_in)
        match user_in.strip():
            case '/quit':
                break
            case '/model':
                model = run_menu(models)
                continue
            case _:
                if user_in.strip().startswith('!'): # direct command execution
                    cmd = user_in[1:].strip()
                    print(f"{computer}: [Running local command: {cmd}]")
                    cmd_output = run_bash(cmd)
                    print(cmd_output) # Show output to you in the terminal
                    # Morph the user input so the AI sees exactly what you did and the result
                    user_in = f"I ran the local command `{cmd}`.\nOutput:\n```\n{cmd_output}\n```"
                    #continue
                else:
                    pass

        messages.append({"role": "user", "content": user_in})

        loop_count = 0
        try:
            while True:
                loop_count += 1
                if loop_count > MAX_LOOP_LIMIT:
                    print("\n[!] [Max Loop Limit Reached - Stopping Autonomous Execution]")
                    messages.append({"role": "system", "content": "Max tool loop limit reached. Ask the user for further instructions."})
                    break

                try:
                    resp = chat_completion(messages)
                except OSError as e:
                    print(f"\n{computer}: NETWORK ERROR: {e}")
                    break

                if not resp:
                    break

                msg = resp["choices"][0]["message"]
                clean_msg = {"role": msg["role"]}
                if msg.get("content"): clean_msg["content"] = markdown_to_ansi(msg["content"])
                if msg.get("tool_calls"): clean_msg["tool_calls"] = msg["tool_calls"]
                messages.append(clean_msg)
                if clean_msg.get("content"):
                    print(f"\n{model}: {clean_msg['content']}")

                if not clean_msg.get("tool_calls"):
                    break # Turn complete

                for tc in clean_msg["tool_calls"]:
                    fn_name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"])
                        terminal.set_foreground_color(TOOL_CALL_COLOR)
                        print(f"{computer}: Executing Tool: {fn_name} with args:")
                        for k, v in args.items():
                            pprint((k, v))

                        if fn_name == "bash":
                            result = run_bash(args.get("command", ""))
                        elif fn_name == "read":
                            result = run_read(args.get("path", ""))
                        elif fn_name == "write":
                            result = run_write(args.get("path", ""), args.get("content", ""))
                        elif fn_name == "patch":
                            result = run_patch(args.get("path", ""), args.get("old_str", ""), args.get("new_str", ""))
                        else:
                            result = "Unknown function"

                        terminal.reset_colors_and_flags()
                    except Exception as e:
                        result = f"Failed parsing arguments or executing: {e}"
                        terminal.set_background_color(ERROR_COLOR)
                        print(result) # ALSO for us; mainly the result variable itself for the agent.
                        terminal.reset_colors_and_flags()

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": fn_name,
                        "content": str(result)
                    })
        except KeyboardInterrupt:
            terminal.reset_colors_and_flags()
            # ? EMERGENCY BRAKE
            print("\n\n? [EMERGENCY STOP] Agent execution cancelled by user!")
            # If we interrupt while a tool call was requested but unanswered, the API will crash on 
            # the next request. We must surgically remove the unanswered tool request from history.
            if messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls"):
                messages.pop()

            messages.append({
                "role": "system",
                "content": "CRITICAL: The user forcefully stopped your execution via KeyboardInterrupt (Ctrl+C). You were likely looping, making a mistake, or doing something dangerous. Await new instructions."
            })
            continue # Drop immediately back to the User> prompt

if __name__ == '__main__':
    def clean_up(*args, **kwargs):
        save_chat_log()
        terminal.disable_bracketed_paste_mode()
        terminal.disable_clipping_regions()
        terminal.disable_origin_mode()
        terminal.reset_colors_and_flags()
        terminal.clear_screen()

    def clean_up_and_exit(*args, **kwargs):
        clean_up(*args, **kwargs)
        sys.exit(1)

    signal.signal(signal.SIGTERM, clean_up_and_exit)

    terminal.enable_bracketed_paste_mode()
    terminal.enable_origin_mode()
    terminal.clear_screen()

    terminal.reset_colors_and_flags()
    terminal.set_clipping_region(*output_area)
    terminal.goto_position(1, 1)
    terminal.save_cursor_position()

    try:
        main()
        clean_up()
    except Exception as e:
        clean_up()
        raise
    except KeyboardInterrupt:
        clean_up()
