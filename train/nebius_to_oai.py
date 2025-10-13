import re
import shlex
import yaml
import json
from datasets import load_dataset

TOOL_CALL_PATTERN = re.compile(r"```\n(.*?)\n```", re.DOTALL)

def parse_tool_call(tool_call_block: str, tools: list[dict]) -> dict:
    if tool_call_block.startswith("edit"):
        header, replacement_text = tool_call_block.split("\n", 1)
        _, lines = header.split(" ", 1)
        start_line, end_line = map(int, lines.split(":"))

        arguments = json.dumps({
            "start_line": start_line,
            "end_line": end_line,
            "replacement_text": replacement_text.removesuffix("end_of_edit")
        })

        return {
            "name": "edit",
            "arguments": arguments
        }
    
    tool_parts = shlex.split(tool_call_block)
    tool_name = tool_parts[0]
    tool_args = tool_parts[1:]

    matching_tool = next((tool for tool in tools if tool["function"]["name"] == tool_name), None)
    if not matching_tool:
        return {
            "name": "bash",
            "arguments": json.dumps({"command": tool_call_block})
        }

    arguments = {}
    if tool_args:
        tool_properties = matching_tool["function"]["parameters"]["properties"]
        if (len(tool_args) > len(tool_properties)):
            tool_args = tool_args[:len(tool_properties)] + [" ".join(tool_args[len(tool_properties):])]

        for arg, key in zip(tool_args, tool_properties.keys()):
            arg_type = tool_properties[key]["type"]
            if arg_type == "integer":
                arg = int(arg)
            elif arg_type == "number":
                arg = float(arg)
            elif arg_type == "boolean":
                arg = arg.lower() == "true"

            arguments[key] = arg

    arguments_json = json.dumps(arguments)

    return {
        "name": tool_name,
        "arguments": arguments_json
    }
    

def parse_tool_calls_from_message(message: str, tools: list[dict]) -> tuple[list[dict], str]:
    message = message.strip()

    if not "```" in message:
        return [], message
    
    try:
        tool_calls_start = message.find("```") + 3

        matches = TOOL_CALL_PATTERN.findall(message)
        if not matches:
            raise ValueError("No tool call found in message.")
        
        tool_call_blocks = [match.strip() for match in matches]
        message_without_call = message[:tool_calls_start - 3].strip()

        tool_calls = [parse_tool_call(block, tools) for block in tool_call_blocks]

        return tool_calls, message_without_call

    except Exception as e:
        raise ValueError(f"Error parsing tool call: {e}")


def fix_signature(yaml_content):
    lines = yaml_content.splitlines()
    fixed_lines = []
    signature_combined = None

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("signature:"):
            signature_combined = stripped[len("signature:"):].strip()
            continue
        elif signature_combined is not None:
            if "<replacement_text>" in stripped or "end_of_edit" in stripped:
                signature_combined += f" {stripped}"
                continue
            else:
                fixed_lines.append("  signature: |")
                fixed_lines.append("    " + signature_combined.strip())
                signature_combined = None

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def parse_tool_definitions(system_message):
    commands_start = system_message.find("COMMANDS:")
    commands_end = system_message.find("\n\n\n", commands_start)
    assert commands_start != -1 and commands_end != -1, "COMMANDS section not found or improperly formatted."

    commands_section = system_message[commands_start + len("COMMANDS:"):commands_end].strip()
    cleaned_system_prompt = system_message[:commands_start].strip() + system_message[commands_end + 3:].strip()

    try:
        commands_data = yaml.safe_load(fix_signature(commands_section))
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing command YAML: {e}")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "You're free to use any other bash commands you want (e.g. find, grep, cat, ls, cd).However, the environment does NOT support interactive session commands (e.g. python, vim), so please do not invoke them.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Whole command with arguments."
                        }
                    },
                    "required": ["command"],
                    "additionalProperties": False
                }
            }
        }
    ]
    for name, command in commands_data.items():
        assert "docstring" in command, f"Command '{name}' must have a 'docstring'."

        properties = {}
        required = []
        if "arguments" not in command:
            command["arguments"] = []

        for arg in command["arguments"]:
            arg_def, arg_doc = list(arg.items())[0]
            arg_parts = arg_def.split(" ")
            arg_name = arg_parts[0]
            arg_type = arg_parts[1].strip("()")
            is_required = "[required]" in arg_parts

            properties[arg_name] = {"type": arg_type}
            properties[arg_name]["description"] = arg_doc.strip()
            if is_required:
                required.append(arg_name)

        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": command["docstring"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False
                },
                "strict": True
            }
        })

    return tools, cleaned_system_prompt



def nebius_to_oai(entry):
    messages = []
    tools = []
    trajectory = entry["trajectory"]

    try:
        assert len(trajectory) >= 2, "Each trajectory must have at least one system message and one user message."

        first_message = trajectory[0]
        assert first_message["role"] == "system", "First message must be a system message."
        tools, cleaned_system_prompt = parse_tool_definitions(first_message["system_prompt"])
        messages.append({
            "role": "system",
            "content": cleaned_system_prompt,
        })

        second_message = trajectory[1]
        assert second_message["role"] == "user", "Second message must be a user message."
        messages.append({
            "role": "user",
            "content": second_message["text"]
        })

        for message in trajectory[2:]:
            assert message["role"] != "system", "Subsequent messages should not have system messages"

            if message["role"] == "user":  # Tool call result
                messages.append({
                    "role": "tool",
                    "content": message["text"]
                })
            elif message["role"] == "ai":  # AI response
                tool_calls, thought = parse_tool_calls_from_message(message["text"], tools)
                messages.append({
                    "role": "assistant",
                    "content": thought,
                    "tool_calls": tool_calls
                })
            else:
                raise ValueError(f"Unexpected role: {message['role']} in trajectory.")
    except Exception as e:
        messages = None

        # print(f"SKIPPING DUE TO: {e}")

    entry["trajectory"] = {
        "messages": messages,
        "tools": tools
    }

    return entry

if __name__ == '__main__':
    dataset = load_dataset("nebius/SWE-agent-trajectories")["train"]
    dataset = dataset.map(nebius_to_oai, batched=False, num_proc=16,  desc="Parallel Processing")
    processed_dataset_path = "./processed_dataset"
    dataset.save_to_disk(processed_dataset_path)

