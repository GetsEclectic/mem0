import json
import os
from typing import Dict, List, Optional, Union

try:
    import anthropic
except ImportError:
    raise ImportError("The 'anthropic' library is required. Please install it using 'pip install anthropic'.")

from mem0.configs.llms.anthropic import AnthropicConfig
from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase


class AnthropicLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, AnthropicConfig, Dict]] = None):
        # Convert to AnthropicConfig if needed
        if config is None:
            config = AnthropicConfig()
        elif isinstance(config, dict):
            config = AnthropicConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, AnthropicConfig):
            # Convert BaseLlmConfig to AnthropicConfig
            config = AnthropicConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
                http_client_proxies=config.http_client,
            )

        super().__init__(config)

        if not self.config.model:
            self.config.model = "claude-3-5-sonnet-20240620"

        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        auth_token = self.config.auth_token or os.getenv("ANTHROPIC_AUTH_TOKEN")

        # Auto-detect OAuth tokens passed as api_key
        if api_key and api_key.startswith("sk-ant-oat") and not auth_token:
            auth_token = api_key
            api_key = None

        self._is_oauth = bool(auth_token)

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if auth_token:
            client_kwargs["auth_token"] = auth_token
        if self.config.anthropic_base_url:
            client_kwargs["base_url"] = self.config.anthropic_base_url

        # OAuth tokens require specific beta headers to be accepted
        if self._is_oauth:
            client_kwargs["default_headers"] = {
                "anthropic-beta": "oauth-2025-04-20,interleaved-thinking-2025-05-14",
            }

        self.client = anthropic.Anthropic(**client_kwargs)

    @staticmethod
    def _convert_tool(tool: Dict) -> Dict:
        """Convert OpenAI-format tool to Anthropic format."""
        if tool.get("type") == "function" and "function" in tool:
            func = tool["function"]
            return {
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            }
        # Already in Anthropic format or unknown format, pass through
        return tool

    def _parse_response(self, response, tools):
        """Parse Anthropic response, converting tool_use blocks to mem0's expected format."""
        if tools:
            processed = {"content": None, "tool_calls": []}
            for block in response.content:
                if block.type == "text":
                    processed["content"] = block.text
                elif block.type == "tool_use":
                    processed["tool_calls"].append({
                        "name": block.name,
                        "arguments": block.input if isinstance(block.input, dict) else json.loads(block.input),
                    })
            return processed
        return response.content[0].text

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Generate a response based on the given messages using Anthropic.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            **kwargs: Additional Anthropic-specific parameters.

        Returns:
            str: The generated response.
        """
        # Separate system message from other messages
        system_message = ""
        filtered_messages = []
        for message in messages:
            if message["role"] == "system":
                system_message = message["content"]
            else:
                filtered_messages.append(message)

        params = self._get_supported_params(messages=messages, **kwargs)
        # Anthropic doesn't allow both temperature and top_p
        if "temperature" in params and "top_p" in params:
            del params["top_p"]
        params.update(
            {
                "model": self.config.model,
                "messages": filtered_messages,
                "system": system_message,
            }
        )

        if tools:  # TODO: Remove tools if no issues found with new memory addition logic
            params["tools"] = [self._convert_tool(t) for t in tools]
            # Anthropic API expects tool_choice as a dict, e.g. {"type": "auto"}
            if isinstance(tool_choice, str):
                params["tool_choice"] = {"type": tool_choice}
            else:
                params["tool_choice"] = tool_choice

        response = self.client.messages.create(**params)
        return self._parse_response(response, tools)
