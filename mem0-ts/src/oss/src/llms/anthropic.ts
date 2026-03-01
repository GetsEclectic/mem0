import Anthropic from "@anthropic-ai/sdk";
import { LLM, LLMResponse } from "./base";
import { LLMConfig, Message } from "../types";

// Required beta headers for Anthropic OAuth tokens to work.
// Without these, Anthropic returns 401 "OAuth authentication is currently not supported."
// See: https://github.com/anthropics/claude-code/blob/main/src/agents/pi-embedded-runner/extra-params.ts
const OAUTH_REQUIRED_BETAS = ["oauth-2025-04-20", "claude-code-20250219"].join(
  ",",
);

export class AnthropicLLM implements LLM {
  private client: Anthropic;
  private model: string;

  constructor(config: LLMConfig) {
    // Support both OAuth tokens (authToken) and API keys (apiKey)
    // OAuth tokens use Authorization: Bearer header, API keys use X-Api-Key header
    let authToken = config.authToken || process.env.ANTHROPIC_AUTH_TOKEN;
    let apiKey = config.apiKey || process.env.ANTHROPIC_API_KEY;

    // Auto-detect OAuth tokens passed as apiKey
    if (apiKey && apiKey.startsWith("sk-ant-oat") && !authToken) {
      authToken = apiKey;
      apiKey = undefined;
    }

    if (!authToken && !apiKey) {
      throw new Error("Anthropic API key or auth token is required");
    }

    // Prefer authToken if provided (OAuth tokens require Bearer header)
    // OAuth tokens also require specific beta headers to be accepted.
    // When using authToken, explicitly set apiKey to null to prevent the SDK
    // from auto-reading ANTHROPIC_API_KEY env var and sending both headers.
    if (authToken) {
      this.client = new Anthropic({
        apiKey: null,
        authToken,
        defaultHeaders: {
          "anthropic-beta": OAUTH_REQUIRED_BETAS,
        },
      });
    } else {
      this.client = new Anthropic({ apiKey });
    }
    this.model = config.model || "claude-3-sonnet-20240229";
  }

  async generateResponse(
    messages: Message[],
    responseFormat?: { type: string },
  ): Promise<string> {
    // Extract system message if present
    const systemMessage = messages.find((msg) => msg.role === "system");
    const otherMessages = messages.filter((msg) => msg.role !== "system");

    // Handle JSON mode - Anthropic doesn't have native JSON mode like OpenAI,
    // so we enforce it via system prompt modification
    const wantsJson = responseFormat?.type === "json_object";
    let systemContent =
      typeof systemMessage?.content === "string"
        ? systemMessage.content
        : undefined;

    if (wantsJson) {
      const jsonInstruction =
        "\n\nCRITICAL: Respond with valid JSON only. No markdown, no code blocks, no backticks, no explanation - output ONLY the raw JSON object starting with { and ending with }.";
      systemContent = systemContent
        ? systemContent + jsonInstruction
        : jsonInstruction.trim();
    }

    // Build messages array
    const apiMessages: Array<{ role: "user" | "assistant"; content: string }> =
      otherMessages.map((msg) => ({
        role: msg.role as "user" | "assistant",
        content:
          typeof msg.content === "string"
            ? msg.content
            : msg.content.image_url.url,
      }));

    const response = await this.client.messages.create({
      model: this.model,
      messages: apiMessages,
      system: systemContent,
      max_tokens: 4096,
    });

    const firstBlock = response.content[0];
    if (firstBlock.type === "text") {
      return firstBlock.text;
    } else {
      throw new Error("Unexpected response type from Anthropic API");
    }
  }

  async generateChat(messages: Message[]): Promise<LLMResponse> {
    const response = await this.generateResponse(messages);
    return {
      content: response,
      role: "assistant",
    };
  }
}
