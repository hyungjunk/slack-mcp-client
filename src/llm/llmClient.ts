import { OpenAI } from "openai";
import ollama from 'ollama'
import { getOrThrow } from "../shared/utils.js";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import logger from "../shared/logger.js";
import type { Tool } from "../mcp/Tool.js";

export type OpenAiTool = {
    type: "function";
    strict: boolean;
    function: {
        name: string;
        description: string;
        parameters: {
            type: "object";
            properties?: Record<string, { type: string; description: string }>;
            required?: string[];
            additionalProperties: boolean;
        };
    };
};

class LlmClient {
    private _openaiModel: string;
    private _ollamaModel: string | undefined;

    constructor() {
        this._openaiModel = getOrThrow("OPENAI_MODEL");
        this._ollamaModel = process.env.OLLAMA_MODEL;
        logger.info(`LLM configured: ${this._ollamaModel ? 'Ollama' : 'OpenAI'} with model ${this._ollamaModel ? this._ollamaModel : this._openaiModel}`);
    }

    async getResponse(messages: ChatCompletionMessageParam[], tools: Tool[] = []) {
        if (this._ollamaModel) {
            const ollamaMessages = messages.map(msg => ({
                role: msg.role,
                content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
            }));
    
            try {
                const response = await ollama.chat({
                    model: this._ollamaModel,
                    messages: ollamaMessages,
                    stream: false,
                    // Include any other parameters Ollama might need
                    // options: {
                    //     temperature: 0.7
                    // }
                });
                
                logger.debug("Llm response: " + JSON.stringify(response.message));
                return {
                    role: response.message.role,
                    content: response.message.content,
                    tool_calls: [],
                };
            } catch (error) {
                console.error("Detailed Ollama error:", error);
                logger.error("Error in Ollama request:", error);
                throw error;
            }
        } else {
            // OpenAI client implementation remains unchanged
            const client = new OpenAI({ apiKey: getOrThrow("OPENAI_API_KEY") });
            const completion = await client.chat.completions.create({
                model: this._openaiModel,
                messages: messages,
                tools: tools.map((tool) => tool.toOpenAiTool()),
                tool_choice: "auto",
            });
            logger.debug("Llm response: " + JSON.stringify(completion.choices[0]?.message));
            return completion.choices[0]?.message;
        }
    }
}
export const llmClient = new LlmClient();