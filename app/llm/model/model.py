import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils import get_num_tokens


class GroqModel:
    """
    Smart async wrapper for Groq's llama-3.1-70b-versatile model.
    - Auto-loads multiple API keys from .env
    - Auto-switches key if token limit < 3000
    - Async-ready for FastAPI / LangGraph pipelines
    """

    def __init__(self):
        load_dotenv()

        # Load multiple API keys, comma-separated
        raw_keys = os.getenv("GROQ_API_KEYS")
        if not raw_keys:
            raise ValueError("GROQ_API_KEYS not found in environment variables.")

        self.api_keys = [k.strip() for k in raw_keys.split(",")]
        self.current_index = 0
        self.model_name = "llama-3.1-70b-versatile"
        self.context_limit = 8192
        self.client = self._init_client(self.api_keys[self.current_index])

        # Maintain usage tracking per key
        self.usage_tracker = {key: 0 for key in self.api_keys}

    def _init_client(self, key: str):
        """Initialize ChatGroq client for a given key."""
        return ChatGroq(model=self.model_name, groq_api_key=key)

    def _switch_key(self):
        """Rotate to next available key."""
        old_key = self.api_keys[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        new_key = self.api_keys[self.current_index]
        self.client = self._init_client(new_key)
        print(f"⚙️ Switched from key {old_key[:8]}... to {new_key[:8]}...")

    def get_token_count(self, messages: list) -> int:
        """Estimate token usage for a message list."""
        return self.client.get_num_tokens_from_messages(messages)

    def get_remaining_tokens(self, messages: list) -> int:
        """Estimate remaining context tokens."""
        used = self.get_token_count(messages)
        return self.context_limit - used

    async def _check_and_switch_key(self):
        """
        Check if current key's token usage is nearing limit.
        If less than 3000 tokens left, auto-switch key.
        """
        current_key = self.api_keys[self.current_index]
        used_tokens = self.usage_tracker[current_key]

        # assume soft quota ~1M tokens per key (customize per your plan)
        remaining_quota = 1_000_000 - used_tokens

        if remaining_quota < 3000:
            print(f"🚨 Token quota low for key {current_key[:8]}... (remaining {remaining_quota})")
            self._switch_key()

    async def chat(self, prompt: str, system_prompt: str = None) -> str:
        """Async chat with auto key-switch and token tracking."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        used = self.get_token_count(messages)
        remaining = self.get_remaining_tokens(messages)
        print(f"[Token Info] Used: {used}, Remaining context: {remaining}")

        # Check and rotate key if needed
        await self._check_and_switch_key()

        # Perform async invoke
        response = await self.client.ainvoke(messages)

        # Update usage
        current_key = self.api_keys[self.current_index]
        self.usage_tracker[current_key] += used + len(response.content.split())

        return response.content
