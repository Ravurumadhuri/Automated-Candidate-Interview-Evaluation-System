import json
import requests
from typing import Any, Dict, List, Optional, Sequence
from autogen_core.models import (
    CreateResult,
    RequestUsage,
    LLMMessage,
)
from autogen_core import CancellationToken


class GoogleGeminiClient:
    """
    Wrapper for the Google Gemini generateContent API that implements
    the interface expected by autogen's AssistantAgent (same as OpenAIChatCompletionClient).

    Correct endpoint:
        POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
        # autogen checks this dict for vision support when building messages
        self.model_info = {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "unknown",
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _to_gemini_contents(self, messages: Sequence[Any]) -> list:
        """
        Convert autogen LLMMessage objects (or plain dicts) to the
        Gemini `contents` format:
            [{"role": "user"|"model", "parts": [{"text": "..."}]}]

        System messages are prepended as a user turn so Gemini accepts them.
        """
        contents = []
        for msg in messages:
            # get role and content whether it's an object or dict
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                text = msg.get("content", "")
            elif hasattr(msg, "content"):
                role = getattr(msg, "role", "user") if hasattr(msg, "role") else "user"
                text = msg.content if isinstance(msg.content, str) else str(msg.content)
            else:
                role = "user"
                text = str(msg)

            # Gemini only knows "user" and "model"; map everything else
            if role in ("assistant", "model"):
                gemini_role = "model"
            else:
                # system, user, tool, etc. → treat as user turn
                gemini_role = "user"

            # merge consecutive turns of the same role (Gemini requirement)
            if contents and contents[-1]["role"] == gemini_role:
                contents[-1]["parts"][0]["text"] += "\n" + text
            else:
                contents.append({"role": gemini_role, "parts": [{"text": text}]})

        return contents

    # ------------------------------------------------------------------
    # public interface (called by autogen's AssistantAgent)
    # ------------------------------------------------------------------

    async def create(
        self,
        messages: Sequence[Any],
        *,
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs,
    ) -> CreateResult:

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )

        contents = self._to_gemini_contents(messages)

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 1024),
            },
        }

        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60,
        )

        if not resp.ok:
            raise RuntimeError(
                f"Gemini API error {resp.status_code}: {resp.text}"
            )

        data = resp.json()

        # extract generated text
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Gemini response format: {data}") from e

        # build usage stats (Gemini returns token counts in usageMetadata)
        usage_meta = data.get("usageMetadata", {})
        usage = RequestUsage(
            prompt_tokens=usage_meta.get("promptTokenCount", 0),
            completion_tokens=usage_meta.get("candidatesTokenCount", 0),
        )

        return CreateResult(
            content=text,
            usage=usage,
            finish_reason="stop",
            cached=False,
        )

    async def close(self) -> None:
        """No persistent connection to close."""
        pass
