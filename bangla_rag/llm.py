import json
import re
import urllib.error
import urllib.request
from typing import Any


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise LLMError("The local model did not return valid JSON") from exc
    if not isinstance(value, dict):
        raise LLMError("The local model returned an unexpected JSON value")
    return value


class LocalLLM:
    def __init__(self, base_url: str, model: str, timeout: float = 180.0) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout

    def complete_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 640,
    ) -> dict[str, Any]:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return _extract_json(payload["choices"][0]["message"]["content"])
        except (OSError, urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMError(f"Could not use the local model at {self.url}: {exc}") from exc

    def health(self) -> tuple[bool, str]:
        url = self.url.rsplit("/chat/completions", 1)[0] + "/models"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return response.status == 200, url
        except OSError as exc:
            return False, str(exc)
