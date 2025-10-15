from __future__ import annotations

from typing import List, Dict, Any, Optional

from litellm import completion


def _format_transcript(messages: List[Dict[str, Any]]) -> str:
    """Render conversation messages into a readable transcript."""
    lines = []
    for message in messages:
        role = message.get("role", "assistant")
        if role == "tool":
            role_display = f"Tool[{message.get('name')}]"
        else:
            role_display = role.capitalize()
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(content)
        lines.append(f"{role_display}: {content}")
    return "\n".join(lines)


class ConversationSummarizer:
    """Wraps GPT-5 summarisation calls for per-question trajectory logging."""

    def __init__(self, model: str, provider: str, effort: Optional[str] = "medium") -> None:
        self.model = model
        self.provider = provider
        self.effort = effort

    def summarize(
        self,
        question_number: int,
        messages: List[Dict[str, Any]],
        latest_user_message: str,
    ) -> str:
        transcript = _format_transcript(messages)
        system_prompt = (
            "You are GPT-5 assisting an agent run. After each customer question you "
            "produce a structured trajectory summary that helps the agent reason about "
            "next steps."
        )
        user_prompt = (
            f"Question number: {question_number}\n"
            f"Latest customer message: {latest_user_message}\n\n"
            "Conversation transcript so far:\n"
            f"{transcript}\n\n"
            "Instructions:\n"
            "- If this is question 1, provide a concise recap of the scenario, key facts, "
            "and open goals discovered so far.\n"
            "- If this is question 2 or later, start your response with the literal text "
            f"'Question {question_number}: {latest_user_message}' on its own line.\n"
            "- After that line, add a 'Trajectory' section describing the reasoning "
            "path you recommend next.\n"
            "- Follow with a 'Relevant Files' section listing any files or data sources "
            "the agent should review, or 'None' if not applicable.\n"
            "- Keep the overall response short (3-5 sentences) while being concrete."
        )
        kwargs: Dict[str, Any] = {}
        if self.effort is not None:
            kwargs["reasoning"] = {"effort": self.effort}
        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )
        return res.choices[0].message.content
