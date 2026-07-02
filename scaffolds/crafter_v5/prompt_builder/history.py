from collections import deque
from typing import List, Optional

class Message:
    def __init__(self, role: str, content: str, attachment: Optional[object] = None):
        self.role = role  
        self.content = content  
        self.attachment = attachment

    def __repr__(self):
        return f"Message(role={self.role}, content={self.content}, attachment={self.attachment})"

class HistoryPromptBuilder:
    def __init__(
        self,
        max_text_history: int = 16,
        max_image_history: int = 1,
        system_prompt: Optional[str] = None,
        max_cot_history: int = 1,
    ):
        self.max_text_history = max_text_history
        self.max_image_history = max_image_history
        self.max_history = max(max_text_history, max_image_history)
        self.system_prompt = system_prompt
        self._events = deque(maxlen=self.max_history * 2)  
        self._last_short_term_obs = None  
        self.previous_reasoning = None
        self.max_cot_history = max_cot_history

    def update_instruction_prompt(self, instruction: str):
        self.system_prompt = instruction

    def update_observation(self, obs: dict):
        long_term_context = obs["text"].get("long_term_context", "")
        self._last_short_term_obs = obs["text"].get("short_term_context", "")
        text = long_term_context

        image = obs.get("image", None)

        self._events.append(
            {
                "type": "observation",
                "text": text,
                "image": image,
            }
        )

    def update_action(self, action: str):
        self._events.append(
            {
                "type": "action",
                "action": action,
                "reasoning": self.previous_reasoning,
            }
        )

    def update_reasoning(self, reasoning: str):
        self.previous_reasoning = reasoning

    def reset(self):
        self._events.clear()

    def get_prompt(self, icl_episodes=False) -> List[Message]:
        messages = []

        if self.system_prompt and not icl_episodes:
            messages.append(Message(role="user", content=self.system_prompt))

        text_needed = self.max_text_history
        for event in reversed(self._events):
            if event["type"] == "observation":
                if text_needed > 0 and event.get("text") is not None:
                    event["include_text"] = True
                    text_needed -= 1
                else:
                    event["include_text"] = False

        images_needed = self.max_image_history
        for event in reversed(self._events):
            if event["type"] == "observation":
                if images_needed > 0 and event.get("image") is not None:
                    event["include_image"] = True
                    images_needed -= 1
                else:
                    event["include_image"] = False

        reasoning_needed = self.max_cot_history
        for event in reversed(self._events):
            if event["type"] == "action":
                if reasoning_needed > 0 and event.get("reasoning") is not None:
                    reasoning_needed -= 1
                else:
                    event["reasoning"] = None

        for idx, event in enumerate(self._events):
            if event["type"] == "observation":
                message_parts = []

                if idx == len(self._events) - 1:
                    message_parts.append("Current Observation:")
                    if self._last_short_term_obs:
                        message_parts.append(self._last_short_term_obs)
                else:
                    message_parts.append("Observation:")

                if event.get("include_text", False):
                    message_parts.append(event["text"])
                    
                image = None
                if event.get("include_image", False):
                    image = event["image"]
                    message_parts.append("Image observation provided.")

                content = "\n".join(message_parts)
                message = Message(role="user", content=content, attachment=image)

                for flag in ["include_text", "include_image"]:
                    if flag in event:
                        del event[flag]
            elif event["type"] == "action":
                if event.get("reasoning") is not None:
                    content = "Previous plan:\n" + event["reasoning"]
                else:
                    content = event["action"]
                message = Message(role="assistant", content=content)
            messages.append(message)

        return messages
