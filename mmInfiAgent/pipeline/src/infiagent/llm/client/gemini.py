import json
import logging
import os
from abc import ABC
from typing import Callable, List

import google.generativeai as genai
from tenacity import (  # for exponential backoff
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from ..base_llm import BaseLLM
from ...schemas import *

logger = logging.getLogger(__name__)

MAX_GEN_LENGTH = 2048

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(100), reraise=True)
def completion_with_backoff(**kwargs):
    return genai.GenerativeModel('gemini-pro').generate_content(**kwargs)


class GeminiClient(BaseLLM, ABC):
    """
    Wrapper class for OpenAI GPT API collections.
    """

    def __init__(self, **data):
        super().__init__(**data)
        genai.configure(api_key=data.get("api_key", None))

    @classmethod
    async def create(cls, config_data):
        return GeminiClient(**config_data)

    def get_model_name(self) -> str:
        return self.model_name

    async def async_completion(self, prompt: str, **kwargs) -> BaseCompletion:
        """
        Completion method for OpenAI GPT API.

        :param prompt: The prompt to use for completion.
        :type prompt: str
        :param kwargs: Additional keyword arguments.
        :type kwargs: dict
        :return: BaseCompletion object.
        :rtype: BaseCompletion

        """
        messages = [{'role': 'user',
                     'parts': [prompt[-self.params.get('max_tokens', 30720):]]}]

        response = completion_with_backoff(contents=messages,
                                           generation_config=genai.types.GenerationConfig(
                                               temperature=self.params.get('temperature', 0.7)))
        return BaseCompletion(state="success", content=response.text)

