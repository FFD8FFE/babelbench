import base64
import json
import logging
import os
from abc import ABC
from typing import Callable, List
from ...utils import get_logger
import httpx

import openai
from tenacity import (  # for exponential backoff
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from PIL import Image
from anthropic import AsyncAnthropic
from ..base_llm import BaseLLM
from ...schemas import *

logger = get_logger()
MAX_GEN_LENGTH = 1500


def compress_image(image_path, output_path, max_size_mb=4.0):
    max_size = max_size_mb * 1024 * 1024
    with Image.open(image_path) as img:
        img_size = os.path.getsize(image_path)
        if img_size > max_size:
            ratio = (max_size / img_size) ** 0.5
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)
            img = img.resize((new_width, new_height), Image.ANTIALIAS)
            img.save(output_path, optimize=True, quality=95)
            return True
        else:
            return False


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5), reraise=True,
       before_sleep=before_sleep_log(logger, logging.WARNING))
async def async_chat_completion_with_backoff(**kwargs):
    client = AsyncAnthropic(api_key=kwargs.get('api_key', None))
    async def _internal_coroutine():
        return await client.messages.create(**kwargs)

    return await _internal_coroutine()


class ClaudeClient(BaseLLM, ABC):
    """
    Wrapper class
    """

    def __init__(self, **data):
        super().__init__(**data)
        self.model_name = data.get("model_name", 'claude3')
        self.api_key = data.get("api_key", None)

    @classmethod
    async def create(cls, config_data):
        return ClaudeClient(**config_data)

    def get_model_name(self) -> str:
        return self.model_name

    async def async_completion(self, prompt: str, input_imgs: Optional[List[str]] = None, **kwargs) -> BaseCompletion:
        """
        Completion method for OpenAI GPT API.

        :param prompt: The prompt to use for completion.
        :type prompt: str
        :param kwargs: Additional keyword arguments.
        :type kwargs: dict
        :return: BaseCompletion object.
        :rtype: BaseCompletion

        """
        message_content = [
            {
                "type": "text", "text": prompt[-(self.params.get('max_tokens', 4096) - MAX_GEN_LENGTH):]
            }
        ]
        if input_imgs:
            for img in input_imgs:
                file_name = f"./src/infiagent/tmp/{img.sandbox_path}"
                compressed_file_name = file_name[:-4] + '_compressed.png'
                if compress_image(file_name, compressed_file_name):
                    file_name = compressed_file_name

                with open(file_name, "rb") as image_file:
                    image_data = image_file.read()

                message_content.append({
                    "type": "image", "source": {"type": "base64",
                                                "media_type": "image/png",
                                                "data": base64.b64encode(image_data).decode("utf-8")}
                })

        logger.info(f"The message send to LLM is: {message_content}")

        response = await async_chat_completion_with_backoff(
            model=self.get_model_name(),
            messages=[
                {"role": "user", "content": message_content}
            ],
            max_tokens=self.params.get('max_tokens', 4096),
            temperature=self.params.get('temperature', 0.7),
            api_key=self.api_key,
            # top_p=self.params.get('top_p', 0.9),
            # frequency_penalty=self.params.get('frequency_penalty', 1.0),
            **kwargs
        )

        return BaseCompletion(state="success",
                              content=response.content[0].text,
                              prompt_token=response.usage.input_tokens,
                              completion_token=response.usage.output_tokens)





