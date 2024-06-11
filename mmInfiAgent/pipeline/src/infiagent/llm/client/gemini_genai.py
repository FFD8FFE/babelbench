import http.client
import typing
import urllib.request
import json
import logging
import os
from abc import ABC
from typing import Callable, List
import requests
from concurrent.futures import ThreadPoolExecutor
import json
from http import HTTPStatus
import dashscope
import asyncio
import google.generativeai as genai
import PIL.Image

from tenacity import (  # for exponential backoff
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from ..base_llm import BaseLLM
from ...schemas import *

logger = logging.getLogger(__name__)

MAX_GEN_LENGTH = 4096

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True,
       before_sleep=before_sleep_log(logger, logging.WARNING))
async def async_chatcompletion_with_backoff(model, inputs, generation_config):
    async def _internal_coroutine():
        return model.generate_content(inputs, generation_config=generation_config)
    return await _internal_coroutine()


class geminiGenAIClient(BaseLLM, ABC):
    def __init__(self, **data):
        super().__init__(**data)
        genai.configure(api_key=data.get("api_key", None))
        self.model = genai.GenerativeModel(self.get_model_name())

    @classmethod
    async def create(cls, config_data):
        return geminiGenAIClient(**config_data)

    def get_model_name(self) -> str:
        return self.model_name

    async def async_completion(self, prompt: str, input_imgs: Optional[List[str]] = None, **kwargs) -> BaseCompletion:
        message_content = [
            prompt[-(self.params.get('max_tokens', 4096) - MAX_GEN_LENGTH):]
        ]

        root_directory = os.path.abspath(__file__)
        while 'infiagent' not in os.path.basename(root_directory):
            root_directory = os.path.dirname(root_directory)
        if input_imgs:
            for img in input_imgs:
                img_path = os.path.join(root_directory,'tmp',img.sandbox_path)
                img_content = PIL.Image.open(img_path)
                message_content.append(img_content)

        generation_config = {
            'top_p': self.params.get('top_p', 0.2),
            'temperature': self.params.get('temperature', 0.7),
        }

        task = asyncio.create_task(async_chatcompletion_with_backoff(
            model=self.model,
            inputs=message_content,
            generation_config=generation_config
        )
        )
        response = await task

        try:
            rsp_state = "succeed"
            rsp_content = response.text
            prompt_token = 0
            completion_token = 0
        except:
            rsp_state = "error"
            rsp_content = "The LLM fails to gen the answer."
            prompt_token = 0
            completion_token = 0
        return BaseCompletion(state=rsp_state,
                              content=rsp_content,
                              prompt_token=prompt_token,
                              completion_token=completion_token)


