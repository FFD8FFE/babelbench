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

import openai
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
async def async_chatcompletion_with_backoff(**kwargs):
    async def _internal_coroutine():
        return dashscope.MultiModalConversation.call(**kwargs)
    return await _internal_coroutine()




class qwenvlGPTClient(BaseLLM, ABC):
    """
    https://help.aliyun.com/zh/dashscope/developer-reference/vl-plus-quick-start
    """

    def __init__(self, **data):
        super().__init__(**data)
        dashscope.api_key = data.get("api_key", None)

    @classmethod
    async def create(cls, config_data):
        return qwenvlGPTClient(**config_data)

    def get_model_name(self) -> str:
        return self.model_name

    async def async_completion(self, prompt: str, input_imgs: Optional[List[str]] = None, **kwargs) -> BaseCompletion:
        message_content = [
            {
                "text": prompt[-(self.params.get('max_tokens', 4096) - MAX_GEN_LENGTH):]
            }
        ]
        if input_imgs:
            for img in input_imgs:
                message_content.append({
                    "image": img.open_path
                })

        task = asyncio.create_task(async_chatcompletion_with_backoff(
            model=self.get_model_name(),
            messages=[{
                "role": "user",
                "content": message_content
            }],
            top_p=self.params.get('top_p', 0),
            temperature=self.params.get('temperature', 0.7),
            stream=False,
            **kwargs
        )
        )
        response = await task

        if response.get('status_code') == HTTPStatus.OK:
            rsp_status = "succeed"
        else:
            rsp_status = "fail"
        return BaseCompletion(state=rsp_status,
                              content=response['output']['choices'][0]['message']['content'][0]['text'],
                              prompt_token=response.get("usage", {}).get("prompt_tokens", 0),
                              completion_token=response.get("usage", {}).get("completion_tokens", 0))

