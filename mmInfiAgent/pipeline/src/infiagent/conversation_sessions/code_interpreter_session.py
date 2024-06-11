import logging
import os
import time
from typing import Any, Dict, Union
import pandas as pd
from io import StringIO

from werkzeug.datastructures import FileStorage

from ..agent import BaseAgent
from ..agent.react import AsyncReactAgent
from ..schemas import AgentRequest, MediaFile, Message, RoleType
from ..utils import generate_random_string, get_logger, get_model_config_path

logger = get_logger()


class CodeInterpreterSession:
    def __init__(
            self,
            session_id: Union[None, str] = None,
            model_name: Union[None, str] = "openai",
            config_path: Union[None, str] = None,
            agent: AsyncReactAgent = None,
            **kwargs):
        self.session_id = session_id
        self.config_path = config_path
        self.input_files = []
        self.output_files = []
        self.messages = []
        self.agent = agent
        self.llm_model_name = self.agent.llm.model_name

        logger.info("Use model {} and llm in config {} for conversation {}"
                    .format(model_name, self.llm_model_name, self.config_path, self.session_id))

    @classmethod
    async def create(cls,
                     config_path: Union[None, str] = None,
                     **kwargs: Dict[str, Any]):

        logger.info(f"Use Config Path: {config_path}")
        sandbox_id = generate_random_string(8)

        # setup agent
        agent = await BaseAgent.async_from_config_path_and_kwargs(config_path, **kwargs)
        await agent.plugins_map["python_code_sandbox"].set_sandbox_id(sandbox_id)

        return cls(session_id=sandbox_id,
                   config_path=config_path,
                   agent=agent)

    async def upload_to_sandbox(self, file: Union[str, FileStorage], open_path_file: str = None):
        dst_path = await self.agent.sync_to_sandbox(file)
        message = f'User uploaded the following files: {dst_path}\n'
        logging.info(f"The file path {file} has been synced to sandbox with file path {dst_path}")
        self.messages.append(Message(RoleType.System, message))

        file_base_name = os.path.basename(dst_path)
        if ".png" in file_base_name:
            file_type = "img"
        elif ".csv" in file_base_name:
            file_type = "csv"
        else:
            file_type = "file"
        if file_type == "img":
            open_path = open_path_file + file_base_name
            file_basic_info = None
        elif file_type == 'csv':
            open_path = None
            df = pd.read_csv(file)
            output = StringIO()
            df.info(buf=output)
            info_str = output.getvalue()
            info_lines = info_str.splitlines()
            info_str_top20 = "\n".join(info_lines[2:20])
            file_basic_info = info_str_top20
        else:
            open_path = None
            file_basic_info = None
        self.input_files.append(MediaFile(
                file_name=file_base_name,
                sandbox_path=dst_path,
                file_type=file_type,
                open_path=open_path,
                file_basic_info=file_basic_info,
        ))

    async def chat(self, user_messages, input_files=None):
        start_time = time.time()

        self.messages.extend(user_messages)
        agent_request = AgentRequest(
            messages=self.messages,
            input_files=self.input_files,
            sandbox_id=self.session_id
        )
        logger.info(f"Agent request: {agent_request.__dict__}\n[Request End]")

        async for agent_response in self.agent.async_run(agent_request):
            logger.info(f"Agent response:\n{agent_response.output_text}\n[Response End]")
            self.messages.append(Message(RoleType.System, agent_response.output_text))
            yield agent_response

        exec_time = time.time()
        logger.info(
            f'Agent Execution Latency: {exec_time - start_time}'
        )

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass
