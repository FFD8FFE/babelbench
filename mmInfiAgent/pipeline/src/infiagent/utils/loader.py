import importlib
import os
from pathlib import Path
from typing import IO, Any

import yaml


class Loader(yaml.SafeLoader):
    """
        A custom YAML loader that adds support for various custom tags:

        - !include: includes a YAML file as a subdocument
        - !prompt: returns a prompt class based on the specified string
        - !tool: returns a tool class based on the specified string
        - !env: returns the value of an environment variable
        - !file: returns the contents of a file
    """
    def __init__(self, stream: IO[Any]) -> None:
        """
           Initializes a new instance of the Loader class.
           :param stream: The stream to load YAML from.
           :type stream: IOBase
       """
        self._root = Path(stream.name).resolve().parent
        super(Loader, self).__init__(stream)
        self.add_constructor("!include", Loader.include)
        self.add_constructor("!prompt", Loader.prompt)
        self.add_constructor("!tool", Loader.tool)
        self.add_constructor("!env", Loader.env)
        self.add_constructor("!file", Loader.file)

    def include(self, node: yaml.Node) -> Any:
        """
        Loads a YAML file from a path relative to the current file. Use this tag to include other agent configs as plugins.

        :param node: The YAML node to be loaded.
        :type node: yaml.Node
        :return: The loaded YAML file.
        :rtype: Any
        """
        filename = Path(self.construct_scalar(node))
        if not filename.is_absolute():
            filename = self._root / filename
        with open(filename, 'r') as f:
            return yaml.load(f, Loader)

    def prompt(self, node: yaml.Node) -> Any:
        """
            Returns a PromptTemplate class based on the specified string.

            :param node: The YAML node representing the prompt string.
            :type node: yaml.Node
            :return: The prompt class.
            :rtype: type
            :raises AssertionError: If the resolved prompt class is not a subclass of PromptTemplate.
        """
        from ..prompt import PromptTemplate, SimpleReactPrompt, ZeroShotReactPrompt
        prompt = self.construct_scalar(node)
        if '.' in prompt:
            _path = prompt.split('.')
            module = importlib.import_module('.'.join(_path[:-1]))
            prompt_cls = getattr(module, _path[-1])
        else:
            prompt_cls = eval(prompt)
        assert issubclass(prompt_cls.__class__, PromptTemplate)
        return prompt_cls

    def tool(self, node: yaml.Node) -> Any:
        """
        Loads a Custom BaseTool class from a path relative to the current file.

        :param node: The YAML node to be loaded.
        :type node: yaml.Node
        :return: The loaded BaseTool class.
        :rtype: Any
        """
        from ..tools import BaseTool, PythonSandBoxTool

        tool = self.construct_scalar(node)
        if '.' in tool:
            _path = tool.split('.')
            module = importlib.import_module('.'.join(_path[:-1]))
            tool_cls = getattr(module, _path[-1])
        else:
            tool_cls = eval(tool)
        
        assert issubclass(tool_cls, BaseTool)
        return tool_cls

    def env(self, node: yaml.Node) -> Any:
        """
        Loads an environment variable from the current environment, defaults to an empty string if the variable is not set.

        :param node: The YAML node to be loaded.
        :type node: yaml.Node
        :return: The loaded environment variable.
        :rtype: Any
        """
        return os.environ.get(self.construct_scalar(node), "")

    def file(self, node: yaml.Node) -> Any:
        """
        Loads any readable file from a path relative to the current file.

        :param node: The YAML node to be loaded.
        :type node: yaml.Node
        :return: The loaded file.
        :rtype: Any
        """
        filename = Path(self.construct_scalar(node))
        if not filename.is_absolute():
            filename = self._root / filename
        with open(filename, 'r') as f:
            return f.read().strip()