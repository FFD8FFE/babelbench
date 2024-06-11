from ..prompt import PromptTemplate, OBSERVATION_KEY, THOUGHT_KEY, FINAL_ANSWER_KEY, DEFAULT_OBSERVATION, \
    DEFAULT_THOUGHT, DEFAULT_FINAL_ANSWER


class ZeroShotReactPrompt(PromptTemplate):
    _input_variables = ["instruction", "agent_scratchpad", "tool_names", "tool_description"]
    # _template = (
    #     "Answer the following questions as best as you can. "
    #     "You have access to the following tools:\n"
    #     "{tool_description}\n"
    #     "Use the following format:\n\n"
    #     "Question: the input question you must answer.\n\n"
    #     "Thought: you should always think about what to do.\n\n"
    #     "Action: the action to take, should be one of [{tool_names}].\n\n"
    #     "Action Input:\n```python\n[the input to the action]\n```.\n\n"
    #     "Observation: the result of the action.\n\n"
    #     "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
    #     "Thought: I now know the final answer\n"
    #     "Final Answer: the final answer to the original input question\n"
    #     "-----------------------------------------------------------------------\nNow Begin!\n\n"
    #     "Question: {instruction}\nThought:"
    #     "{agent_scratchpad}"
    # )
    _template = (
        "Answer the following question as best as you can.\n"
        "<---------------------------------Format Requirements-------------------------------------->\n"
        "Question: the input question you must answer. If an image is input, remember to answer the question with the visual information in the image. (User Provided)\n\n"
        "File Info: the attached files or images related with the question. (User Provided)\n\n"
        "Thought: you should always think about what to do. (LLM Generated)\n\n"
        "Action: you can use python_code_sandbox to execute code and process files. Remember to use the `print()' function to inspect the information of interests. Follow the format of 'Action: python_code_sandbox'. (LLM Generated)\n\n"
        "Action Input: the input of the python_code_sandbox with the format of \nAction Input: ```python\n[the code you want to execute in the sandbox]\n```. (LLM Generated)\n\n"
        "Observation: the execution result of the action. (User provided. You should not generate it.)\n\n"
        "... (this Thought/Action/Action Input/Observation can repeat N times)\n\n"
        "Thought: I now know the final answer.\nFinal Answer: the final answer to the original input question with the format of '```json[your final answer]```'. (LLM Generated)\n"
        "<---------------------------------Other Explanations-------------------------------------->\n"
        "- You are encouraged to first view the contents of the csv file by performing operations such as df.head(), df.info(), print(df[col].unique()), and then write other code.\n"
        "- I have uploaded the image to you, so you can visually inspect it.\n"
        "- For the chart/sheet/table, you cannot process it directly. But you can use the tool of python sandbox to inspect or process it (the file path can be found in 'File Info'). I will execute it and tell you the execution results of the sandbox through 'Observation'.\n"
        "- If you need to process the image, also use the tool of python sandbox. I will execute the code for you.\n"
        "- You should NOT generate observation. I will provide it.\n"
        "- If you do not know the final answer, do not generate the string of 'Thought: I now know the final answer.\nFinal Answer:', espcially when you have actions to take.\n"
        "<---------------------------------Begin the QA-------------------------------------->\n"
        "{instruction}\nThought:"
        "{agent_scratchpad}"
    )
    _keywords = {
        OBSERVATION_KEY: DEFAULT_OBSERVATION,
        THOUGHT_KEY: DEFAULT_THOUGHT,
        FINAL_ANSWER_KEY: DEFAULT_FINAL_ANSWER
    }
    _name = 'ZeroShotReactPrompt'
    _validate_template = True
    _skip_on_failure = True

    def __init__(self, **data):
        super().__init__(**data)
