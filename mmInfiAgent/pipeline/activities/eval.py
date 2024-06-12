import os
import random
import re
import argparse
import asyncio
import logging
import sys
import json
import io
import pandas as pd
import openai
import time


import infiagent
from infiagent.utils import get_logger, upload_files, get_file_name_and_path
from infiagent.services.chat_complete_service import predict


logger = get_logger()



class UploadedFile(io.BytesIO):
    def __init__(self, path):
        with open(path, 'rb') as file:
            data = file.read()

        super().__init__(data)

        self.name = path.split("/")[-1]  # 获取文件名
        if self.name.endswith('.png'):
            self.type = 'image/png'
        else:
            self.type = 'application/octet-stream'  # 或者其他适当的 MIME 类型
        self.size = len(data)

    def __repr__(self):
        return f"MyUploadedFile(name={self.name}, size={self.size}, type={self.type})"

    def __len__(self):
        return self.size


def _get_script_params():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config_path',
                            help='Config path',
                            default="configs/agent_configs/react_agent_dashscope_qwenvl_plus.yaml",
                            # "configs/agent_configs/react_agent_dashscope_qwenvl_max.yaml",
                            # "configs/agent_configs/react_agent_azure_gpt_4V.yaml"
                            # "configs/agent_configs/react_agent_dashscope_qwenvl_plus.yaml"
                            # "configs/agent_configs/react_agent_dashscope_qwenvl_max.yaml"
                            # "configs/agent_configs/react_agent_genai_GeminiProVision.yaml"
                            required=False, type=str)
        parser.add_argument('--open_path_img',
                            type=str)
        parser.add_argument('--output',
                            help='Output path of evaluation results',
                            default="../output/results_qwenvl_plus.jsonl",
                            # "../output/results_gpt4v.jsonl",
                            # "../output/results_qwenvl_max.jsonl"
                            # "../output/results_geminiProVision.jsonl"
                            required=False, type=str)
        args = parser.parse_args()
        return args

    except Exception as e:
        logger.error("Failed to get script input arguments: {}".format(str(e)), exc_info=True)

    return None


def extract_questions_and_concepts(file_path):
    # Read the content of the text file
    with open(file_path, 'r') as file:
        content = file.read()

    # Use regular expressions to extract questions and concepts
    pattern = r'\\Question{(.*?)}\s*\\Concepts{(.*?)}'
    matches = re.findall(pattern, content, re.DOTALL)

    # Build a list of dictionaries containing the questions and concepts
    data = []
    for match in matches:
        question = match[0].strip()
        concepts = [concept.strip() for concept in match[1].split(',')]
        data.append({
            'question': question,
            'concepts': concepts
        })

    return data

def read_dicts_from_file(file_name):
    """
    Read a file with each line containing a JSON string representing a dictionary,
    and return a list of dictionaries.

    :param file_name: Name of the file to read from.
    :return: List of dictionaries.
    """
    dict_list = []
    with open(file_name, 'r') as file:
        for line in file:
            # Convert the JSON string back to a dictionary.
            dictionary = json.loads(line.rstrip('\n'))
            dict_list.append(dictionary)
    return dict_list

def read_questions(file_path):
    print(file_path)
    with open(file_path) as f:
        questions = json.load(f)

    return questions

def extract_data_from_folder(folder_path):

    print(f'folder_path {folder_path}')
    extracted_data = {}
    # Traverse the files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.questions'):  # You can filter files based on their type
            file_path = os.path.join(folder_path, file_name)
            file_data = read_questions(file_path)
            file_name_without_extension = os.path.splitext(file_name)[0]
            extracted_data[file_name_without_extension] = file_data

    return extracted_data


async def main():

    args = _get_script_params()

    root_directory = os.path.abspath(__file__)
    while 'infiagent' not in os.path.basename(root_directory).lower():
        root_directory = os.path.dirname(root_directory)

    data_path = os.path.join(os.path.dirname(root_directory), "data/benchmark_tmp.csv")
    table_dir = os.path.join(os.path.dirname(root_directory), "data/000-csvs")
    img_dir = os.path.join(os.path.dirname(root_directory), "data/000-imgs")
    extracted_data = pd.read_csv(data_path)
    extracted_data = extracted_data.to_dict(orient="records")
    # random.shuffle(extracted_data)

    prompt2ans = {}
    if os.path.exists(args.output):
        with open(args.output, "r") as fr:
            for line in fr:
                try:
                    line = json.loads(line.strip())
                    prompt2ans[line['prompt']] = line
                except:
                    continue

    start_time = time.time()

    for index, q in enumerate(extracted_data):

        # if "First, as shown in the picture, there is a number on each balloon." not in q['prompt'] and "What is the color of the geometric object which is shiny? Please generate" not in q['prompt'] and "How many people in the table were born in the same year" not in q['prompt'] and "Based on the pricing information provided in the CSV file" not in q['prompt']:
        #     continue
        # if index == 6:
        #     break

        input_text, file_names, img_names = (q['prompt'], q['attachments'], q['imgs'])
        if isinstance(file_names, str):
            file_names = eval(file_names)
        if isinstance(img_names, str):
            img_names = eval(img_names)
        if input_text is None:
            continue
        if input_text in prompt2ans:
            continue

        uploaded_files = []
        for file_name in file_names:
            uploaded_file = UploadedFile(os.path.join(table_dir, file_name))
            uploaded_files.append(uploaded_file)

        uploaded_imgs = []
        for img_name in img_names:
            uploaded_img = UploadedFile(os.path.join(img_dir, img_name))
            uploaded_imgs.append(uploaded_img)

        prompt = f"Question: {input_text}\n"

        response = await predict(
            prompt=prompt,
            uploaded_files=uploaded_files,
            uploaded_imgs=uploaded_imgs,
            config_path=args.config_path,
            open_path_img=args.open_path_img,
        )

        q['response'] = response
        # results.append(q)
        print(f"response: {response}")

        with open(f"{args.output}", 'a') as outfile:
            outfile.write(json.dumps(q)+'\n')

    # 在这里写下你需要进行计时的代码

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"执行时间为：{elapsed_time}秒")

if __name__ == '__main__':
    asyncio.run(main())


