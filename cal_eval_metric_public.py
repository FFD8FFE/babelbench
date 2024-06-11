# encoding = utf-8
import pandas as pd
import json
import re
from collections import defaultdict
import os
import numpy as np

N_total = 247

def clean_json_str(json_str):
    if json_str is None:
        return None
    try:
        # 替换元组为列表
        json_str = re.sub(r'\(([^()A-Z*/+-]+)\)', r'[\1]', json_str)

        # 确保 JSON 字符串中的 key 和 value 使用双引号
        json_str = re.sub(r'(?<!\\)"', '\\"', json_str)  # 先转义所有双引号
        json_str = re.sub(r'(?<!")"(?!")', '"', json_str)  # 还原之前转义过的正确的双引号
        json_str = json_str.replace('\\"', '"')  # 去掉多余的转义符

        # 处理bool值
        json_str = json_str.replace("True", "true").replace("False", "false")

        # 为缺少引号的值添加引号
        json_str = re.sub(r'(":\\s*)([a-zA-Z]+)(?=\\s*,|\\s*})', r'\\1"\\2"', json_str)

        # 删除非法的逗号
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        tmp = json.loads(json_str)
    except Exception as e:
        # print(json_str)
        # print(e)
        pass
    return json_str

def extract_and_parse_json(text):
    if text is None or pd.isna(text):
        return {}

    patterns = [
        r'```json\n*([\s\S]*?)```',
        r'(?<=final answer\.)(\n*\ *{[\s\S]+?\})',
        r'```python\n*(\{[\s\S]*?\})\n*```',
        r'```py\n*(\{[\s\S]*?\})\n*```'
    ]

    json_strings = []
    for pattern in patterns:
        json_strings = re.findall(pattern, text, re.DOTALL)
        if json_strings:
            break

    json_objects = []
    for json_str in json_strings:
        try:
            json_str = clean_json_str(json_str).strip()
            json_objects.append(json.loads(json_str))
        except json.JSONDecodeError:
            # print("Parse json str failed!")
            pass
        except Exception as e:
            print(e)
    return json_objects


class mmAgentBenchEval:
    def __init__(self):
        pass

    def match_str(self, pred, gt, match_method):
        if not isinstance(pred, str):
            pred = str(pred)
        pred = pred.strip()
        if match_method == 'exact':
            return gt.strip() == pred
        elif match_method == 'either_ok':
            assert isinstance(gt, list)
            return pred in gt
        elif match_method == 'fuzzy':
            if pred.lower() == gt.lower() or gt in pred:
                return True
            else:
                # raise NotImplementedError('sorry...')
                return False
        elif match_method == 'execution_exp':
            try:
                pred_res = eval(pred)
            except:
                return False
            gt_res = eval(gt)
            return pred_res == gt_res
        else:
            raise NotImplementedError(f"match_method of {match_method} is not supported")
        return False

    def match_int(self, pred, gt, match_method):
        if match_method == 'exact':
            try:
                pred, gt = int(pred), int(gt)
                return pred == gt
            except:
                return False
        else:
            raise NotImplementedError(f"match_method of {match_method} is not supported")
        return pred == gt

    def match_float(self, pred, gt, tolerance):
        def match_num(num, gt):
            return abs(num - gt) <= tolerance or (gt != 0 and abs(num - gt) / abs(gt) <= 0.001)

        if isinstance(pred, (float, int)):
            return match_num(num=pred, gt=gt)
        elif isinstance(pred, str):
            potential_floats = re.findall(r"[-+]?\d*\.\d+|\d+", pred)
            for num_str in potential_floats:
                try:
                    num = float(num_str)
                    if match_num(num, gt):
                        return True
                except ValueError:
                    continue
        return False

    def match_bool(self, pred, gt):
        if isinstance(pred, str):
            if 'true' in pred.lower():
                pred=True
            else:
                pred=False
            return pred == gt
        else:
            return pred == gt

    def match_list(self, pred, gt, match_method, tolerance=1e-4):
        def is_close(query, target, tolerance):
            if isinstance(target, (float, int)):
                return abs(query - target) <= tolerance
            else:
                return query==target

        if not isinstance(pred, (list, np.ndarray)):
            return False
        if isinstance(pred, np.ndarray):
            pred = pred.tolist()
        if isinstance(gt, np.ndarray):
            gt = gt.tolist()

        if match_method in ['exact', 'list_of_list']:
            if len(pred) != len(gt):
                return False
            for pred_x, gt_x in zip(pred, gt):
                if not is_close(pred_x, gt_x, tolerance):
                    return False
            return True

        elif match_method == 'disordered_match':
            if len(pred) != len(gt):
                return False
            pred_sorted = sorted(pred)
            gt_sorted = sorted(gt)
            for pred_x, gt_x in zip(pred_sorted, gt_sorted):
                if not is_close(pred_x, gt_x, tolerance):
                    return False
            return True
        else:
            raise NotImplementedError(f"match_method of {match_method} is not supported")
        return False

    def eval_single(self, pred_str, gt_info):
        if gt_info['answer_type'] == 'str':
            res = self.match_str(pred=pred_str, gt=gt_info["gt_answer"], match_method=gt_info.get('eval_method','exact'))
        elif gt_info['answer_type'] == 'float':
            tolerance = gt_info.get('tolerance', 0.00001)
            res = self.match_float(pred=pred_str, gt=gt_info["gt_answer"], tolerance=tolerance)
        elif gt_info['answer_type'] == 'bool':
            res = self.match_bool(pred=pred_str, gt=gt_info["gt_answer"])
        elif gt_info['answer_type'] == 'int':
            res = self.match_int(pred=pred_str, gt=gt_info['gt_answer'], match_method=gt_info.get('eval_method','exact'))
        elif gt_info['answer_type'] in ['list','list_of_int', 'list_of_str','list_of_list','np.ndarray']:
            if gt_info['answer_type'] in ['list_of_list','np.ndarray']:
                match_method = gt_info.get('eval_method', 'list_of_list')
            else:
                match_method = gt_info.get('eval_method', 'exact')
            res = self.match_list(pred=pred_str, gt=gt_info['gt_answer'], match_method=match_method)
        else:
            raise NotImplementedError(f"type of {gt_info['answer_type']} is not supported")
        return res


def get_ques2eval():
    raw_data = pd.read_csv("data/benchmark_tmp.csv")
    ques2eval,ques_info = {},[]
    for index, item in raw_data.iterrows():
        tmp_eval_info = json.loads(item['eval_info'])
        ques_uuid = f"{item['prompt']}_{item['imgs']}"
        ques2eval[ques_uuid] = tmp_eval_info
        item['ques'] = ques_uuid
        ques_info.append(item)
    return ques2eval,ques_info


if __name__=='__main__':
    ques2eval,ques_info = get_ques2eval()

    in_path = "data/output/results_chatgpt.jsonl" # done
    # in_path = "data/output/results_gpt4v.jsonl" # done
    # in_path = "data/output/results_llava.jsonl"
    ques2ans = defaultdict(list)
    with open(in_path, "r") as fr:
        for line in fr:
            try:
                line = json.loads(line.strip())
            except:
                continue
            ques2ans[f"{line['prompt']}_{line['imgs']}"].append(line)
    print(f"Have {len(ques2ans)} questions in the model prediction...\n")

    judge = mmAgentBenchEval()

    ques2evalRes = []
    for ques in ques2ans:
        prompt = ques2ans[ques][0]['prompt']
        if ques not in ques2eval:
            continue
        eval_info = ques2eval[ques]
        eval_flag_prompt = False
        for ans in ques2ans[ques]:
            rsp = extract_and_parse_json(ans['response'])
            if pd.isna(ans['response']) or len(rsp)<1 or not isinstance(rsp[-1], dict):
                continue
            rsp = rsp[-1]
            eval_flag_rsp = True
            for var in eval_info:
                if var not in rsp or judge.eval_single(pred_str=rsp[var], gt_info=eval_info[var]) is False:
                    eval_flag_rsp = False
                    break
            if eval_flag_rsp:
                eval_flag_prompt = True
                break
        ques2evalRes.append({
            'ques': ques,
            'eval_res': eval_flag_prompt,
        })

    df = pd.DataFrame(ques2evalRes)
    acc = round(len(df[df['eval_res']==True])/N_total*100, 2)
    print(f"\nAcc is {acc}")

