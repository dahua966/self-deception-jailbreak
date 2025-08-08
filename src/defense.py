#!/usr/bin/env python
# -*- coding: utf-8 -*-
from utils import *
from prompts import *


def defense_analysis(model_name="qwen3"):
    convs = jsonl_2_list(f"../conversation/attack_logs/{model_name}.jsonl")
    defense_model = llm_adapter("gpt-4o-mini")

    out_file = os.path.join(PROJECT_DIR, "conversation/defense_logs", f"{model_name}.jsonl")
    results = []

    false_neg = 0
    for conv in tqdm(convs):
        if conv.get("broken").startswith("successful"):
            match = re.search(r'step(\d+)', conv["broken"])
            if match:
                step_num = int(match.group(1))
            else:
                step_num = len(conv["conversation"]) // 2 - 1

            history = conv["conversation"][:(step_num-1)*2]
            prompt = conv["conversation"][-2]["content"]

            defense_conv = LLM.sys_prompt(defense_sys_prompt) + \
                    LLM.user_prompt(defense_prompt.format(history=history, prompt=prompt))

            # defense_model.set_conv(defense_conv)
            res = json_request(defense_model, defense_conv)
            if res["label"].lower() == "safe":
                false_neg += 1
                results.append(conv)

    print(f"[*]Model: {model_name}")
    print(f"[*] After Intent-Filter: {false_neg / 1260 * 100:.1f}%")
    save_with_indent(results, out_file)


if __name__ == "__main__":
    for model in ["gpt-3.5-turbo", "gpt-4", "qwen3", "deepseek"]:
        defense_analysis(model)

