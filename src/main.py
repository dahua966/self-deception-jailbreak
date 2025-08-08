#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import *
from prompts import *


def check_violation_response(response, assist_model, verbose=False):
    res = json_request(assist_model, checkViolation.format(response=response))
    if verbose:
        pprint(res)
    if res["judge"].lower() == "yes":
        return True
    else:
        return False


def main(args):
    out_file = os.path.join(PROJECT_DIR, "conversation/attack_logs", f"{args.model}.jsonl")

    model = llm_adapter(args.model, device=args.device)
    assist_model = llm_adapter("gpt-4o-mini")

    existing_hashId = jsonl_2_list(out_file, "hashId")
    print(f"[*]Existing conversation number: {len(existing_hashId)}\n")
    conv_dict = dict(zip(existing_hashId, jsonl_2_list(out_file)))
    results = []

    for lang in Languages:
        pbar = tqdm(total=len(Senarios[lang])*3*10, desc=f"Breaking {lang}")
        for scenario in Senarios[lang]:
            for evil_category in init_vectors:
                evil_adjs = init_vectors[evil_category][lang]
                for evil_adj in evil_adjs:
                    hashId = getMd5(f"{args.model}_{scenario}_{evil_adj}_{lang}", 10)
                    if hashId in existing_hashId:
                        print(f"[*]The conversation has been checked.\n")
                        chat_log = conv_dict[hashId]
                        results.append(chat_log)
                        pbar.update(1)
                        continue
                    else:
                        print(f"[*]Check new conversation.\n")
                        chat_log = {"target_model": args.model, "lang": Languages[lang], "hashId": hashId,
                                "evil_category": evil_category, "scenario": scenario,
                                "evil_adj": evil_adj, "broken": False,
                                "conversation": []}

                    design_scenario = step1[lang].format(environment=scenario, evil=evil_adj)

                    conversation = LLM.user_prompt(design_scenario)
                    print("[*]Designing the scenario")
                    model.set_conv(conversation)
                    scenario_setting = model.generate()

                    # 获取标题
                    title = extract_keyword(scenario_setting)
                    title = title if title else "story"
                    print(f"[+]Title: {title}")

                    # 记录场景的世界观设定
                    conversation.extend(LLM.assistant_prompt(scenario_setting))

                    print("\n[*]Checking whether chatbot jailbreak in step1")
                    if check_violation_response(scenario_setting, assist_model, verbose=True):
                        chat_log["broken"] = "successful@step1"

                    # 设计反派
                    design_villain = step2[lang].format(evil=evil_adj, title=title)
                    conversation.extend(LLM.user_prompt(design_villain))
                    print("[*]Designing the villain")
                    model.set_conv(conversation)
                    villain_setting = model.generate()

                    # 获取反派姓名
                    name = extract_keyword(villain_setting)
                    name = name if name else "the villain"
                    print(f"[+]Name: {name}")

                    # 记录反派的设定
                    conversation.extend(LLM.assistant_prompt(villain_setting))

                    print("\n[*]Checking whether chatbot jailbreak in step2")
                    if not chat_log["broken"] and check_violation_response(villain_setting, assist_model, verbose=True):
                        chat_log["broken"] = "successful@step2"

                    # 描述反派的具体行为
                    detail_evil = step3[lang].format(evil=evil_adj, name=name)
                    conversation.extend(LLM.user_prompt(detail_evil))
                    print("[*]Describing the villain's behaviour")

                    # 请求生成反派的具体恶意行为
                    model.set_conv(conversation)
                    evil_detail = model.generate()
                    conversation.extend(LLM.assistant_prompt(evil_detail))

                    # 询问反派的恶意行为
                    if not chat_log["broken"]:
                        print("\n[*]Checking whether chatbot refuse or jailbreak in step3")
                        if check_violation_response(evil_detail, assist_model, verbose=True):
                            chat_log["broken"] = "successful@step3"
                        else:
                            chat_log["broken"] = "failed"

                    chat_log["conversation"] = conversation

                    results.append(chat_log)
                    save_with_indent(results, out_file)
                    pbar.update(1)

        pbar.close()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", type=str, default="qwen3")
    parser.add_argument("-d", "--device", type=str, default="cuda:0")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)