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
    out_file = os.path.join(PROJECT_DIR, "conversation/dan_logs", f"{args.model}.jsonl")

    model = llm_adapter(args.model, device=args.device)
    assist_model = llm_adapter("gpt-4o-mini")

    existing_hashId = jsonl_2_list(out_file, "hashId")
    print(f"[*]Existing conversation number: {len(existing_hashId)}\n")
    conv_dict = dict(zip(existing_hashId, jsonl_2_list(out_file)))
    results = []

    dan_prompts = jsonl_2_list(os.path.join(PROJECT_DIR, "data", "dan_prompts.jsonl"))
    for dan_prompt in tqdm(dan_prompts):
        lang, title, prompt = dan_prompt.values()
        hashId = getMd5(f"{args.model}_{lang}_{title}", 10)
        if hashId in existing_hashId:
            print(f"[*]The conversation has been checked.\n")
            chat_log = conv_dict[hashId]
            results.append(chat_log)
            continue
        else:
            print(f"[*]Check new conversation.\n")
            chat_log = {"target_model": args.model, "lang": lang, "hashId": hashId,
                    "title": title, "broken": "failed", "conversation": []}

            conversation = LLM.user_prompt(prompt)
            model.set_conv(conversation)
            response = model.generate()

            conversation.extend(LLM.assistant_prompt(response))

            if check_violation_response(response, assist_model, verbose=True):
                chat_log["broken"] = "successful"

            chat_log["conversation"] = conversation

            results.append(chat_log)
            save_with_indent(results, out_file)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", type=str, default="qwen3")
    parser.add_argument("-d", "--device", type=str, default="cuda:0")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
