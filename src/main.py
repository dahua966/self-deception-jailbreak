from tqdm import trange

from utils import *
from prompts import *


def main(args):
    questions = jsonl_2_list(data_path("jailbreakbench.jsonl"))
    lang_list = ["zh", "en", "fr", "es"]

    out_file = os.path.join(PROJECT_DIR, "conversation/attack_logs", f"{args.target_model}.jsonl")

    target_model = llm_adapter(args.target_model, device=args.device)
    assist_model = llm_adapter(args.assist_model, device=args.device)

    judge_model = HarmBench(device=args.device2)

    existing_hashId = jsonl_2_list(out_file, "hashId")
    print(f"[*]Existing conversation number: {len(existing_hashId)}\n")
    conv_dict = dict(zip(existing_hashId, jsonl_2_list(out_file)))
    results = []
    
    for lang in lang_list:
        for i in trange(len(questions), desc="Fuzzing Questions"):
            goal = questions[i]["question"]
            category = questions[i]["category"]
            for scenario in Senarios[lang]:
                hashId = getMd5(f"{goal}_{scenario}_{lang}", 10)
    
                if hashId in existing_hashId:
                    print(f"[*]The conversation has been checked.\n")
                    chat_log = conv_dict[hashId]
                    results.append(chat_log)
                    save_with_indent(results, out_file)
                    continue
                else:
                    print(f"[*]Check new conversation.\n")
                    chat_log = {"target_model": args.target_model, "assist_model": args.assist_model,
                                "lang": Languages[lang], "hashId": hashId,
                                "goal": goal, "category": category, "scenario": scenario,
                                "broken": False, "open_questions": "", "conversation": []}
                print(f"[*]Goal:\t{goal}\n[*]Scenario:\t{scenario}\n")
    
                open_questions = json_request(assist_model, open_question_prompt[lang].format(query=goal, scenario=scenario))
                print(f"[*]Open questions:\n{open_questions}\n")
                chat_log["open_questions"] = open_questions
    
                for i, attack_prompt in enumerate(open_questions.values()):
                    print(f"[*]Round {i + 1}\n")
                    print(f"[*]Attacker:\n{attack_prompt}\n")
                    target_model.add_conv(LLM.user_prompt(attack_prompt))
    
                    resp = target_model.generate()
                    print(f"[*]Assistant:\n{resp}\n{'=' * 99}\n")
                    target_model.add_conv(LLM.assistant_prompt(resp))
    
                chat_log["conversation"] = target_model.conversation
                target_model.clear_conv()
    
                qa_pair = [(goal, chat_log["conversation"][-1]["content"])]
                judge_res = judge_model.classify_behavior_generation(qa_pair, True)[0]
                if judge_res.lower() == "yes":
                    print_freedom()
                    chat_log["broken"] = True
    
                results.append(chat_log)
                save_with_indent(results, out_file)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--target-model", type=str, default="gpt-4-turbo")
    parser.add_argument("-a", "--assist-model", type=str, default="qwen2.5")
    # parser.add_argument("--lang", type=str, default="zh")

    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-d2", "--device2", type=str, default="cuda:1")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    # python main.py -m gpt-4-turbo -d2 cuda:2 -d cuda:3
    main(args)
