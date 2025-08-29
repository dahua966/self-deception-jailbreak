from pathlib import Path
import os

API_HOST = "https://api.huiyan-ai.com/v1"
API_KEY = "sk-"
DIRECT_KEY = "sk-"

import subprocess
result = subprocess.run(['whoami'], capture_output=True, text=True)
BREAKER = True if result.stdout.strip() == "breaker" else False
WIN = True if "desktop" in result.stdout.strip() else False

PUBLIC_MODEL = ""

MODEL_PATH = {"llama2": f"{PUBLIC_MODEL}/Llama-2-7b-chat-hf",
              "qwen3": f"{PUBLIC_MODEL}/Qwen3-8B",
              "qwen2.5": f"{PUBLIC_MODEL}/Qwen2.5-7B-Instruct",
              "tinyQwen": f"{PUBLIC_MODEL}/Qwen2.5-0.5B-Instruct",
              "tinyQwen3": f"{PUBLIC_MODEL}/Qwen3-0.6B",
              "deepseek": f"{PUBLIC_MODEL}/DeepSeek-R1-Distill-Llama-8B",
              "harmbench": f"{PUBLIC_MODEL}/HarmBench-Llama-2-13b-cls",
              # "llama_guard3": f"{PUBLIC_MODEL}/Meta-Llama-Guard-3-8B",
              "llama_guard3": f"{PUBLIC_MODEL}/Llama-Guard-3-1B",
              "llama_guard2": f"{PUBLIC_MODEL}/Meta-Llama-Guard-2-8B",
              "prompt_guard2": f"{PUBLIC_MODEL}/Llama-Prompt-Guard-2-86M",}

PROJECT_DIR = str(Path(__file__).parent.parent.resolve())
DATA_DIR = f"{PROJECT_DIR}/data"


Languages = {"zh":"Chinese", "en":"English", "ru":"Russian", "fr":"French", "es":"Spanish", "ar":"Arabic"}


# 对话场景
Senarios = {'en': ['game', 'teleplay', 'script', 'novel', 'comic', 'movie'],
            'es': ['juego', 'televisión', 'guión', 'novela', 'cómic', 'película'],
            'zh': ['游戏', '电视剧', '剧本', '小说', '漫画', '电影'],
            'fr': ['jeu', 'télévision', 'scénario', 'roman', 'bande dessinée', 'film'],
            }


def data_path(*file_name):
    return os.path.join(PROJECT_DIR, "data", *file_name)


def conv_path(*file_name):
    return os.path.join(PROJECT_DIR, "conversation", *file_name)


cate_abbr = {
    'Malware/Hacking': 'Malware',
    'Privacy': 'Priv',
    'Fraud/Deception': 'Fraud',
    'Economic harm': 'EconHarm',
    'Government decision-making': 'GovDec',
    'Expert advice': 'ExpAdv',
    'Disinformation': 'Disinfo',
    'Harassment/Discrimination': 'Harrass',
    'Physical harm': 'PhysHarm',
    'Sexual/Adult content': 'Sexual',
}
