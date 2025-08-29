## Introduction

This repository contains the implementation code for the paper "Self-Deception: Reverse Penetrating the Semantic Firewall of Large Language Models".

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Attack Mode

Run the following command to execute attacks:

```bash
python src/main.py --target-model gpt-3.5-turbo --assist-model qwen2.5 --device cuda:0 --device cuda:1
```

The conversation content is saved in the `./conversation/attack_logs` directory. 

Here is the conversation file:
https://pan.baidu.com/s/1QUA552UkGD3RHTmdQIUtUQ?pwd=gwz8
