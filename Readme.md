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
python src/main.py --model gpt-3.5-turbo -device cuda:0
```

The conversation content is saved in the `./conversation/attack_logs` directory. 

### Defense Mode

Run the following command to execute defense:

```bash
python src/defense.py
```

The conversation content is saved in the `./conversation/defense_logs` directory. 

