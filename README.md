# You Only Anonymize What Is Not Intent-Relevant: Suppressing Non-Intent Privacy Evidence

<p align="center">
  📄 <a href="https://arxiv.org/abs/2601.04265">arXiv</a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  🛡️ Intent-Aware Privacy Protection
  &nbsp;&nbsp;|&nbsp;&nbsp;
  🤖 Large Language Models
</p>

---

## 🔍 Overview

**IntentAnony** is a pragmatic *intent-conditioned text anonymization framework* built on large language models (LLMs).  
It protects user privacy under **inference-based threat models** while preserving **communicative intent and textual utility**.
Unlike surface-level masking or generic rewriting, IntentAnony **selectively suppresses non-intent privacy evidence**, ensuring that only information irrelevant to the user’s communicative intent is anonymized.

---

## ✨ Key Features

- 🎯 **Intent-aware anonymization** rather than blanket masking  
- 🛡️ Defense against **attribute inference and profiling attacks**  
- 📊 Integrated **privacy–utility evaluation** (automatic + human)  
- 🔄 Supports multiple anonymization strategies and threat settings

---

## 📁 Project Structure

```
IntentAnony_Updated/
├── anonymized/                 # Core anonymization module
│   ├── anonymizers/            # Anonymizer implementations
│   ├── run_workflow.py         # End-to-end anonymization workflow
│   └── eval_workflow.py        # Privacy & utility evaluation workflow
├── configs/                    # Configuration definitions
│   └── config.py               # Configuration class
├── privacy_configs/            # Example privacy task configurations
├── prompt_kits/                # Prompt management and policies
│   ├── prompts/                # Prompt templates
│   └── policy_manager.py       # Policy manager
├── llm_tools/                  # LLM provider wrappers
│   ├── openai_tool.py
│   └── async_openai_tool.py
├── pu_eval/                    # Privacy & utility evaluation
│   ├── eval_privacy.py
│   ├── eval_utility.py
│   └── async_eval_utility.py
├── infer_attack/               # Inference attack implementations
├── utils/                      # Shared utility functions
├── dataset/                    # Datasets
├── main.py                     # Main entry point
└── requirements.txt            # Dependencies
```

---

## 🖥️ System Requirements

* **Python** ≥ 3.8
* **MongoDB** (optional, for dataset storage and experiment logging)
* Sufficient API quotas for supported LLM providers
  (OpenAI, DeepSeek, Google, GLM, etc.)

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone <repository-url>
cd IntentAnony
```

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Download NLTK resources (for BLEU computation)

```python
import nltk
nltk.download('punkt')
```

---

## 🔑 API Key Configuration

Create or edit `llm_tools/keys.json`:

```json
{
  "openai": "sk-your-openai-api-key",
  "deepseek": "sk-your-deepseek-api-key",
  "google": "your-gemini-api-key",
  "glm": "sk-your-glm-api-key"
}
```

> **Note**
>
> * Only include providers you intend to use
> * Keys are automatically loaded when initializing LLM tools

### MongoDB (Optional)

```bash
export MONGODB_HOST="localhost"
export MONGODB_PORT="27017"
```

---

## 🚀 Quick Start

Run a complete anonymization → inference → evaluation pipeline:

```bash
python main.py \
  --config_path ./privacy_configs/personal_reddit/synthetic_glm_ad_piec.yaml \
  --new
```

---

## 🧩 Configuration

Configuration files are written in **YAML** and include:

```yaml
output_dir: "results"
seed: 10
task: "ANONYMIZED"
dataset_name: "personal_reddit"
collection_name: "personal_reddit"

task_config:
  profile_path: "dataset/..."
  outpath: "results/..."
  anonymizer:
    anon_type: "llm"
    target_mode: "single"
  anon_model:
    name: "gemini-3-pro-preview"
    provider: "google"
    prompt_policy_version: "7.0"
  inference_model:
    name: "deepseek-reasoner"
    provider: "deepseek"
  utility_model:
    name: "deepseek-chat"
    provider: "deepseek"
```

---

## 🧠 Main Modules

### 1️⃣ Anonymization (`anonymized/`)

* **IntentAnonymizer** – intent-aware selective anonymization
* **PIECAnonymizer** – privacy inference evidence chain suppression
* **AzureAnonymizer** – Azure text analytics anonymizer

### 2️⃣ Evaluation (`pu_eval/`)

* **Privacy Evaluation** – inference success & protection rate
* **Utility Evaluation** – BLEU, ROUGE, LLM Judge
* **Attack Evaluation** – adversarial inference success rate

### 3️⃣ LLM Tools (`llm_tools/`)

Supported providers include:

* OpenAI (GPT series)
* DeepSeek
* Google (Gemini)
* GLM
* Claude
* Custom providers

### 4️⃣ Prompt Management (`prompt_kits/`)

* Structured prompt organization
* Multi-language support
* Prompt versioning and policy control

---

## 📊 Evaluation Metrics

### Privacy Metrics

* **Inference Accuracy**
* **Privacy Protection Rate**

### Utility Metrics

* **BLEU**
* **ROUGE-1 / ROUGE-L / ROUGE-Lsum**
* **LLM Judge** (readability, semantic preservation, hallucination)

---

## 🧪 Usage Examples

### Example 1: End-to-End Anonymization

```python
from anonymized.run_workflow import run_anon_infer_eval
from utils.initialization import read_config_from_yaml
import asyncio

cfg = read_config_from_yaml("configs/my_config.yaml")
asyncio.run(run_anon_infer_eval(cfg, {}))
```

### Example 2: Batch Utility Evaluation

```python
from anonymized.run import batch_evaluate_utility
from llm_tools.async_openai_tool import create_async_any_tool
from prompt_kits.prompt_manager_final import get_manager
from utils.mongo_utils import MongoDBConnector
import asyncio

prompt_manager = get_manager(default_category="eval_utility")
llm_model = create_async_any_tool(model="gpt-5", provider="openai")

mongo = MongoDBConnector()
mongo.connect()

profiles = mongo.read_data("personal_reddit", query={...})
stats = asyncio.run(batch_evaluate_utility(
    profiles=profiles,
    prompt_manager=prompt_manager,
    llm_model=llm_model,
    mongo=mongo
))
```

---

## ⚠️ Notes

1. Ensure all required API keys are correctly set
2. MongoDB must be running if enabled
3. Input data should follow the expected JSONL format

---

## 🙏 Acknowledgements

We thank the authors of  
[LLM-Anonymization](https://github.com/eth-sri/llm-anonymization)  
for releasing their code and inspiring this work.

---

## 📌 Citation

If you use this code, please consider citing our work:

```bibtex
@article{intentanony2026,
  title   = {You Only Anonymize What Is Not Intent-Relevant: Suppressing Non-Intent Privacy Evidence},
  author  = {Shen, Weihao and Xu, Yaxin and Li, Shuang and Chen, Wei and Lan, Yuqin and Yuan, Meng and Zhuang, Fuzhen},
  journal = {arXiv preprint arXiv:2601.04265},
  year    = {2026}
}
```







