# IntentAnony - Intent-Based Anonymization System

An intent-based anonymization system built on large language models (LLMs) that protects user privacy while maintaining text utility. The system supports multiple anonymization strategies, privacy evaluation, and utility assessment.

## Features

- **Intent-Based Anonymization**: Intelligent anonymization processing based on intent recognition
- **Privacy Protection**: Supports multiple anonymization strategies (LLM, Azure, Span, etc.)
- **Privacy Evaluation**: Assesses privacy protection effectiveness after anonymization
- **Utility Evaluation**: Evaluates text utility after anonymization (BLEU, ROUGE, LLM Judge)
- **Intent Recognition**: Automatically identifies privacy intents in user text
- **Attack Assessment**: Evaluates privacy leakage risks from adversarial attacks
- **Async Processing**: Efficient asynchronous batch processing capabilities
- **Multi-Model Support**: Supports multiple LLM providers (OpenAI, DeepSeek, Google, GLM, etc.)

## Project StructureIntentAnony_Updated/

```
├── anonymized/              # Core anonymization module
│   ├── anonymizers/         # Anonymizer implementations
│   ├── run_workflow.py      # Anonymization workflow
│   └── eval_workflow.py   #  Evaluationworkflow
├── configs/                 # Configuration files
│   └── config.py            # Configuration class definitions
├── privacy_configs/         # Privacy configuration examples
├── prompt_kits/             # Prompt management
│   ├── prompts/             # Prompt templates
│   └── policy_manager.py    # Policy manager
├── llm_tools/               # LLM tool wrappers
│   ├── openai_tool.py       # OpenAI tools
│   └── async_openai_tool.py # Async OpenAI tools
├── pu_eval/                 # Privacy and utility evaluation
│   ├── eval_privacy.py      # Privacy evaluation
│   ├── eval_utility.py      # Utility evaluation
│   └── async_eval_utility.py # Async utility evaluation
├── infer_attack/            # Inference attack module
├── utils/                   # Utility functions
├── dataset/                 # Datasets
├── main.py                  # Main entry point
└── requirements.txt         # Dependencies
```

## System Requirements

- Python >= 3.8
- MongoDB (optional, for data storage)
- Sufficient API quotas (OpenAI, DeepSeek, Google, etc.)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd IntentAnony
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (for BLEU calculation)

```python
import nltk
nltk.download('punkt')
```

### 5. Configure API Keys

Create or edit the `llm_tools/keys.json` file to set your API keys:

```json
{
    "openai": "sk-your-openai-api-key",
    "deepseek": "sk-your-deepseek-api-key",
    "google": "your-gemini-api-key",
    "glm": "sk-your-glm-api-key",
}
```

**Note**: Only include the API keys for the providers you plan to use. The system will automatically load these keys from `llm_tools/keys.json` when initializing LLM tools.

For MongoDB connection (optional), you can configure it in your configuration files or set environment variables:

```bash
export MONGODB_HOST="localhost"
export MONGODB_PORT="27017"
```

## Quick Start

### 1. Run anonymization task

```bash
python main.py --config_path .\privacy_configs\personal_reddit\synthetic_glm_ad_piec.yaml --new
```

## Configuration

Configuration files use YAML format and mainly contain the following sections:

```yaml
output_dir: "results"           # Output directory
seed: 10                        # Random seed
task: "ANONYMIZED"              # Task type
dataset_name: "personal_reddit"  # Dataset name
collection_name: "personal_reddit"  # MongoDB collection name

task_config:
  profile_path: "dataset/..."    # Dataset path
  outpath: "results/..."        # Output path
  anonymizer:                    # Anonymizer configuration
    anon_type: "llm"             # Anonymization type
    target_mode: "single"        # Target mode
  anon_model:                    # Anonymization model
    name: "gemini-3-pro-preview"
    provider: "google"
    prompt_policy_version: "7.0"
  inference_model:               # Inference model
    name: "deepseek-reasoner"
    provider: "deepseek"
  utility_model:                 # Utility evaluation model
    name: "deepseek-chat"
    provider: "deepseek"
```

## Main Modules

### 1. Anonymization Module (`anonymized/`)

- **IntentAnonymizer**: Intent-based anonymizer
- **PIECAnonymizer**: Privacy inference evidence chain anonymizer
- **AzureAnonymizer**: Azure text analytics anonymizer

### 2. Evaluation Module (`pu_eval/`)

- **Privacy Evaluation**: Assesses privacy protection effectiveness after anonymization
- **Utility Evaluation**: Calculates BLEU, ROUGE, and LLM Judge scores
- **Attack Evaluation**: Evaluates success rate of adversarial attacks

### 3. LLM Tools (`llm_tools/`)

Supports multiple LLM providers:

- OpenAI (GPT series)
- DeepSeek
- Google (Gemini)
- GLM
- Claude
- Custom providers

### 4. Prompt Management (`prompt_kits/`)

- Structured prompt management
- Multi-language support
- Version control
- Policy management

## Evaluation Metrics

### Privacy Metrics

- **Inference Accuracy**: Accuracy of attackers inferring privacy attributes from anonymized text
- **Privacy Protection Rate**: Proportion of privacy information successfully protected

### Utility Metrics

- **BLEU**: BLEU score for text similarity
- **ROUGE**: ROUGE-1, ROUGE-L, ROUGE-Lsum scores
- **LLM Judge**: LLM-based scores for readability, semantic preservation, and hallucination detection

## Usage Examples

### Example 1: Basic Anonymization

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

prompt_manager = get_manager(default_category='eval_utility')
llm_model = create_async_any_tool(model='gpt-5', provider='openai')
mongo = MongoDBConnector()
mongo.connect()

profiles = mongo.read_data('personal_reddit', query={...})
stats = asyncio.run(batch_evaluate_utility(
    profiles=profiles,
    prompt_manager=prompt_manager,
    llm_model=llm_model,
    mongo=mongo
))
```

## Notes

1. **API Keys**: Ensure all required API keys are properly configured in `llm_tools/keys.json`
2. **MongoDB**: If using MongoDB, ensure the service is running
3. **Data Format**: Ensure input data conforms to the expected format (JSONL)
