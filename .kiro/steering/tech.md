# Technology Stack

## Core Technologies

- **Python**: Primary programming language for data processing and ML workflows
- **Datasets Library**: HuggingFace datasets for data loading and management
- **JSON/CSV**: Primary data formats for training datasets
- **PDF**: Documentation and research papers

## Data Processing

- **HuggingFace Datasets**: Standard library for dataset loading and preprocessing
- **Pandas**: CSV data manipulation and analysis
- **JSON**: Structured data handling for conversational datasets

## Machine Learning Focus

- **Intent Classification**: Multi-class classification with 77+ banking intents
- **Natural Language Understanding (NLU)**: Text classification and entity recognition
- **Large Language Models (LLMs)**: Fine-tuning for domain-specific chatbots
- **Conversational AI**: Multi-turn dialogue systems

## Data Formats

- **BANKING77**: CSV format with text/label pairs
- **Bitext**: CSV/Parquet with instruction/response pairs
- **Schema-Guided**: JSON format for multi-turn dialogues
- **Twitter Support**: CSV format for customer service interactions

## Common Commands

```bash
# Dataset exploration
python -c "from datasets import load_dataset; ds = load_dataset('path/to/dataset')"

# Data analysis
python -m pandas.read_csv('Dataset/*/data.csv')

# Intent analysis
python Dataset/BANKING77/banking77.py
```

## Development Environment

- Jupyter notebooks for data exploration
- Python scripts for data processing
- Git for version control
- Standard Python data science stack (pandas, numpy, scikit-learn)
