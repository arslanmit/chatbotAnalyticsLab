# Project Structure

## Root Directory Organization

```
├── Dataset/                    # All training datasets
├── assignment/                 # Project requirements and documentation
├── README.md                   # Project overview
└── .kiro/                     # Kiro configuration and steering
```

## Dataset Directory Structure

### Core Datasets

- **`Dataset/BANKING77/`**: Fine-grained intent classification
  - `data/`: JSON files (train.json, test.json, val.json)
  - `banking77.py`: HuggingFace dataset loader
  - `README.md`: Dataset documentation

- **`Dataset/BitextRetailBanking/`**: LLM training data
  - `.csv` and `.parquet` files with Q&A pairs
  - 25,545 instruction/response pairs for banking domain

- **`Dataset/SchemaGuidedDialogue/`**: Multi-turn conversations
  - `banking_only/`: Filtered banking dialogues
  - `dstc8-schema-guided-dialogue-master/`: Full dataset
  - Organized by train/dev/test splits

- **`Dataset/CustomerSupportOnTwitter/`**: Real customer interactions
  - `sample.csv`: Sample data for exploration
  - Full dataset requires Kaggle authentication

- **`Dataset/SyntheticTechSupportChats/`**: Generated support data
  - Multiple CSV files with synthetic conversations

## File Naming Conventions

- **Dataset folders**: PascalCase (e.g., `BitextRetailBanking`)
- **Data files**: lowercase with underscores (e.g., `train.json`, `intent_labels.json`)
- **Documentation**: `README.md` in each dataset folder
- **Python loaders**: `{dataset_name}.py` format

## Data Organization Patterns

- Each dataset maintains its own subdirectory structure
- Training/validation/test splits are clearly separated
- Metadata and schema files included where applicable
- Both raw and processed data formats available

## Assignment Materials

- **`assignment/`**: Contains project requirements and explanations
- PDF documents with detailed specifications
- Reference materials for implementation guidance
