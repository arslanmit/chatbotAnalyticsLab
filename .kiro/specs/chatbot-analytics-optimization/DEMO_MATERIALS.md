# Demo Materials: Chatbot Analytics and Optimization

**Project:** Retail Banking Chatbot Analytics and Optimization System  
**Date:** October 17, 2025  
**Purpose:** Live demonstration and walkthrough materials  

---

## 1. Dashboard Demo Walkthrough

### Demo Script Overview

**Duration:** 15-20 minutes  
**Audience:** Technical and non-technical stakeholders  
**Objective:** Showcase system capabilities and user experience  

### Pre-Demo Setup Checklist

**Environment Preparation:**
```bash
# 1. Ensure all services are running
docker-compose up -d

# 2. Verify dashboard is accessible
curl http://localhost:8501

# 3. Check API health
curl http://localhost:8000/health

# 4. Load sample data (if needed)
python examples/dataset_pipeline_demo.py

# 5. Clear browser cache for clean demo
# Open browser in incognito/private mode
```

**Data Verification:**
- ✅ At least 10 experiments logged
- ✅ Multiple model versions available
- ✅ Recent training runs completed
- ✅ Conversation data loaded
- ✅ Sentiment data available

**Browser Setup:**
- Open dashboard: `http://localhost:8501`
- Prepare multiple tabs for different pages
- Ensure screen resolution is 1920x1080 or higher
- Test all interactive elements beforehand


### Demo Walkthrough Script

#### Part 1: Overview Page (3 minutes)

**Talking Points:**
```
"Welcome to the Chatbot Analytics Dashboard. This is our executive overview 
page, designed to give leadership a quick snapshot of system health and 
recent activity."

[Point to metric cards]
"Here we can see we've logged 42 experiments, tracked 8 different model 
architectures, and achieved 35 successful training runs. That's an 83% 
success rate, which indicates a mature experimentation process."

[Point to latest accuracy]
"Our most recent model achieved 87.3% validation accuracy, which exceeds 
our target of 85%. The upward arrow shows we've improved by 2.1 percentage 
points from the previous run."

[Scroll to recent experiments table]
"This table shows our recent training history. Notice we can see the model 
type, accuracy, and timestamp for each experiment. Let me show you how we 
can export this data..."

[Click Download CSV button]
"With one click, we can export this data for further analysis in Excel or 
other tools."
```

**Interactive Elements to Demonstrate:**
- Hover over metric cards to show tooltips
- Click on experiment rows to highlight
- Demonstrate CSV export functionality
- Show refresh button and last update timestamp

#### Part 2: Experiments Page (4 minutes)

**Talking Points:**
```
[Navigate to Experiments page]
"The Experiments page is where our data science team tracks model training 
history and compares different approaches."

[Use model filter]
"We can filter by specific model architectures. Let me select 'bert-base' 
to see only BERT experiments..."

[Point to metrics visualization]
"This chart shows how our accuracy has improved over time. You can see we 
started at around 75% and have steadily climbed to 87.3%."

[Demonstrate date range filter]
"We can also filter by date range. Let me show you just the last 7 days..."

[Show experiment comparison]
"One powerful feature is experiment comparison. I can select multiple runs 
and see their metrics side-by-side. This helps us understand which 
hyperparameters or data preprocessing steps made the biggest difference."
```

**Interactive Elements to Demonstrate:**
- Model ID dropdown filter
- Date range selector
- Experiment selection checkboxes
- Metrics comparison table
- Export to PDF functionality


#### Part 3: Intent Distribution Page (3 minutes)

**Talking Points:**
```
[Navigate to Intent Distribution page]
"This page analyzes the distribution of customer intents across our 
banking dataset. Understanding intent frequency helps us prioritize 
optimization efforts."

[Select dataset]
"We support multiple datasets. Let me select BANKING77, which has 77 
fine-grained banking intent categories..."

[Adjust top N slider]
"We can control how many intents to display. Let me show the top 20..."

[Point to bar chart]
"As you can see, 'balance' and 'transfer' are the most common intents, 
accounting for about 15% of all queries. This tells us these are critical 
paths to optimize."

[Show statistics]
"Below the chart, we have detailed statistics: total intents, unique 
categories, and coverage metrics. This helps us understand if we're 
handling the full range of customer needs."
```

**Interactive Elements to Demonstrate:**
- Dataset selector dropdown
- Top N intents slider (adjust from 10 to 30)
- Hover over bars to see exact counts
- Export chart as PNG
- Export data as CSV

#### Part 4: Conversation Flow Page (4 minutes)

**Talking Points:**
```
[Navigate to Conversation Flow page]
"The Conversation Flow page helps us understand how conversations progress 
and where users might be getting stuck."

[Adjust sample size]
"For performance, we can control the sample size. Let me analyze 200 
conversations..."

[Point to turn statistics]
"We can see the average conversation takes 3.4 turns, with a median of 3 
and a maximum of 12. This tells us most conversations are efficient, but 
some require more back-and-forth."

[Show state distribution chart]
"This chart shows the distribution of conversation states. Most 
conversations reach 'completed' status, which is great. But we can also 
see some 'abandoned' and 'escalated' cases that need attention."

[Demonstrate transition matrix]
"The transition matrix shows how conversations flow from one state to 
another. This helps us identify common paths and potential bottlenecks."
```

**Interactive Elements to Demonstrate:**
- Sample size slider
- State distribution pie chart (hover for percentages)
- Transition matrix heatmap (hover for probabilities)
- Export functionality


#### Part 5: Sentiment Trends Page (3 minutes)

**Talking Points:**
```
[Navigate to Sentiment Trends page]
"Customer satisfaction is critical, so we track sentiment across all 
conversations."

[Select granularity]
"We can view sentiment at different granularities: by conversation, daily, 
or hourly. Let me show daily trends..."

[Point to trend line]
"This chart shows sentiment over time. We're maintaining a positive 
sentiment score of around 0.4, which indicates satisfied customers. The 
trend is stable with slight upward movement."

[Show summary statistics]
"Our summary statistics show 65% positive sentiment, 25% neutral, and only 
10% negative. This is well within our target ranges."

[Demonstrate alert]
"If negative sentiment spikes above 20%, the system automatically generates 
an alert so we can investigate and respond quickly."
```

**Interactive Elements to Demonstrate:**
- Granularity selector (conversation, daily, hourly)
- Trend line chart with zoom and pan
- Sentiment distribution pie chart
- Summary statistics table
- Export functionality

#### Part 6: Settings and Configuration (2 minutes)

**Talking Points:**
```
[Navigate to Settings page]
"The Settings page provides access to documentation and configuration 
options."

[Show auto-refresh toggle]
"We can enable auto-refresh to keep the dashboard updated in real-time. 
This is useful for monitoring during active operations."

[Demonstrate refresh interval]
"The refresh interval can be adjusted from 10 seconds to 5 minutes, 
depending on your needs."

[Point to documentation links]
"We have comprehensive documentation covering API usage, deployment guides, 
and troubleshooting. All accessible directly from the dashboard."
```

**Interactive Elements to Demonstrate:**
- Auto-refresh toggle
- Refresh interval slider
- Manual refresh button
- Documentation links
- Last refresh timestamp


### Demo Closing (1 minute)

**Talking Points:**
```
"To summarize what we've seen:

1. The dashboard provides real-time visibility into chatbot performance
2. We can track model training experiments and compare results
3. Intent distribution analysis helps prioritize optimization efforts
4. Conversation flow tracking identifies bottlenecks and drop-off points
5. Sentiment monitoring ensures we maintain high customer satisfaction

The system is production-ready, with comprehensive monitoring, alerting, 
and export capabilities. It's designed to support both technical teams 
doing deep analysis and executives who need high-level insights.

Are there any questions or specific features you'd like to explore further?"
```

---

## 2. Code Examples and Explanations

### Example 1: Loading and Processing a Dataset

**Purpose:** Demonstrate how to load and preprocess banking conversation data

**Code:**
```python
from src.services.dataset_loader import DatasetLoader
from src.config.dataset_config import DatasetType

# Initialize the dataset loader
loader = DatasetLoader()

# Load BANKING77 dataset
dataset = loader.load_dataset(
    dataset_type=DatasetType.BANKING77,
    preprocess=True,
    normalize_text=True
)

# Display dataset statistics
print(f"Total samples: {len(dataset)}")
print(f"Unique intents: {len(dataset.get_unique_intents())}")
print(f"Average text length: {dataset.get_avg_text_length():.1f} characters")

# Show sample conversation
sample = dataset[0]
print(f"\nSample query: {sample['text']}")
print(f"Intent: {sample['intent']}")
print(f"Confidence: {sample.get('confidence', 'N/A')}")
```

**Explanation:**
```
This code demonstrates the core data loading functionality. The DatasetLoader 
class handles multiple dataset formats (JSON, CSV, Parquet) and provides 
consistent preprocessing:

1. Text normalization (lowercase, punctuation removal)
2. Intent label standardization
3. Data validation and quality checks
4. Train/validation/test splitting

The preprocess=True flag enables automatic text cleaning, while 
normalize_text=True applies additional normalization steps like removing 
extra whitespace and special characters.
```

**Expected Output:**
```
Total samples: 13083
Unique intents: 77
Average text length: 45.3 characters

Sample query: what is my account balance
Intent: balance
Confidence: N/A
```


### Example 2: Training an Intent Classifier

**Purpose:** Show how to train a BERT-based intent classification model

**Code:**
```python
from src.models.intent_classifier import IntentClassifier
from src.services.dataset_loader import DatasetLoader
from src.config.dataset_config import DatasetType

# Load training data
loader = DatasetLoader()
dataset = loader.load_dataset(DatasetType.BANKING77, preprocess=True)

# Split into train/validation/test
train_data, val_data, test_data = dataset.train_test_split(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
)

# Initialize classifier
classifier = IntentClassifier(
    model_name="bert-base-uncased",
    num_labels=77,
    learning_rate=2e-5,
    batch_size=16,
    num_epochs=3
)

# Train the model
training_results = classifier.train(
    train_data=train_data,
    val_data=val_data,
    save_path="models/intent_classifier"
)

# Evaluate on test set
test_results = classifier.evaluate(test_data)

print(f"Training completed!")
print(f"Validation Accuracy: {training_results['val_accuracy']:.3f}")
print(f"Test Accuracy: {test_results['accuracy']:.3f}")
print(f"Test F1-Score: {test_results['f1_macro']:.3f}")
```

**Explanation:**
```
This example demonstrates the complete model training pipeline:

1. Data Loading: Load and preprocess the BANKING77 dataset
2. Data Splitting: Create train/validation/test splits with fixed random seed
3. Model Initialization: Configure BERT-based classifier with hyperparameters
4. Training: Train the model with automatic validation and checkpointing
5. Evaluation: Assess performance on held-out test set

The IntentClassifier class handles:
- Tokenization using BERT tokenizer
- GPU acceleration (if available)
- Learning rate scheduling
- Early stopping based on validation loss
- Model checkpointing (saves best model)
- Comprehensive metrics calculation
```

**Expected Output:**
```
Training completed!
Validation Accuracy: 0.873
Test Accuracy: 0.869
Test F1-Score: 0.842
```


### Example 3: Making Predictions with Confidence Scores

**Purpose:** Demonstrate real-time intent classification with confidence scoring

**Code:**
```python
from src.models.intent_classifier import IntentClassifier

# Load trained model
classifier = IntentClassifier.load_from_checkpoint(
    "models/intent_classifier/best_model.pt"
)

# Single prediction
query = "I want to transfer money to my savings account"
prediction = classifier.predict(query)

print(f"Query: {query}")
print(f"Predicted Intent: {prediction['intent']}")
print(f"Confidence: {prediction['confidence']:.3f}")
print(f"\nTop 3 Alternative Intents:")
for alt_intent, alt_conf in prediction['alternatives'][:3]:
    print(f"  - {alt_intent}: {alt_conf:.3f}")

# Batch prediction
queries = [
    "What is my account balance?",
    "I lost my credit card",
    "How do I apply for a loan?",
    "What are your interest rates?"
]

batch_predictions = classifier.predict_batch(queries)

print(f"\nBatch Predictions:")
for query, pred in zip(queries, batch_predictions):
    print(f"{query[:40]:40s} -> {pred['intent']:20s} ({pred['confidence']:.3f})")
```

**Explanation:**
```
This example shows how to use a trained model for inference:

1. Model Loading: Load a saved checkpoint from disk
2. Single Prediction: Classify a single query with confidence score
3. Alternative Intents: Get top-k alternative predictions for uncertainty
4. Batch Prediction: Process multiple queries efficiently

The predict() method returns:
- intent: The most likely intent category
- confidence: Probability score (0-1)
- alternatives: List of (intent, confidence) tuples for top-k predictions

Batch prediction is optimized for throughput, processing multiple queries 
in parallel on GPU when available.
```

**Expected Output:**
```
Query: I want to transfer money to my savings account
Predicted Intent: transfer
Confidence: 0.945

Top 3 Alternative Intents:
  - balance: 0.032
  - transaction_history: 0.015
  - savings_account: 0.008

Batch Predictions:
What is my account balance?              -> balance              (0.978)
I lost my credit card                    -> card_lost            (0.892)
How do I apply for a loan?               -> loan_application     (0.856)
What are your interest rates?            -> interest_rate        (0.923)
```


### Example 4: Analyzing Conversation Flows

**Purpose:** Demonstrate conversation analysis and pattern detection

**Code:**
```python
from src.services.conversation_analyzer import ConversationAnalyzer
from src.services.dataset_loader import DatasetLoader
from src.config.dataset_config import DatasetType

# Load conversation data
loader = DatasetLoader()
dataset = loader.load_dataset(DatasetType.SCHEMA_GUIDED, preprocess=True)

# Initialize analyzer
analyzer = ConversationAnalyzer()

# Analyze conversation flows
flow_analysis = analyzer.analyze_dialogue_flow(dataset.conversations)

print("Conversation Flow Analysis:")
print(f"Total conversations: {flow_analysis['total_conversations']}")
print(f"Average turns: {flow_analysis['avg_turns']:.2f}")
print(f"Median turns: {flow_analysis['median_turns']}")
print(f"Max turns: {flow_analysis['max_turns']}")
print(f"\nCompletion rate: {flow_analysis['completion_rate']:.1%}")
print(f"Abandonment rate: {flow_analysis['abandonment_rate']:.1%}")
print(f"Escalation rate: {flow_analysis['escalation_rate']:.1%}")

# Detect failure points
failure_points = analyzer.detect_failure_points(dataset.conversations)

print(f"\nTop 5 Failure Points:")
for i, failure in enumerate(failure_points[:5], 1):
    print(f"{i}. Turn {failure['turn_index']}: {failure['reason']}")
    print(f"   Frequency: {failure['frequency']} ({failure['percentage']:.1%})")
    print(f"   Suggested fix: {failure['suggestion']}")

# Calculate success metrics
success_metrics = analyzer.calculate_success_metrics(dataset.conversations)

print(f"\nSuccess Metrics:")
print(f"First Contact Resolution: {success_metrics['fcr']:.1%}")
print(f"Average Resolution Time: {success_metrics['avg_resolution_time']:.1f}s")
print(f"User Satisfaction Score: {success_metrics['satisfaction_score']:.2f}/5")
```

**Explanation:**
```
The ConversationAnalyzer provides comprehensive conversation analysis:

1. Flow Analysis: Statistics on conversation length and structure
2. Failure Detection: Identifies common drop-off points and reasons
3. Success Metrics: Calculates resolution rates and satisfaction

Key insights from this analysis:
- Average turns indicates conversation efficiency
- Completion rate shows how often users achieve their goals
- Failure points highlight areas needing optimization
- Success metrics tie to business KPIs

This data drives optimization decisions like improving fallback responses, 
adding clarification prompts, or redesigning conversation flows.
```

**Expected Output:**
```
Conversation Flow Analysis:
Total conversations: 1247
Average turns: 3.42
Median turns: 3
Max turns: 12

Completion rate: 83.2%
Abandonment rate: 11.5%
Escalation rate: 5.3%

Top 5 Failure Points:
1. Turn 2: Low confidence intent classification
   Frequency: 87 (7.0%)
   Suggested fix: Add clarification prompt for ambiguous queries

2. Turn 4: Missing required information
   Frequency: 64 (5.1%)
   Suggested fix: Implement progressive disclosure for complex tasks

3. Turn 3: User frustration detected
   Frequency: 52 (4.2%)
   Suggested fix: Offer human escalation earlier in conversation

4. Turn 5: API timeout
   Frequency: 38 (3.0%)
   Suggested fix: Optimize backend service response times

5. Turn 2: Out-of-scope query
   Frequency: 31 (2.5%)
   Suggested fix: Expand intent taxonomy or improve routing

Success Metrics:
First Contact Resolution: 68.3%
Average Resolution Time: 142.5s
User Satisfaction Score: 4.28/5
```


---

## 3. System Architecture Presentation

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                             │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │   Streamlit  │  │   FastAPI    │  │  CLI Tools   │            │
│  │   Dashboard  │  │   REST API   │  │  & Scripts   │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   SERVICES   │   │    MODELS    │   │  MONITORING  │
│              │   │              │   │              │
│ • Dataset    │   │ • Intent     │   │ • Metrics    │
│   Loader     │   │   Classifier │   │   Tracker    │
│ • Preprocess │   │ • Evaluator  │   │ • Alerting   │
│ • Analyzer   │   │ • Optimizer  │   │ • Logging    │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      REPOSITORIES                                   │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │  Experiment  │  │ Conversation │  │    Model     │            │
│  │  Repository  │  │  Repository  │  │  Repository  │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA STORAGE                                   │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │    SQLite    │  │   Parquet    │  │    Model     │            │
│  │   Metadata   │  │   Datasets   │  │  Checkpoints │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Descriptions

**User Interfaces Layer:**
- **Streamlit Dashboard:** Interactive web-based analytics interface
- **FastAPI REST API:** Programmatic access for integrations
- **CLI Tools:** Command-line utilities for automation

**Services Layer:**
- **Dataset Loader:** Multi-format data ingestion (JSON, CSV, Parquet)
- **Preprocessor:** Text normalization, cleaning, feature extraction
- **Analyzer:** Conversation flow analysis, pattern detection

**Models Layer:**
- **Intent Classifier:** BERT-based transformer for intent detection
- **Evaluator:** Performance metrics calculation and reporting
- **Optimizer:** Hyperparameter tuning and model selection

**Monitoring Layer:**
- **Metrics Tracker:** Real-time performance monitoring
- **Alerting:** Threshold-based notifications
- **Logging:** Comprehensive audit trails

**Repositories Layer:**
- **Experiment Repository:** Training history and metadata
- **Conversation Repository:** Dialogue data and annotations
- **Model Repository:** Trained model artifacts and versions

**Data Storage Layer:**
- **SQLite:** Lightweight relational database for metadata
- **Parquet:** Columnar storage for large datasets
- **Model Checkpoints:** Serialized model weights and configurations


### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION FLOW                              │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Raw Datasets │
                    │ (JSON/CSV)   │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │   Validation │
                    │   & Cleaning │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Preprocessing│
                    │ & Normalization│
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  Feature     │
                    │  Extraction  │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  Processed   │
                    │  Dataset     │
                    └──────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINING FLOW                                    │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Train/Val/   │
                    │ Test Split   │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Model        │
                    │ Initialization│
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Training     │
                    │ Loop         │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Validation   │
                    │ & Checkpointing│
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Model        │
                    │ Evaluation   │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Save Best    │
                    │ Model        │
                    └──────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    INFERENCE FLOW                                   │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ User Query   │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Text         │
                    │ Preprocessing│
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Tokenization │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Model        │
                    │ Inference    │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Confidence   │
                    │ Scoring      │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Intent       │
                    │ Prediction   │
                    └──────────────┘
```

### Technology Stack Details

**Machine Learning:**
- **Framework:** PyTorch 2.0+
- **Transformers:** HuggingFace Transformers 4.30+
- **Model:** BERT-base-uncased (110M parameters)
- **Training:** AdamW optimizer, linear learning rate schedule
- **Evaluation:** scikit-learn metrics (accuracy, precision, recall, F1)

**Backend:**
- **API Framework:** FastAPI 0.100+
- **ORM:** SQLAlchemy 2.0+
- **Database:** SQLite 3.40+
- **Data Processing:** pandas 2.0+, numpy 1.24+
- **File Format:** Parquet (pyarrow 12.0+)

**Frontend:**
- **Dashboard:** Streamlit 1.25+
- **Visualization:** Plotly 5.15+, matplotlib 3.7+, seaborn 0.12+
- **UI Components:** Streamlit native widgets

**DevOps:**
- **Containerization:** Docker 24.0+, Docker Compose 2.20+
- **Testing:** pytest 7.4+, pytest-cov 4.1+
- **Linting:** black, flake8, mypy
- **CI/CD:** GitHub Actions (optional)

**Deployment:**
- **Environment:** Python 3.9+
- **OS:** Linux (Ubuntu 22.04), macOS, Windows
- **Hardware:** CPU (minimum), GPU (recommended for training)
- **Memory:** 8GB RAM (minimum), 16GB+ (recommended)


---

## 4. Live Demo Instructions

### Option A: Local Demo

**Prerequisites:**
```bash
# Ensure Docker and Docker Compose are installed
docker --version  # Should be 24.0+
docker-compose --version  # Should be 2.20+

# Ensure Python environment is set up
python --version  # Should be 3.9+
```

**Step-by-Step Demo:**

1. **Start Services:**
```bash
# Navigate to project root
cd /path/to/chatbot-analytics-optimization

# Start all services
docker-compose up -d

# Verify services are running
docker-compose ps
```

2. **Access Dashboard:**
```bash
# Open browser to dashboard
open http://localhost:8501

# Or manually navigate to:
# http://localhost:8501
```

3. **Verify API:**
```bash
# Test API health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "version": "1.0.0"}
```

4. **Load Sample Data (if needed):**
```bash
# Run dataset pipeline demo
python examples/dataset_pipeline_demo.py

# This will:
# - Load BANKING77 dataset
# - Preprocess and validate data
# - Display statistics
```

5. **Train a Model (optional):**
```bash
# Quick training run (5 minutes)
python examples/train_intent_classifier_quick.py

# Full training run (30-60 minutes)
python examples/train_intent_classifier.py
```

6. **Navigate Dashboard:**
- Overview page: System health and recent activity
- Experiments page: Model training history
- Intent Distribution: Query analysis
- Conversation Flow: Dialogue patterns
- Sentiment Trends: Customer satisfaction

7. **Demonstrate Features:**
- Filter experiments by model ID
- Adjust date ranges
- Export data to CSV/PDF
- Show real-time metrics
- Demonstrate auto-refresh

8. **Stop Services:**
```bash
# Stop all services
docker-compose down

# Or keep running for continued demo
```

### Option B: Video Recording Demo

**Recording Setup:**
```bash
# Use screen recording software:
# - macOS: QuickTime Player (Cmd+Shift+5)
# - Windows: Xbox Game Bar (Win+G)
# - Linux: SimpleScreenRecorder, OBS Studio

# Recommended settings:
# - Resolution: 1920x1080
# - Frame rate: 30 fps
# - Audio: Include microphone narration
# - Duration: 10-15 minutes
```

**Recording Script:**

**Segment 1: Introduction (1 minute)**
- Show project overview slide
- Explain system purpose and capabilities
- Preview what will be demonstrated

**Segment 2: Dashboard Overview (2 minutes)**
- Navigate to dashboard homepage
- Highlight key metrics
- Show recent experiments table
- Demonstrate export functionality

**Segment 3: Experiments Analysis (3 minutes)**
- Filter experiments by model
- Show accuracy trends over time
- Compare multiple experiments
- Export comparison report

**Segment 4: Intent and Flow Analysis (3 minutes)**
- Show intent distribution chart
- Analyze conversation flow statistics
- Demonstrate transition matrix
- Highlight drop-off points

**Segment 5: Sentiment Monitoring (2 minutes)**
- Display sentiment trends
- Show summary statistics
- Demonstrate alert system
- Export sentiment report

**Segment 6: Code Examples (3 minutes)**
- Show dataset loading code
- Demonstrate model training
- Make predictions with confidence scores
- Analyze conversation flows

**Segment 7: Conclusion (1 minute)**
- Summarize key features
- Highlight business impact
- Provide next steps
- Call to action


### Demo Troubleshooting

**Common Issues and Solutions:**

**Issue 1: Dashboard won't load**
```bash
# Check if services are running
docker-compose ps

# Restart services
docker-compose restart

# Check logs
docker-compose logs dashboard
```

**Issue 2: No data displayed**
```bash
# Load sample data
python examples/dataset_pipeline_demo.py

# Verify data files exist
ls -la data/processed/
ls -la experiments/
```

**Issue 3: Slow performance**
```bash
# Clear cache
rm -rf .mypy_cache/
rm -rf __pycache__/

# Reduce sample size in dashboard
# Use slider to select smaller sample (e.g., 50-100 conversations)
```

**Issue 4: API not responding**
```bash
# Check API health
curl http://localhost:8000/health

# Restart API service
docker-compose restart api

# Check API logs
docker-compose logs api
```

---

## 5. Demo Checklist

### Pre-Demo Checklist

- [ ] All services running (docker-compose ps shows "Up")
- [ ] Dashboard accessible at http://localhost:8501
- [ ] API accessible at http://localhost:8000
- [ ] Sample data loaded (at least 10 experiments)
- [ ] Browser cache cleared (use incognito mode)
- [ ] Screen resolution set to 1920x1080 or higher
- [ ] Audio/microphone tested (if recording)
- [ ] Backup slides prepared (in case of technical issues)
- [ ] Demo script reviewed and practiced
- [ ] Questions and answers prepared

### During Demo Checklist

- [ ] Start with overview and context
- [ ] Navigate through all dashboard pages
- [ ] Demonstrate interactive features (filters, exports)
- [ ] Show code examples with explanations
- [ ] Highlight business impact and ROI
- [ ] Address questions as they arise
- [ ] Keep to time limit (15-20 minutes)
- [ ] Conclude with summary and next steps

### Post-Demo Checklist

- [ ] Answer remaining questions
- [ ] Share demo recording (if recorded)
- [ ] Provide access to documentation
- [ ] Share code repository link
- [ ] Schedule follow-up meetings
- [ ] Collect feedback
- [ ] Document lessons learned
- [ ] Update demo materials based on feedback

---

**End of Demo Materials**
