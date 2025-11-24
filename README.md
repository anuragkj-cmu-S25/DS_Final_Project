# Automated Data Science Workflow System: DS Final (Anurag)

An intelligent system for end-to-end automated data science analysis, from CSV upload to publication-ready research reports. Built with LLMs and multi-agent collaboration.

## 🎯 Overview

This project automates the complete data science pipeline, eliminating repetitive tasks and preventing common pitfalls like data leakage. Two complementary approaches are provided:

1. **LLM Pipeline Approach**: Fast, streamlined workflow for quick exploratory analysis
2. **Multi-Agent System**: Comprehensive research-grade analysis with quality control and literature review

## ✨ Key Features

- **Zero-Code Analysis**: Upload CSV and get complete analysis automatically
- **Data Leakage Prevention**: Automated validation ensures proper train/test separation
- **Explainable AI**: Detailed reasoning for every decision (variable selection, encoding, model choice)
- **Publication-Ready Reports**: Full research reports with methodology, results, and visualizations
- **Multi-Agent Collaboration**: Specialized agents for coding, visualization, quality review, and reporting
- **Human-in-the-Loop**: Review and approve at critical stages

## 📊 Supported Tasks

- **Classification**: Logistic Regression, Random Forest, XGBoost, SVC, Gradient Boosting
- **Regression**: Linear, Ridge, Lasso, Random Forest, Gradient Boosting
- **Clustering**: K-Means, DBSCAN, Gaussian Mixture, Hierarchical

## 🚀 Quick Start

### LLM Pipeline Approach
```bash
cd LLM_Pipeline_Approach
pip install -r requirements.txt
streamlit run app/main.py
```

### Multi-Agent System
```bash
cd Multi_Agent_Approach
pip install -r requirements.txt
# Configure API keys in .env
python main.py
```

## 📁 Project Structure

```
.
├── LLM_Pipeline_Approach/     # Fast pipeline system
│   ├── app/src/               # Core modules
│   └── config/                # Configuration
│
└── Multi_Agent_Approach/      # Research-grade system
    ├── src/agents/            # 9 specialized agents
    ├── src/core/              # LangGraph workflow
    ├── src/tools/             # Agent tools
    └── config/                # Model configuration
```

## 🔧 Requirements

- Python 3.8+
- OpenAI API key (GPT-4)
- Google AI API key (Gemini) - for Multi-Agent approach
- See individual `requirements.txt` files for complete dependencies

## 📈 Performance

**Time Savings**: Reduces analysis time from hours to 5-30 minutes  
**Accuracy**: Matches expert-level performance (96-98% on standard datasets)  
**Cost**: $0.02-0.50 per analysis depending on approach

## 🎓 Use Cases

- **Students**: Learn data science best practices with explainable AI guidance
- **Researchers**: Generate comprehensive analysis reports with literature review
- **Business Analysts**: Quick insights from data without coding
- **Data Scientists**: Automate repetitive exploratory data analysis

## 📝 Example Output

Input: `housing_data.csv` (1460 rows, 81 features)  
Output:
- Hypothesis with literature review
- Clean preprocessing code (no data leakage)
- 5+ visualizations (correlation, feature importance, predictions)
- 12-page research report with 28+ references
- All in 15-30 minutes

## ⚙️ Configuration

Both approaches support multiple LLM providers:
- OpenAI (GPT-4, GPT-4-mini)
- Google (Gemini-2.5-Pro, Gemini-2.5-Flash)
- Anthropic (Claude)
- Ollama (local models)

Configure in `config/agent_models.yaml` (Multi-Agent) or `.env` files.