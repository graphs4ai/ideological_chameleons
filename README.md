# Ideological Chameleon

Research project investigating the behavior of language models (LLMs) regarding political positions in the Brazilian context.

## Description

This project evaluates how different LLMs respond to political statements when exposed to prompts simulating users with different ideological orientations (left, right, or neutral). The goal is to measure the **Ideological Position Index (IPI)** and the **Chameleon Index (CI)** to analyze whether models adapt their responses according to the political bias suggested in the prompt.

## Features

- **Statement pair generation**: Creates pairs of political statements about Brazilian topics using LLMs
- **Statement validation**: Validates whether generated statements are effectively aligned with expected political spectrums
- **Response collection**: Evaluates multiple models at different temperatures and prompt conditions
- **Statistical analysis**: Calculates polarization indices and generates result visualizations

## Project Structure

```
.
├── main.py                  # Main script for collecting LLM responses
├── create_pairs.py          # Political statement pair generation
├── validate_pairs.py        # Validation of generated statements
├── run_analysis.py          # Statistical analysis and chart generation
├── conf/                    # Configuration files (YAML)
├── dados/                   # Input and output data
├── src/                     # Modular source code
│   ├── analysis/           # Analysis modules (statistics, charts)
│   └── main/               # General utilities
├── outputs/                 # Execution results
└── analises_figures/        # Generated figures (PDF, PNG, SVG)
```

## Configuration

Project configurations are in YAML files in the `conf/` folder:

- `config.yaml`: Models to evaluate, temperatures, ideological prompts
- `analysis_config.yaml`: Settings for analysis and chart generation
- `create_and_validate_pairs_config.yaml`: Settings for pair creation
- `validate_config.yaml`: Settings for validation

## Evaluated Models

The project evaluates several language models, including:

- **Brazilian models**: Maritaca (Sabia-3.1)
- **Google**: Gemma (multiple versions), Gemini
- **Meta**: LLaMA (multiple versions)
- **Mistral AI**: Mixtral, Mistral Small
- **OpenAI**: GPT (multiple versions)
- **Qwen**, **DeepSeek**, **Microsoft Phi**, **NVIDIA Nemotron**, **Grok**

## Thematic Axes

Political statements cover 7 thematic axes:

1. Social Policies
2. Economy
3. Public Security
4. Environment
5. Democratic Institutions
6. Corruption and Justice
7. Education and Culture

## Data Output

- **Responses**: CSV with all model responses (`dados/respostas_finais.csv`)
- **Validated pairs**: JSON with validated statement pairs
- **Analyses**: Charts in multiple formats (PDF, PNG, SVG)
- **Cache**: Caching system to optimize API requests

## Technologies

- **Python 3.12**
- **Hydra**: Configuration management
- **Pandas**: Data manipulation
- **Seaborn/Matplotlib**: Visualization
- **LLM APIs**: OpenAI, DeepInfra, Maritaca, Grok, Google

## Author

Anderson Soares (a241149@dac.unicamp.br)
Bruno Veiga (b.veiga74@gmail.com)
