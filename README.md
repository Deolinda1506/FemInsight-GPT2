# FemInsight - Domain-Specific Women's Health Chatbot

## Project Overview

**FemInsight** addresses the critical gap in accessible period education by providing period tracking advice, detailed symptom explanations, evidence-based pain management tips, and comprehensive cycle education. Built using fine-tuned GPT-2 transformer model and deployed with an intuitive web interface.

### Domain Focus

FemInsight operates in the healthcare domain, specifically focusing on women's health and menstrual wellness. The target audience includes adolescent and young adult women seeking period education, women looking for reliable menstrual health information, anyone wanting to understand menstrual cycles better, and partners and family members seeking to provide support. The project addresses the lack of accessible, accurate women's health education by covering specific topics including menstrual cycle management, PMS symptoms, and general health & wellness.

## Technical Implementation

### Model Architecture
- **Base Model**: GPT-2
- **Fine-tuning**: Custom dataset of 530 women's health Q&A pairs
- **Framework**: Hugging Face Transformers with PyTorch
- **Deployment**: Gradio web interface

### Dataset Information
- **Source**: [Menstrual-Health-Awareness-Dataset](https://huggingface.co/datasets/gjyotk/Menstrual-Health-Awareness-Dataset) (Hugging Face)
- **Size**: 530 training samples, 45 test samples
- **Format**: Instruction-output pairs (question-answer)
- **Quality**: High-quality women's health Q&A dataset


### Hyperparameter Experiments
| Experiment | Learning Rate | Batch Size | Epochs | Weight Decay | Eval Loss |
|------------|---------------|------------|--------|--------------|-----------|
| 1          | 5e-5         | 16         | 10     | 0.01         | 1.016883  |
| 2          | 1e-5         | 16         | 20     | 0.1          | 1.147389  |
| 3          | 2e-5         | 16         | 15     | 0.05         | 1.074074  |
| 4          | 1e-5         | 8          | 10     | 0.0          | 1.216039  |
| 5          | 1e-4         | 32         | 25     | 0.05         | 1.126583  |

**Best Performance**: Experiment 1 achieved the lowest evaluation loss (1.016883), outperforming the worst experiment by 16% (Experiment 4: 1.216039 → Experiment 1: 1.016883)
### Performance Evaluation

#### Quantitative Metrics
- **BLEU Score**: 0.0714 (text similarity to reference answers)
- **F1 Score**: 0.1637 (token-based overlap between generated and reference)
- **Perplexity**: 26.06 (model confidence in generated responses)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Deolinda1506/FemInsight-GPT2.git
   cd FemInsight-GPT2
   ```

2. **Install required packages**
   ```bash
   pip install gradio transformers torch pandas numpy matplotlib seaborn nltk ipywidgets jupyter
   ```

3. **Run the Gradio app**
   ```bash
   python Feminsight_app.py
   ```

4. **Open your browser** and navigate to the provided local URL

## Example Conversations

### Sample Interaction 1
**User**: "What is a normal menstrual cycle length?"
**FemInsight**: "A normal menstrual cycle typically ranges from 21 to 35 days, with the average cycle lasting around 28 days."

### Sample Interaction 2
**User**: "How can I alleviate menstrual cramps?"
**FemInsight**: "Some effective ways to manage menstrual cramps include applying heat to your lower abdomen, taking over-the-counter pain relievers like ibuprofen, gentle exercise like walking or yoga, staying hydrated, and getting adequate rest."

### Sample Interaction 3
**User**: "When should I see a doctor about my period?"
**FemInsight**: "You should see a doctor if you experience severe pain that doesn't respond to medication, heavy bleeding that soaks through pads/tampons every hour, bleeding that lasts more than 7 days, or if you have concerns about irregular cycles."



## Project Structure

```
FemInsight-GPT2/
├── Feminsight_app.py            # Gradio web interface
├── Notebook/
│   └── FemInsight_GPT2.ipynb    # Training and evaluation
└── README.md                    # This file
```

## Technical Details

### Data Preprocessing
1. **Text Cleaning**: Remove special characters, normalize case
2. **Tokenization**: GPT-2 tokenizer using Byte Pair Encoding (BPE)
3. **Normalization**: Lemmatization and stop word removal
4. **Formatting**: Instruction-output format for fine-tuning

### Evaluation Methodology
- **BLEU Score**: Measures text similarity to reference answers
- **F1 Score**: Token-based overlap between generated and reference
- **Perplexity**: Model confidence in generated responses
- **Qualitative Testing**: Manual evaluation of response quality


## Future Enhancements

- **Multi-language Support**: Spanish, French, and other languages
- **Voice Interface**: Speech-to-text and text-to-speech
- **Mobile App**: Flutter application
- **Integration**: Connect with health tracking apps
- **Personalization**: Customized responses based on user history


## Acknowledgments

- **Dataset**: [Menstrual-Health-Awareness-Dataset](https://huggingface.co/datasets/gjyotk/Menstrual-Health-Awareness-Dataset) by gjyotk
- **Model**: GPT-2 by OpenAI
- **Framework**: Hugging Face Transformers and Gradio


