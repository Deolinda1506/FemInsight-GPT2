import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your fine-tuned model and tokenizer (using cache for efficiency)
# @gr.cache # Gradio caching might not be suitable for model loading in this context
def load_model():
    model_path = "/content/content/fine-tuned-gpt2-feminsight"
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

model, tokenizer = load_model()

# Define the generation function (same as before)
def feminsight_generate(prompt, max_length=150, temperature=0.8, top_k=50, top_p=0.9):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()
    return response

# Create Gradio Interface
iface = gr.Interface(
    fn=feminsight_generate,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    # Change output back to Textbox and set lines for minimum height
    outputs=gr.Textbox(label="FemInsight Response", lines=10), # Changed back to Textbox and added lines
    title="🌸 FemInsight GPT-2 Interactive App 🩸",
    description="✨ Enter a question related to menstrual health and get a response from the fine-tuned GPT-2 model. ✨",
    theme=gr.themes.Soft(), # Using a built-in light theme
    examples=[
        ["What are healthy ways to manage menstrual cramps?"],
        ["How does stress affect the menstrual cycle?"],
        ["Is it normal to experience mood swings during menstruation?"],
        ["What is considered a normal amount of menstrual bleeding?"]
    ]
)

# Launch the Gradio App - Note: running from app.py typically doesn't use inline=True
# iface.launch(inline=True, height=800) # This line is for running inline in the notebook
