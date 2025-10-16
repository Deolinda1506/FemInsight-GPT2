# Use a Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the notebook
COPY FemInsight_GPT2.ipynb .

# Expose the port Gradio runs on
EXPOSE 7860

# Run the notebook and start the Gradio app
CMD ["jupyter", "nbconvert", "--to", "script", "FemInsight_GPT2.ipynb", "--stdout"] | python -
