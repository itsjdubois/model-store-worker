import runpod
import torch
import os
from transformers import pipeline

os.environ['HF_HOME'] = '/runpod/model-store/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/runpod/model-store/huggingface'

MODEL_NAME = os.environ.get('MODEL_NAME', 'stabilityai/stablelm-zephyr-3b')
MODEL_REVISION = os.environ.get('MODEL_REVISION', 'main')

print(f"Loading model: {MODEL_NAME} (revision: {MODEL_REVISION})")
print(f"Using cache directory: {os.environ['HF_HOME']}")

pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    revision=MODEL_REVISION,
    dtype=torch.bfloat16,
    device_map="auto"
)

print(f"Model loaded successfully")

def handler(event):
    """
    TinyLlama inference handler.

    Input:
    {
        "prompt": "Your text here"
    }
    """
    input_data = event['input']
    user_prompt = input_data.get('prompt', 'Hello!')

    print(f"Generating response for: {user_prompt}")

    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": user_prompt}
        ]

        formatted_prompt = pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = pipe(
            formatted_prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )

        return {"generated_text": outputs[0]["generated_text"]}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
