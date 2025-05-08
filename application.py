import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import concurrent.futures

device = torch.device("cpu")


def load_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    
    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return tokenizer, model.to(device)

tokenizer1, model1 = load_model("Gensyn/Qwen2.5-0.5B-Instruct")
tokenizer2, model2 = load_model("tiiuae/falcon-rw-1b")
tokenizer3, model3 = load_model("microsoft/phi-1_5")

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=100,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def multi_agent_chat(user_input):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(generate_response, model1, tokenizer1, user_input),
            executor.submit(generate_response, model2, tokenizer2, user_input),
            executor.submit(generate_response, model3, tokenizer3, user_input)
        ]
        results = [f.result() for f in futures]
    return results


interface = gr.Interface(
    fn=multi_agent_chat,
    inputs=gr.Textbox(lines=2, placeholder="Ask something..."),
    outputs=[
        gr.Textbox(label="Agent 1 (Gensyn/Qwen2.5-0.5B-Instruct)"),
        gr.Textbox(label="Agent 2 (tiiuae/falcon-rw-1b)"),
        gr.Textbox(label="Agent 3 (microsoft/phi-1_5)")
    ],
    title="3-Agent AI Chatbot",
    description="Three GPT-style agents respond to your input in parallel!"
)

interface.launch()

