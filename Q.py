import customtkinter as ctk
from llama_cpp import Llama
import threading

app = ctk.CTk()
app.geometry("600x500")
app.title("Henny's Qwen Tester 🥺✨")

# Just our main girl Qwen now! 💅
qwen_path = "/media/henry/Henry_s Backup/Users/Henry/Downloads/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
current_llm = None

def load_model_thread():
    global current_llm
    output_box.insert("end", "\nLoading Qwen 2.5... pls wait! 😭\n")
    
    # Taking off the chat format so we can inject the prompt raw! 🫣💅
    current_llm = Llama(model_path=qwen_path, n_ctx=2048, verbose=False)
    
    output_box.insert("end", "Loaded! Ready to spill tea! 🍵✨\n")
    load_btn.configure(state="disabled", text="Brain Loaded! 🧠")

def load_model():
    threading.Thread(target=load_model_thread, daemon=True).start()

def generate_text_thread(prompt):
    # We are writing the raw ChatML tags ourselves to force-feed her the personality! 😼✨
    raw_prompt = f"<|im_start|>system\nAlways list every steps on how to do something, never leave any detail out, and avoid being vague with your replies<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    res = current_llm(
        raw_prompt,
        max_tokens=1500,
        stop=["<|im_end|>", "<|im_start|>"], # Tell her exactly when to shut up! 💀
        echo=False
    )
    
    text = res['choices'][0]['text'].strip()
    output_box.insert("end", f"AI: {text}\n")
    gen_btn.configure(state="normal")

def generate_text():
    if not current_llm:
        output_box.insert("end", "Load the model first bestie! 🙄\n")
        return
    
    prompt = prompt_entry.get()
    output_box.insert("end", f"\nHenny: {prompt}\nAI is thinking... 💅\n")
    prompt_entry.delete(0, "end") # Clears the box for your next message! 😽
    
    gen_btn.configure(state="disabled") 
    threading.Thread(target=generate_text_thread, args=(prompt,), daemon=True).start()

# Sleek new UI! 🎀
load_btn = ctk.CTkButton(app, text="Load Qwen 🧠", command=load_model)
load_btn.pack(pady=20)

prompt_entry = ctk.CTkEntry(app, width=400, placeholder_text="Type your prompt here Henny! 😼")
prompt_entry.pack(pady=10)

gen_btn = ctk.CTkButton(app, text="Generate! ✨", command=generate_text)
gen_btn.pack(pady=10)

output_box = ctk.CTkTextbox(app, width=500, height=300)
output_box.pack(pady=20)

app.mainloop()
