from llama_cpp import Llama

llm = Llama(
    model_path="models/Llama-4-Scout-17B-16E-Instruct-UD-Q2_K_XL.gguf",
    n_gpu_layers=30,   # off-load ~¾ of the network
    n_ctx=2048,        # plenty for most probing work
    n_batch=32,        # safe batch size for GPU kernels
    logits_all=True    # needed by logit-lens & patching tools
)

messages = [
    {"role": "user", "content": "The capital of Germany is"}
]

out = llm.create_chat_completion(
    messages=messages,
    max_tokens=64,
    temperature=0
)
print("\n=== OUTPUT ========================")
print(out["choices"][0]["message"]["content"].strip())
print("=== END OF OUTPUT =================\n")