from vllm import LLM, SamplingParams

model_path = "/data/hf-cache/huggingface/hub/models--mistralai--Mistral-Small-24B-Instruct-2501"
model = LLM(model = model_path)

prompts = [
    "Once upon a time,",
    "In a galaxy far, far away,",
    "The quick brown fox",
    "The meaning of life is",
    "The president of the United States",
    ]

sampling_params = SamplingParams(temperature=0.7, top_k=40)

outputs = model.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")
