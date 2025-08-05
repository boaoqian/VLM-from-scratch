from model.Config import Qwen3Config
from model.Qwen3 import Qwen3ModelForCausalLM
from transformers import AutoTokenizer



model_path = '/media/qba/Data/Project/DeepLearning/Model/Qwen3-0.6B'
model = Qwen3ModelForCausalLM.from_pretrained(model_path)
print(model)
tokenizer = AutoTokenizer.from_pretrained("/media/qba/Data/Project/DeepLearning/Model/Qwen3-0.6B/", trust_remote_code=True)
msg = {"role": "user", "content": "写一个计算鸡兔同笼的代码"} 
prompt = tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=True,enable_thinking=False)
print(prompt)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
model = model.to("cuda")
emb = model.model.embed_tokens(inputs.input_ids)
print(emb.shape)
import time 

t1 = time.time()
out1,kv = model.generate(inputs_embeds=emb, max_new_tokens=20, stop_tokens=[151645,151643],use_cache=True)
t2 = time.time()
print("Time taken for generation with cache:", t2 - t1)
print(tokenizer.decode(out1[0], skip_special_tokens=True))