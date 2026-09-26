import torch, time
from transformers import AutoModelForCausalLM

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    r"G:\1A岩爆预测\sundial-rockburst\models\sundial-base-128m",
    trust_remote_code=True, local_files_only=True, torch_dtype=torch.float32).to("cuda").eval()
model.requires_grad_(False)
print("LOAD_OK %.1fs" % (time.time() - t0))
print("PARAMS", sum(p.numel() for p in model.parameters()))
x = torch.randn(1, 16, device="cuda")
with torch.inference_mode():
    y = model.generate(x, max_new_tokens=4, num_samples=1)
print("GEN_SHAPE", tuple(y.shape))
print("VRAM_GB %.2f" % (torch.cuda.memory_allocated() / 1e9))
print("DONE")
