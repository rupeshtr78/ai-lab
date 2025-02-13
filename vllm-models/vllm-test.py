from vllm.platforms import current_platform
import torch
from typing import Optional


device = Optional[torch.device]

print(device)
print(current_platform.device_type)
