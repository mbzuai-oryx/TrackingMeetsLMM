import os 
from llama.llama_adapter import LLaMA_adapter
import util.misc as misc
import torchvision.transforms as transforms
import util.extract_adapter_from_checkpoint as extract
from PIL import Image
import cv2
import torch
import llama
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
llama_dir = "/data1/DriveLM/llama_wts/"
llama_type = '7B'
llama_ckpt_dir = os.path.join(llama_dir, llama_type)
llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')
model = LLaMA_adapter(llama_ckpt_dir, llama_tokenzier_path)
misc.load_model(model, './saved_models/checkpoint-3.pth')
model.eval()
model.to(device)
print('model to device')

extract.save(model,"./saved_models/converted-checkpoint.pth",'BIAS') # Please end it with -llama_type.pth