import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

path = "F:\Documents\etudie\courses\CS8103_research\cat.png"
# path = "D:\\Desktop\\GT-Grad\\CS6476 Computer Vision\\1.Introduction&ImageProcessing\\ps1-all\\ps1-input0.png"

image = preprocess(Image.open(path)).unsqueeze(0).to(device) 
text = clip.tokenize([ "a cat", "a tree", "a raw shrimp", "a cooked shrimp"]).to(device) 

with torch.no_grad():
    # image_features = model.encode_image(image) # 将图片进行编码
    # text_features = model.encode_text(text)    # 将文本进行编码
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  