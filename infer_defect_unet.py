#!/usr/bin/env python3
"""Run trained shared U-Net on an image crop and save a named/color overlay."""
import argparse
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from pathlib import Path
from train_unet import UNet

COLORS=np.array([[0,0,0],[255,0,0],[0,255,0],[0,0,255]],dtype=np.uint8)
NAMES=["background","dent","scratch","broken"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True); ap.add_argument("--image",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    ck=torch.load(args.checkpoint,map_location=device,weights_only=False)
    size=int(ck["input_size"]); model=UNet(4,int(ck["base_channels"])).to(device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    im=Image.open(args.image).convert("RGB"); original=np.asarray(im)
    resized=im.resize((size,size),Image.Resampling.BILINEAR)
    x=np.asarray(resized,dtype=np.float32)/255.
    x=torch.from_numpy(x.transpose(2,0,1).copy()).unsqueeze(0).to(device)
    with torch.no_grad(): pred=model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
    pred=Image.fromarray(pred).resize((original.shape[1],original.shape[0]),Image.Resampling.NEAREST)
    pred=np.asarray(pred)
    overlay=original.copy()
    for c in (1,2,3):
        m=pred==c
        overlay[m]=(0.45*overlay[m]+0.55*COLORS[c]).astype(np.uint8)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(overlay).save(out)
    print("Saved:",out)
    for c in (1,2,3):
        print(f"{NAMES[c]} pixels:",int((pred==c).sum()))
if __name__=="__main__": main()
