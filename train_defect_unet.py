#!/usr/bin/env python3
"""Train one shared multiclass U-Net on all part-wise masked crops."""
import argparse, random, json, csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

NAMES = ["background", "dent", "scratch", "broken"]
COLORS = np.array([[0,0,0], [255,0,0], [0,255,0], [0,0,255]], dtype=np.uint8)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class PartCropDataset(Dataset):
    def __init__(self, rows, size):
        self.rows, self.size = rows, size
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        image_path, mask_path, _ = self.rows[i]
        im = Image.open(image_path).convert("RGB").resize((self.size,self.size), Image.Resampling.BILINEAR)
        ma = Image.open(mask_path).convert("L").resize((self.size,self.size), Image.Resampling.NEAREST)
        x = np.asarray(im, dtype=np.float32) / 255.0
        y = np.asarray(ma, dtype=np.uint8)
        x = torch.from_numpy(x.transpose(2,0,1).copy())
        y = torch.from_numpy(y.astype(np.int64).copy())
        return x, y

def conv_block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                         nn.Conv2d(b,b,3,padding=1,bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
class UNet(nn.Module):
    def __init__(self, n_classes=4, base=32):
        super().__init__()
        self.e1=conv_block(3,base); self.e2=conv_block(base,base*2)
        self.e3=conv_block(base*2,base*4); self.e4=conv_block(base*4,base*8)
        self.pool=nn.MaxPool2d(2)
        self.center=conv_block(base*8,base*16)
        self.u4=nn.ConvTranspose2d(base*16,base*8,2,2); self.d4=conv_block(base*16,base*8)
        self.u3=nn.ConvTranspose2d(base*8,base*4,2,2); self.d3=conv_block(base*8,base*4)
        self.u2=nn.ConvTranspose2d(base*4,base*2,2,2); self.d2=conv_block(base*4,base*2)
        self.u1=nn.ConvTranspose2d(base*2,base,2,2); self.d1=conv_block(base*2,base)
        self.out=nn.Conv2d(base,n_classes,1)
    def forward(self,x):
        a=self.e1(x); b=self.e2(self.pool(a)); c=self.e3(self.pool(b)); d=self.e4(self.pool(c))
        z=self.center(self.pool(d))
        z=self.d4(torch.cat([self.u4(z),d],1))
        z=self.d3(torch.cat([self.u3(z),c],1))
        z=self.d2(torch.cat([self.u2(z),b],1))
        z=self.d1(torch.cat([self.u1(z),a],1))
        return self.out(z)

def dice_loss(logits, target, n=4):
    probs=torch.softmax(logits,1)
    one=F.one_hot(target,n).permute(0,3,1,2).float()
    losses=[]
    for c in range(1,n):
        p=probs[:,c]; t=one[:,c]
        losses.append(1-(2*(p*t).sum((1,2))+1)/(p.sum((1,2))+t.sum((1,2))+1))
    return torch.stack(losses).mean()

def metrics(model, loader, device):
    model.eval(); inter=np.zeros(4); union=np.zeros(4)
    with torch.no_grad():
        for x,y in loader:
            pred=model(x.to(device)).argmax(1).cpu().numpy()
            gt=y.numpy()
            for c in range(1,4):
                inter[c]+=np.logical_and(pred==c,gt==c).sum()
                union[c]+=np.logical_or(pred==c,gt==c).sum()
    return {NAMES[c]: (float(inter[c]/union[c]) if union[c] else None) for c in range(1,4)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True,help="part_wise_dataset directory")
    ap.add_argument("--out",required=True)
    ap.add_argument("--epochs",type=int,default=60)
    ap.add_argument("--batch-size",type=int,default=4)
    ap.add_argument("--size",type=int,default=256)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--base-channels",type=int,default=32)
    args=ap.parse_args()
    seed_all(args.seed)
    root=Path(args.dataset); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    # Group split by original source image stem to prevent crop leakage.
    rows=[]
    for part_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ip in sorted((part_dir/"images").glob("*.jpg")):
            mp=part_dir/"masks"/(ip.stem+".png")
            if mp.exists():
                source=ip.stem.split("__")[0]
                rows.append((str(ip),str(mp),source))
    if len(rows)<3:
        raise SystemExit(f"Need at least 3 part crops with masks; found {len(rows)}")
    groups=defaultdict(list)
    for r in rows: groups[r[2]].append(r)
    keys=sorted(groups); random.shuffle(keys)
    n=len(keys)
    n_test=max(1,round(n*.15)); n_val=max(1,round(n*.15))
    if n_test+n_val>=n: n_test=1; n_val=1
    test_keys=set(keys[:n_test]); val_keys=set(keys[n_test:n_test+n_val])
    train_keys=set(keys[n_test+n_val:])
    train_rows=[r for r in rows if r[2] in train_keys]
    val_rows=[r for r in rows if r[2] in val_keys]
    test_rows=[r for r in rows if r[2] in test_keys]
    if not train_rows: raise SystemExit("No training crops after group split.")

    print(f"Crops: total={len(rows)}, train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")
    print(f"Unique source images: {len(keys)}; split is by source image.")
    train_ds=PartCropDataset(train_rows,args.size); val_ds=PartCropDataset(val_rows,args.size)
    test_ds=PartCropDataset(test_rows,args.size)
    train_dl=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True)
    val_dl=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    test_dl=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=UNet(4,args.base_channels).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    # Slightly reduce background dominance; class weights are applied to pixel CE.
    weights=torch.tensor([0.25,1.0,1.0,1.0],device=device)
    criterion=nn.CrossEntropyLoss(weight=weights)
    scaler=torch.amp.GradScaler("cuda",enabled=device=="cuda")
    best=float("inf")
    for epoch in range(1,args.epochs+1):
        model.train(); running=0
        for x,y in train_dl:
            x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",enabled=device=="cuda"):
                logits=model(x); loss=criterion(logits,y)+0.5*dice_loss(logits,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            running+=loss.item()
        val_loss=0; model.eval()
        with torch.no_grad():
            for x,y in val_dl:
                x=x.to(device); y=y.to(device)
                logits=model(x); val_loss+=(criterion(logits,y)+.5*dice_loss(logits,y)).item()
        val_loss/=max(1,len(val_dl)); train_loss=running/max(1,len(train_dl))
        print(f"Epoch {epoch:03d}/{args.epochs} train={train_loss:.4f} val={val_loss:.4f}")
        if val_loss<best:
            best=val_loss
            torch.save({"model_state_dict":model.state_dict(),"num_classes":4,
                        "class_names":NAMES,"input_size":args.size,
                        "base_channels":args.base_channels},out/"best_unet.pth")
    checkpoint=torch.load(out/"best_unet.pth",map_location=device,weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    result={"test_iou":metrics(model,test_dl,device),"num_crops":len(rows),
            "source_images":len(keys),"train_crops":len(train_rows),
            "val_crops":len(val_rows),"test_crops":len(test_rows),
            "class_names":NAMES}
    (out/"test_metrics.json").write_text(json.dumps(result,indent=2))
    print("[TEST]",json.dumps(result,indent=2))
    print("[SAVED]",out/"best_unet.pth")

if __name__=="__main__": main()
