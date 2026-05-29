#!/usr/bin/env python3
"""
qat_hw_exact.py — Hardware-EXACT QAT for the MNIST ViT.

The previous QAT (mnist_poc.QATMNISTViT) only *partially* simulated the int8
hardware: it used float patch_embed/classifier, real softmax, real-sqrt
LayerNorm, and a wrong attention scale (/32 instead of the hardware /4).
Result: the trained model (76%) disagreed with the bit-exact int8 pipeline
(~42%) on ~60% of images.

This module runs the EXACT integer pipeline of golden_model / the VHDL in the
forward pass (int8 GEMM with sat8(acc>>7), LUT softmax, LUT GELU, saturating
LayerNorm, int8 patch_embed and classifier, attention >>9), with a
straight-through estimator (STE) so gradients flow through smooth surrogates.

Train:   python qat_hw_exact.py train --epochs 30
Verify:  python qat_hw_exact.py verify
Export:  copies weights into the QAT checkpoint format, then run
         `python mnist_poc.py export_qat` as usual.
"""
from __future__ import annotations
import argparse, math, warnings
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ── dims (match mnist_poc / basys3_top) ──────────────────────────────────────
PATCH=7; SEQ=16; D=32; DFF=64; NCLS=10; Q=128.0
LOG_SQRT_HD=2                     # attention shift = (DATA_WIDTH-1)+LOG_SQRT_HD = 9
SM_DEPTH=256; SM_XMIN=-10.0
LN_HEADROOM=2                     # extra >> on LayerNorm output (must match golden + VHDL)

# ── straight-through estimator helper ────────────────────────────────────────
def ste(hard, soft):
    """Forward = hard (exact int op), backward flows to soft (differentiable)."""
    return soft + (hard - soft).detach()

def q8(x):
    """Round+clamp a float-Q1.7 tensor to int8 *value* scale (integer-valued)."""
    xi = x * Q
    return ste(xi.round().clamp(-128, 127), xi)

def sat8_shift7(acc):
    """sat8(acc >> 7): arithmetic floor shift then clamp, STE to acc/128."""
    hard = torch.floor(acc / Q).clamp(-128, 127)
    return ste(hard, acc / Q)

# ── LUTs as tensors (identical construction to mnist_poc / golden) ───────────
def _gelu_lut():
    s=0.7978845608028654; t=[]
    for i in range(256):
        xr=(i if i<128 else i-256)/128.0
        y=0.5*xr*(1.0+math.tanh(s*(xr+0.044715*xr**3)))
        t.append(max(-128,min(127,int(y*128.0))))   # truncate (matches HW)
    return torch.tensor(t,dtype=torch.float32)
GELU_LUT=_gelu_lut()

def _exp_lut():
    t=[]
    for i in range(SM_DEPTH):
        xr=SM_XMIN+i*(-SM_XMIN/(SM_DEPTH-1))
        t.append(int(math.floor(math.exp(xr)*(1<<16)+0.5)))
    return torch.tensor(t,dtype=torch.float32)
EXP_LUT=_exp_lut()

def gelu_hw(x_q8):
    """x_q8: int8-valued tensor. LUT lookup; STE surrogate = real GELU."""
    idx=(x_q8.round().clamp(-128,127)).long()%256
    hard=GELU_LUT.to(x_q8.device)[idx]
    soft=F.gelu(x_q8/Q)*Q
    return ste(hard, soft)

def softmax_hw(scores_q8):
    """scores_q8: [...,SEQ] int8-valued. Exact golden int softmax; STE = real softmax."""
    s=scores_q8.round().clamp(-128,127)
    row_max=s.max(dim=-1,keepdim=True).values
    diff=(s-row_max).clamp(min=-128,max=0)                    # <=0
    mag=(-diff).clamp(0,128)
    scaled=torch.floor(mag*(SM_DEPTH-1)/(10*128)).clamp(0,SM_DEPTH-1)
    idx=(SM_DEPTH-1-scaled).long().clamp(0,SM_DEPTH-1)
    expv=torch.floor(EXP_LUT.to(s.device)[idx]/512.0).clamp(max=127)
    expv=torch.where(diff>=0, torch.full_like(expv,127.0), expv)  # diff==0 -> 127
    denom=expv.sum(dim=-1,keepdim=True)
    hard=torch.floor(expv*Q/denom.clamp(min=1)).clamp(-128,127)
    soft=F.softmax(scores_q8/4.0,dim=-1)*Q                    # surrogate (hw scale /4)
    return ste(hard, soft)

def layernorm_hw(x_q8):
    """x_q8: [...,D] int8-valued. EXACT LOD-shift LayerNorm matching layernorm.vhd:
       mean=sum>>5, var=(ssq>>5)-mean^2, lod=floor(log2 var), shift=(lod+1)//2,
       out=sat8((x-mean) << (7-shift)).  STE surrogate = real-sqrt standardize."""
    xi=x_q8.round()
    mean=torch.floor(xi.sum(dim=-1,keepdim=True)/32.0)        # >>5 (D=32)
    ssq=(xi*xi).sum(dim=-1,keepdim=True)
    var=(torch.floor(ssq/32.0)-mean*mean).clamp(min=0)
    lod=torch.where(var>=1, torch.floor(torch.log2(var+1e-9)), torch.zeros_like(var))
    shift=torch.floor((lod+1)/2.0)
    diff=xi-mean
    out=diff*torch.pow(2.0, (7.0-LN_HEADROOM)-shift)          # << (7-HR-shift): extra headroom
    hard=torch.floor(out).clamp(-128,127)
    mean_f=x_q8.mean(dim=-1,keepdim=True)
    var_f=x_q8.var(dim=-1,unbiased=False,keepdim=True)
    soft=(x_q8-mean_f)/torch.sqrt(var_f+1e-3)*Q               # smooth surrogate, same scale
    return ste(hard, soft)

# ── the HW-exact model ───────────────────────────────────────────────────────
class HWExactViT(nn.Module):
    def __init__(self):
        super().__init__()
        # float Q1.7 weights (quantized in forward); init small
        self.ppw=nn.Parameter(torch.empty(D,PATCH*PATCH));  nn.init.xavier_uniform_(self.ppw)
        self.ppb=nn.Parameter(torch.zeros(D))
        self.pos=nn.Parameter(torch.zeros(SEQ,D))
        self.WQ=nn.Parameter(torch.empty(D,D)); self.WK=nn.Parameter(torch.empty(D,D))
        self.WV=nn.Parameter(torch.empty(D,D)); self.WO=nn.Parameter(torch.empty(D,D))
        for w in (self.WQ,self.WK,self.WV,self.WO): nn.init.xavier_uniform_(w)
        self.W1=nn.Parameter(torch.empty(DFF,D)); nn.init.xavier_uniform_(self.W1); self.b1=nn.Parameter(torch.zeros(DFF))
        self.W2=nn.Parameter(torch.empty(D,DFF)); nn.init.xavier_uniform_(self.W2); self.b2=nn.Parameter(torch.zeros(D))
        self.cw=nn.Parameter(torch.empty(NCLS,D)); nn.init.xavier_uniform_(self.cw); self.cb=nn.Parameter(torch.zeros(NCLS))

    def gemm(self, A, Wf, bf=None):
        """A:[...,K] int8-valued; Wf:[N,K] float weight; bf:[N] float bias. -> [...,N] int8."""
        W=q8(Wf)
        acc=A@W.t()
        if bf is not None: acc=acc + q8(bf)*Q
        return sat8_shift7(acc)

    def forward(self, patches):
        """patches: [B, SEQ, 49] int8-valued normalized pixels."""
        B=patches.shape[0]
        # patch embed: [B,SEQ,49]@[49,D] + pos, sat8
        x=self.gemm(patches, self.ppw, self.ppb)              # [B,SEQ,D] int8
        x=ste((x+q8(self.pos)).round().clamp(-128,127), x+q8(self.pos))
        # attention
        Qm=self.gemm(x,self.WQ); Km=self.gemm(x,self.WK); Vm=self.gemm(x,self.WV)
        acc=Qm@Km.transpose(1,2)                              # [B,SEQ,SEQ] integer
        scores=ste(torch.floor(acc/512.0).clamp(-128,127), acc/512.0)
        probs=softmax_hw(scores)                              # [B,SEQ,SEQ] int8 (0..127)
        ctx=sat8_shift7(probs@Vm)                             # [B,SEQ,D] int8
        mha=self.gemm(ctx,self.WO)
        y1=layernorm_hw(ste((x+mha).round().clamp(-128,127), x+mha))
        # ffn
        h=self.gemm(y1,self.W1,self.b1)
        h=gelu_hw(h)
        ffn=self.gemm(h,self.W2,self.b2)
        y2=layernorm_hw(ste((y1+ffn).round().clamp(-128,127), y1+ffn))
        # GAP (÷SEQ = >>LOG_SL) + classifier (int8)
        gsum=y2.sum(dim=1)
        gap=ste(torch.floor(gsum/SEQ).clamp(-128,127), gsum/SEQ)
        logits=self.gemm(gap, self.cw, self.cb)               # [B,NCLS] int8
        return logits

    def load_init(self, qat_ckpt="mnist_vit_qat.pth", float_ckpt="mnist_vit.pth"):
        import mnist_poc as mp
        q=mp.QATMNISTViT(); q.load_state_dict(torch.load(qat_ckpt,map_location="cpu",weights_only=True))
        fm=mp.MNISTViT(); fm.load_state_dict(torch.load(float_ckpt,map_location="cpu",weights_only=True))
        with torch.no_grad():
            self.ppw.copy_(fm.patch_embed.proj.weight.data.reshape(D,PATCH*PATCH))
            self.ppb.copy_(fm.patch_embed.proj.bias.data)
            self.pos.copy_(q.pos_embed.data[0])
            e=q.encoder[0]
            self.WQ.copy_(e.attn.q.weight.data); self.WK.copy_(e.attn.k.weight.data)
            self.WV.copy_(e.attn.v.weight.data); self.WO.copy_(e.attn.o.weight.data)
            self.W1.copy_(e.ffn.fc1.weight.data); self.b1.copy_(e.ffn.fc1.bias.data)
            self.W2.copy_(e.ffn.fc2.weight.data); self.b2.copy_(e.ffn.fc2.bias.data)
            self.cw.copy_(fm.classifier.weight.data); self.cb.copy_(fm.classifier.bias.data)


# ── data: MNIST -> normalized int8 patches [N, SEQ, 49] ──────────────────────
def _norm_px(p): return max(-128,min(127,int(round((p/255.0-0.1307)/0.3081*128.0))))
def _pix_addr(p,k): return (p//4*PATCH+k//PATCH)*28 + (p%4*PATCH+k%PATCH)

def load_patches(split, count=None):
    base=Path("./data/MNIST/raw")
    img=("train-images-idx3-ubyte" if split=="train" else "t10k-images-idx3-ubyte")
    lab=("train-labels-idx1-ubyte" if split=="train" else "t10k-labels-idx1-ubyte")
    with (base/img).open("rb") as f:
        import struct; _,n,_,_=struct.unpack(">IIII",f.read(16))
        if count: n=min(n,count)
        raw=f.read(n*784)
    with (base/lab).open("rb") as f:
        f.read(8); labels=list(f.read(n))
    # build [n, SEQ, 49] normalized int8
    import numpy as np
    arr=np.frombuffer(raw,dtype=np.uint8).reshape(n,784).astype(np.int64)
    # normalization LUT
    lut=np.array([_norm_px(v) for v in range(256)],dtype=np.int64)
    arr=lut[arr]
    # patch gather index [SEQ,49]
    gidx=np.array([[_pix_addr(p,k) for k in range(49)] for p in range(SEQ)])
    pat=arr[:,gidx]   # [n, SEQ, 49]
    return torch.tensor(pat,dtype=torch.float32), torch.tensor(labels,dtype=torch.long)


def evaluate(model, X, y, bs=500):
    model.eval(); correct=0
    with torch.no_grad():
        for i in range(0,len(X),bs):
            logits=model(X[i:i+bs])
            correct+=(logits.argmax(1)==y[i:i+bs]).sum().item()
    return correct/len(X)


def train(epochs=30, lr=2e-3, bs=256):
    Xtr,ytr=load_patches("train"); Xte,yte=load_patches("test")
    print(f"train {len(Xtr)}  test {len(Xte)}")
    m=HWExactViT()
    try:
        m.load_init(); print("initialized from existing checkpoints")
    except Exception as e:
        print("fresh init:",e)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    best=0.0
    n=len(Xtr)
    for ep in range(1,epochs+1):
        m.train(); perm=torch.randperm(n); tot=0.0
        for i in range(0,n,bs):
            idx=perm[i:i+bs]; xb=Xtr[idx]; yb=ytr[idx]
            opt.zero_grad()
            # logits are int8-scale (+-128); divide for a sane CE temperature
            loss=F.cross_entropy(m(xb)/32.0,yb)
            loss.backward(); opt.step(); tot+=loss.item()*len(idx)
        sch.step()
        acc=evaluate(m,Xte,yte)
        if acc>best: best=acc; torch.save(m.state_dict(),"hw_exact.pth")
        print(f"ep {ep:2d}/{epochs} | loss {tot/n:.4f} | test_acc {acc:.4f} | best {best:.4f}",flush=True)
    print(f"BEST {best:.4f} -> hw_exact.pth")


def verify():
    Xte,yte=load_patches("test")
    m=HWExactViT(); m.load_state_dict(torch.load("hw_exact.pth",map_location="cpu")); m.eval()
    print(f"HW-exact model test accuracy: {evaluate(m,Xte,yte):.4f}")


if __name__=="__main__":
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd")
    t=sub.add_parser("train"); t.add_argument("--epochs",type=int,default=30); t.add_argument("--lr",type=float,default=2e-3)
    sub.add_parser("verify")
    a=ap.parse_args()
    if a.cmd=="train": train(a.epochs,a.lr)
    elif a.cmd=="verify": verify()
    else: ap.print_help()
