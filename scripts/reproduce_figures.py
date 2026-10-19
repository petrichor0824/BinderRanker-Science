#!/usr/bin/env python3
"""Regenerate BinderRanker main Figures 1-4 from repository source data."""

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Polygon
from PIL import Image

REPOSITORY = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--source-data',
    type=Path,
    default=REPOSITORY / 'data' / 'source_data',
    help='directory containing the figure source-data TSV files',
)
parser.add_argument(
    '--benchmark-data',
    type=Path,
    default=REPOSITORY / 'data' / 'benchmark' / 'high_recall',
    help='directory containing macro_pruning_recall.tsv',
)
parser.add_argument(
    '--output-dir',
    type=Path,
    default=REPOSITORY / 'figures' / 'reproduced',
)
args = parser.parse_args()
SRC = args.source_data.resolve()
BENCHMARK = args.benchmark_data.resolve()
OUT = args.output_dir.resolve()
PAN = OUT / '_panels'
PAN.mkdir(parents=True, exist_ok=True)

# Publication palette (color-blind aware, restrained)
BLUE = '#0072B2'
ORANGE = '#D55E00'
AMBER = '#E69F00'
GREEN = '#009E73'
RED = '#C43C39'
PURPLE = '#7A5AA6'
GREY = '#6B7280'
LIGHT_GREY = '#E9EEF2'
DARK = '#18212B'
PALE_BLUE = '#EFF6FB'
PALE_ORANGE = '#FFF5EB'
PALE_PURPLE = '#F5F1FA'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8.3,
    'ytick.labelsize': 8.3,
    'legend.fontsize': 7.8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)


def save_panel(fig, stem):
    png = PAN / f'{stem}.png'
    pdf = PAN / f'{stem}.pdf'
    svg = PAN / f'{stem}.svg'
    fig.savefig(png, dpi=350, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    fig.savefig(svg, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return png


def combine_grid(paths, out_png, cols=2, pad=34, row_pad=None, bg='white'):
    imgs = [Image.open(p).convert('RGB') for p in paths]
    if row_pad is None:
        row_pad = pad
    widths = []
    heights = []
    rows = math.ceil(len(imgs)/cols)
    for r in range(rows):
        row = imgs[r*cols:(r+1)*cols]
        widths.append(sum(i.width for i in row) + pad*(len(row)-1))
        heights.append(max(i.height for i in row))
    W = max(widths) + 2*pad
    H = sum(heights) + (rows-1)*row_pad + 2*pad
    canvas = Image.new('RGB', (W,H), bg)
    y = pad
    idx=0
    for r in range(rows):
        row = imgs[r*cols:(r+1)*cols]
        roww = sum(i.width for i in row) + pad*(len(row)-1)
        x = (W-roww)//2
        for img in row:
            canvas.paste(img,(x, y+(heights[r]-img.height)//2))
            x += img.width + pad
            idx += 1
        y += heights[r] + row_pad
    canvas.save(out_png, dpi=(350,350))
    canvas.save(Path(out_png).with_suffix('.pdf'), 'PDF', resolution=350.0)


def rounded(ax, x,y,w,h, face='white', edge=DARK, lw=1.0, radius=0.05):
    p = FancyBboxPatch((x,y),w,h,
                       boxstyle=f'round,pad=0.012,rounding_size={radius}',
                       linewidth=lw, edgecolor=edge, facecolor=face)
    ax.add_patch(p)
    return p


def arrow(ax, x1,y1,x2,y2, lw=1.0):
    a = FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>', mutation_scale=10,
                        linewidth=lw, color=DARK, shrinkA=1.5, shrinkB=1.5)
    ax.add_patch(a)
    return a


def protein_icon(ax, cx, cy, scale=1.0, color=BLUE, alpha=0.9):
    # Abstract ribbon-like backbone, entirely vector and non-atomistic.
    t=np.linspace(0,2*np.pi,160)
    x=cx + scale*(0.22*np.sin(2.2*t)+0.10*np.sin(5.1*t))
    y=cy + scale*(0.16*np.cos(1.7*t)+0.08*np.sin(4.2*t))
    ax.plot(x,y,color=color,lw=2.0*scale,alpha=alpha,solid_capstyle='round')


def funnel_icon(ax, cx, cy, scale=1.0):
    pts=np.array([[-.22,.16],[.22,.16],[.08,-.02],[.04,-.19],[-.04,-.19],[-.08,-.02]])*scale
    pts[:,0]+=cx; pts[:,1]+=cy
    ax.add_patch(Polygon(pts, closed=True, facecolor='#E8EDF2', edgecolor=GREY, lw=1.0))
    for dx,dy in [(-.16,.28),(-.05,.32),(.08,.28),(.18,.34),(-.10,.40),(.03,.41)]:
        ax.add_patch(Circle((cx+dx*scale,cy+dy*scale),0.035*scale,facecolor='#8FA4B6',edgecolor='none'))
    for dx in [-.05,.05]:
        ax.add_patch(Circle((cx+dx*scale,cy-.30*scale),0.035*scale,facecolor=AMBER,edgecolor='none'))


# ===========================
# Figure 1: conceptual + algorithmic architecture
# ===========================
fig, ax = plt.subplots(figsize=(10.8,3.05))
ax.set_xlim(0,12); ax.set_ylim(0,3.05); ax.axis('off')
ax.text(0.02,2.92,'a',fontsize=13,fontweight='bold',va='top')
ax.text(0.42,2.92,'Overall workflow',fontsize=12,fontweight='bold',va='top',color=DARK)
boxes=[
    (0.20,1.20,1.75,1.14,'RFdiffusion\nbackbones','large pool',PALE_BLUE),
    (2.38,1.20,1.75,1.14,'BinderRanker','pre-sequence ranking',PALE_ORANGE),
    (4.56,1.20,1.75,1.14,'ProteinMPNN','sequence design',PALE_BLUE),
    (6.74,1.20,1.75,1.14,'AF2 / AF3','complex evaluation',PALE_BLUE),
    (8.92,1.20,1.75,1.14,'Experiment','small set',PALE_BLUE),
]
for x,y,w,h,title,sub,fc in boxes:
    rounded(ax,x,y,w,h,fc,lw=1.05,radius=.09)
    ax.text(x+w/2,y+h-.18,title,ha='center',va='top',fontsize=9.3,fontweight='bold',color=DARK,linespacing=1.0)
    ax.text(x+w/2,y+.13,sub,ha='center',va='bottom',fontsize=7.5,color=DARK)
for i in range(len(boxes)-1):
    arrow(ax,boxes[i][0]+boxes[i][2]+.04,1.77,boxes[i+1][0]-.04,1.77,lw=1.0)
# mini icons
protein_icon(ax,1.08,1.72,.85,BLUE)
funnel_icon(ax,3.26,1.68,.75)
protein_icon(ax,5.15,1.72,.55,BLUE)
for k,s in enumerate(['MKTAY…','GSLDV…','YNEQF…','VTRDG…']):
    ax.text(5.64,1.94-.16*k,s,fontsize=6.7,color=[BLUE,ORANGE,GREEN,PURPLE][k],fontweight='bold')
ax.add_patch(Ellipse((7.57,1.72),.62,.33,facecolor='#9CC8E2',edgecolor='none'))
ax.add_patch(Ellipse((7.88,1.72),.62,.33,facecolor='#E9AC83',edgecolor='none'))
ax.plot([7.72,7.76],[1.50,1.93],color=DARK,lw=1.0)
# experiment plate
rounded(ax,9.54,1.52,.52,.36,'white',GREY,lw=.8,radius=.025)
for r in range(3):
    for c in range(5):
        ax.add_patch(Circle((9.61+c*.095,1.60+r*.09),.024,facecolor=BLUE if (r,c) in [(1,3),(2,4)] else '#DDE9F1',edgecolor='none'))
# discard callout
rounded(ax,2.52,.35,1.49,.49,'#FFF8F2',ORANGE,lw=.9,radius=.06)
ax.text(3.265,.595,'remove low-priority\nbackbones early',ha='center',va='center',fontsize=7.6,color=DARK)
arrow(ax,3.265,.86,3.265,1.16,lw=.9)
p1=save_panel(fig,'Fig1a_workflow')

# Fig 1b
fig, ax = plt.subplots(figsize=(10.8,5.05))
ax.set_xlim(0,12); ax.set_ylim(0,5.05); ax.axis('off')
ax.text(.02,4.90,'b',fontsize=13,fontweight='bold',va='top')
ax.text(.42,4.90,'BinderRanker algorithm',fontsize=12,fontweight='bold',va='top',color=DARK)
ax.text(.42,4.57,'A coherence objective integrates complementary interface geometry before sequence design.',fontsize=8.1,color=GREY,va='top')
# Contact field block
rounded(ax,.22,2.45,1.92,1.25,PALE_BLUE,lw=1.0,radius=.10)
ax.text(1.18,3.46,'orientation-aware\ncontact field',ha='center',va='top',fontsize=9,fontweight='bold',color=DARK)
ax.text(1.18,2.74,r'$W_{ij}=G_{ij}O_{ij}B_{ij}$',ha='center',va='center',fontsize=8.1,color=BLUE)
ax.text(1.18,2.56,'distance / orientation / Cβ support',ha='center',va='center',fontsize=5.6,color=GREY)
# schematic residues
for xx,yy,cc in [(.86,2.94,BLUE),(1.51,3.07,ORANGE),(1.00,3.20,BLUE),(1.58,2.86,ORANGE)]:
    ax.add_patch(Circle((xx,yy),.045,facecolor=cc,edgecolor='none'))
ax.plot([.90,1.47],[2.94,3.07],ls='--',lw=.75,color='#90A4AE')
ax.plot([1.04,1.54],[3.20,2.86],ls='--',lw=.75,color='#90A4AE')
# Three components
component_y=[3.40,2.87,2.34]
labels=[('target coverage',r'$C_{cov}$'),('Cβ-supported contact',r'$C_{\beta}$'),('local microdomain fit',r'$C_{micro}$')]
for yy,(lab,sym) in zip(component_y,labels):
    rounded(ax,2.50,yy,2.78,.38,'#F7FBFE',BLUE,lw=.8,radius=.12)
    ax.text(2.68,yy+.19,lab,va='center',fontsize=7.9,fontweight='bold',color=DARK)
    ax.text(5.08,yy+.19,sym,va='center',ha='right',fontsize=8.2,color=BLUE)
    arrow(ax,2.16,3.06,2.45,yy+.19,lw=.85)
# normalization
rounded(ax,5.64,2.62,2.02,1.00,PALE_PURPLE,PURPLE,lw=.9,radius=.10)
ax.text(6.65,3.39,'robust target-wise\nnormalization',ha='center',va='top',fontsize=9,fontweight='bold',color=DARK)
ax.text(6.65,2.96,'5th-95th percentile',ha='center',va='center',fontsize=7.4,color=GREY)
ax.text(6.65,2.72,r'$\tilde{x}=\mathrm{clip}\!\left(\frac{x-Q_{0.05}}{Q_{0.95}-Q_{0.05}},0,1\right)$',ha='center',va='center',fontsize=7.1,color=DARK)
for yy in component_y:
    arrow(ax,5.32,yy+.19,5.58,3.12,lw=.75)
# E block
rounded(ax,8.10,2.60,2.55,1.05,BLUE,BLUE,lw=.9,radius=.11)
ax.text(9.375,3.42,'coherent-interface score',ha='center',va='top',fontsize=9,fontweight='bold',color='white')
ax.text(9.375,3.03,r'$E=(\tilde{C}_{cov}\,\tilde{C}_{\beta}\,\tilde{C}_{micro})^{1/3}$',ha='center',va='center',fontsize=8.7,color='white')
ax.text(9.375,2.72,'multiplicative integration',ha='center',va='center',fontsize=7.1,color='#DCECF7')
arrow(ax,7.70,3.12,8.04,3.12,lw=.9)
# microdomain equation inset
rounded(ax,.58,.77,4.75,1.05,'#FBFCFE','#A8B7C5',lw=.8,radius=.10)
ax.text(.82,1.60,'Local microdomain fit',fontsize=8.6,fontweight='bold',color=DARK,va='top')
ax.text(.82,1.34,'plane-normal alignment × center-separation fit × local planarity',fontsize=7.2,color=DARK,va='top')
ax.text(.82,1.00,r'$m_j=|n_{T,j}\!\cdot\!n_{B,j}|\,\exp[-(d_j-7.5)^2/(2\cdot3.0^2)]\,p_{T,j}p_{B,j}$',fontsize=6.6,color=GREY,va='top')
# morphology
rounded(ax,5.72,.80,2.22,.98,'#F3F6F8',GREY,lw=.8,radius=.10)
ax.text(6.83,1.56,'morphology prior  $M$',ha='center',va='top',fontsize=8.7,fontweight='bold',color=DARK)
ax.text(6.83,1.18,'target-adaptive\nline / plane / compact modes',ha='center',va='center',fontsize=7.2,color=DARK)
# final score
rounded(ax,8.35,.72,2.48,1.17,PALE_BLUE,BLUE,lw=1.0,radius=.10)
ax.text(9.59,1.65,'final ranking score',ha='center',va='top',fontsize=9,fontweight='bold',color=DARK)
ax.text(9.59,1.26,r'$S_{BR}=0.90E+0.10M$',ha='center',va='center',fontsize=10.1,color=BLUE)
ax.text(9.59,.94,'continuous priority for pruning',ha='center',va='center',fontsize=7.2,color=GREY)
arrow(ax,9.40,2.57,9.58,1.94,lw=.9)
arrow(ax,7.96,1.29,8.29,1.29,lw=.9)
ax.text(.70,.31,'The geometric mean makes the objective coherence-seeking: a weak interface dimension suppresses E rather than being fully offset by another strong dimension.',fontsize=7.7,color=DARK)
p2=save_panel(fig,'Fig1b_algorithm')

# Fig 1c
fig, ax = plt.subplots(figsize=(10.8,2.7))
ax.set_xlim(0,12); ax.set_ylim(0,2.7); ax.axis('off')
ax.text(.02,2.55,'c',fontsize=13,fontweight='bold',va='top')
ax.text(.42,2.55,'Study design',fontsize=12,fontweight='bold',va='top',color=DARK)
blocks=[
    (.30,.86,2.30,.92,'5 calibration targets','model formulation'),
    (3.10,.86,1.95,.92,'fixed score','no held-out tuning'),
    (5.55,.86,2.40,.92,'5 held-out targets','25,000 backbones'),
    (8.45,.86,1.90,.92,'AF2 labels','evaluation'),
]
for x,y,w,h,t,s in blocks:
    rounded(ax,x,y,w,h,'#F7FAFC',DARK,lw=.9,radius=.08)
    ax.text(x+w/2,y+h-.18,t,ha='center',va='top',fontsize=8.8,fontweight='bold',color=DARK)
    ax.text(x+w/2,y+.17,s,ha='center',va='bottom',fontsize=7.2,color=DARK)
for i in range(len(blocks)-1):
    arrow(ax,blocks[i][0]+blocks[i][2]+.04,1.32,blocks[i+1][0]-.04,1.32,lw=.9)
# target dots
for dx,dy,cc in [(-.45,.08,BLUE),(0,.12,GREEN),(.45,.06,AMBER),(-.24,-.12,PURPLE),(.26,-.14,ORANGE)]:
    ax.add_patch(Ellipse((1.45+dx,1.25+dy),.30,.13,facecolor=cc,edgecolor='none',alpha=.75))
for dx,dy,cc in [(-.45,.08,BLUE),(0,.12,GREEN),(.45,.06,AMBER),(-.24,-.12,PURPLE),(.26,-.14,ORANGE)]:
    ax.add_patch(Ellipse((6.75+dx,1.25+dy),.30,.13,facecolor=cc,edgecolor='none',alpha=.75))
# fixed score curve
xx=np.linspace(3.55,4.60,100); yy=.99+.38*np.exp(-((xx-4.08)/.26)**2)
ax.plot(xx,yy,color=BLUE,lw=1.5); ax.axvline(4.08,ymin=.34,ymax=.62,color=GREY,lw=.8,ls='--')
# AF2 scatter icon
sx=np.array([8.78,8.94,9.10,9.26,9.42,9.58]); sy=np.array([1.10,1.16,1.15,1.29,1.33,1.40])
ax.scatter(sx,sy,s=9,color=BLUE,alpha=.75); ax.plot([8.75,9.62],[1.08,1.43],color=BLUE,lw=1.1)
rounded(ax,.42,.15,9.78,.42,PALE_BLUE,BLUE,lw=.7,radius=.08)
ax.text(5.31,.36,'Primary question: how much of the raw backbone pool can be removed while retaining downstream-positive candidates?',ha='center',va='center',fontsize=7.8,color=DARK)
p3=save_panel(fig,'Fig1c_design')
combine_grid([p1,p2,p3],OUT/'Fig1_BinderRanker_algorithm_final.png',cols=1,pad=22,row_pad=18)

# ===========================
# Data loads for figures 2-4
# ===========================
pos_counts=pd.read_csv(SRC/'Fig2_source_positive_counts.tsv',sep='\t')
retr=pd.read_csv(SRC/'Fig2_source_predeclared_retrieval.tsv',sep='\t')
top20=pd.read_csv(SRC/'Fig2_source_target_top20.tsv',sep='\t')
macro_pr=pd.read_csv(BENCHMARK/'macro_pruning_recall.tsv',sep='\t')
ops=pd.read_csv(SRC/'Fig3_source_operating_regimes.tsv',sep='\t')
runtime=pd.read_csv(SRC/'Fig3_source_runtime.tsv',sep='\t')
cont=pd.read_csv(SRC/'Fig4_source_continuous_frontiers.tsv',sep='\t')
disc=pd.read_csv(SRC/'Fig4_source_discrete_frontiers.tsv',sep='\t')
fixed=pd.read_csv(SRC/'Fig4_source_fixed_recall.tsv',sep='\t')

# ===========================
# Figure 2
# ===========================
targets=['IL7RA','TrkA','H1','VEGFA','IL17A']
fig,ax=plt.subplots(figsize=(5.15,3.55))
xx=np.arange(len(targets)); width=.34
for j,(label,col) in enumerate([('medium',BLUE),('loose',ORANGE)]):
    vals=[int(pos_counts[(pos_counts.target_id==t)&(pos_counts.label_definition==label)].n_positive.iloc[0]) for t in targets]
    ax.bar(xx+(j-.5)*width,vals,width,color=col,label=label.capitalize())
ax.set_xticks(xx,targets); ax.set_ylabel('AF2-defined positives'); ax.legend(frameon=False,ncol=2,loc='upper right')
ax.text(-.12,1.03,'a',transform=ax.transAxes,fontsize=12,fontweight='bold')
ax.text(-.03,1.03,'Held-out positives',transform=ax.transAxes,fontsize=10,fontweight='bold')
clean_axes(ax); p2a=save_panel(fig,'Fig2a_counts_final')

fig,ax=plt.subplots(figsize=(5.15,3.55))
for label,col in [('medium',BLUE),('loose',ORANGE)]:
    d=retr[retr.label_definition==label].sort_values('budget_value')
    ax.plot(d.budget_value*100,d.macro_recall*100,marker='o',ms=4.2,lw=1.8,color=col,label=label.capitalize())
x=np.array([1,5,10,20]); ax.plot(x,x,ls='--',lw=1.0,color=GREY,label='Random')
ax.set_xticks([1,5,10,20]); ax.set_xlabel('Candidates retained (%)'); ax.set_ylabel('Macro recall (%)'); ax.set_ylim(0,80)
ax.legend(frameon=False,loc='upper left')
ax.text(-.12,1.03,'b',transform=ax.transAxes,fontsize=12,fontweight='bold')
ax.text(-.03,1.03,'Retrieval at fixed budgets',transform=ax.transAxes,fontsize=10,fontweight='bold')
clean_axes(ax); p2b=save_panel(fig,'Fig2b_retrieval_final')

fig,ax=plt.subplots(figsize=(7.15,3.65))
xx=np.arange(len(targets))
for label,col,marker in [('medium',BLUE,'o'),('loose',ORANGE,'s')]:
    vals=[float(top20[(top20.target_id==t)&(top20.label_definition==label)].recall.iloc[0])*100 for t in targets]
    ax.plot(xx,vals,marker=marker,ms=5,lw=1.7,color=col,label=label.capitalize())
ax.axhline(20,ls='--',lw=1.0,color=GREY,label='Random')
ax.set_xticks(xx,targets); ax.set_ylabel('Recall at 20% retained (%)'); ax.set_ylim(0,105)
ax.legend(frameon=False,ncol=3,loc='lower left')
ax.text(-.09,1.03,'c',transform=ax.transAxes,fontsize=12,fontweight='bold')
ax.text(0.00,1.03,'Target-level enrichment',transform=ax.transAxes,fontsize=10,fontweight='bold')
clean_axes(ax); p2c=save_panel(fig,'Fig2c_target_final')
# custom layout: top two side by side, bottom centered
im1,im2,im3=[Image.open(p).convert('RGB') for p in [p2a,p2b,p2c]]
pad=30; top_h=max(im1.height,im2.height); W=im1.width+im2.width+3*pad
scale=min(1.12,(W-2*pad)/im3.width); im3=im3.resize((int(im3.width*scale),int(im3.height*scale)))
H=top_h+im3.height+3*pad
can=Image.new('RGB',(W,H),'white')
can.paste(im1,(pad,pad+(top_h-im1.height)//2)); can.paste(im2,(2*pad+im1.width,pad+(top_h-im2.height)//2)); can.paste(im3,((W-im3.width)//2,2*pad+top_h))
can.save(OUT/'Fig2_heldout_enrichment_final.png',dpi=(350,350)); can.save(OUT/'Fig2_heldout_enrichment_final.pdf','PDF',resolution=350)

# ===========================
# Figure 3
# ===========================
def frontier(label,col,letter,title,stem):
    d=macro_pr[macro_pr.label_definition==label].sort_values('pruned_fraction')
    fig,ax=plt.subplots(figsize=(5.1,3.6))
    ax.fill_between(d.pruned_fraction*100,d.min_target_recall*100,d.max_target_recall*100,color=col,alpha=.12,lw=0)
    ax.plot(d.pruned_fraction*100,d.macro_recall*100,color=col,lw=2.0,label='BinderRanker macro')
    ax.plot([0,100],[100,0],ls='--',lw=1.0,color=GREY,label='Random')
    ax.set_xlim(0,100); ax.set_ylim(0,105); ax.set_xlabel('Backbones pruned (%)'); ax.set_ylabel('Recall (%)')
    ax.legend(frameon=False,loc='lower left')
    ax.text(-.12,1.03,letter,transform=ax.transAxes,fontsize=12,fontweight='bold'); ax.text(-.03,1.03,title,transform=ax.transAxes,fontsize=10,fontweight='bold')
    clean_axes(ax); return save_panel(fig,stem)
p3a=frontier('medium',BLUE,'a','Medium labels','Fig3a_medium_final')
p3b=frontier('loose',ORANGE,'b','Loose labels','Fig3b_loose_final')

fig,ax=plt.subplots(figsize=(5.1,3.6))
for label,col,marker in [('medium',BLUE,'o'),('loose',ORANGE,'s')]:
    d=ops[ops.label_definition==label].sort_values('pruned_fraction')
    y=d.macro_recall*100; low=(d.macro_recall-d.min_target_recall)*100; high=(d.max_target_recall-d.macro_recall)*100
    ax.errorbar(d.pruned_fraction*100,y,yerr=np.vstack([low,high]),color=col,marker=marker,ms=5,lw=1.5,capsize=3,label=label.capitalize())
    for _,r in d.iterrows():
        if r.pruned_fraction in (0.4,0.5):
            ax.annotate(f'{r.macro_recall*100:.1f}%',(r.pruned_fraction*100,r.macro_recall*100),xytext=(4,5),textcoords='offset points',fontsize=7.2,color=col)
ax.set_xticks([40,50,70]); ax.set_xlabel('Backbones pruned (%)'); ax.set_ylabel('Macro recall (%)'); ax.set_ylim(40,103); ax.legend(frameon=False,loc='lower left')
ax.text(-.12,1.03,'c',transform=ax.transAxes,fontsize=12,fontweight='bold'); ax.text(-.03,1.03,'Practical operating points',transform=ax.transAxes,fontsize=10,fontweight='bold')
clean_axes(ax); p3c=save_panel(fig,'Fig3c_ops_final')

fig,ax=plt.subplots(figsize=(5.1,3.6))
ax.plot(runtime.pruned_fraction*100,runtime.pooled_runtime_saved_fraction*100,color=BLUE,marker='o',ms=5,lw=1.8,label='Recorded AF2 burden')
ax.plot([30,75],[30,75],ls='--',lw=1.0,color=GREY,label='Proportional reference')
for _,r in runtime.iterrows():
    ax.annotate(f'{r.total_saved_af2_hours:.1f} h',(r.pruned_fraction*100,r.pooled_runtime_saved_fraction*100),xytext=(4,5),textcoords='offset points',fontsize=7.2,color=DARK)
ax.set_xlim(30,75); ax.set_ylim(30,75); ax.set_xlabel('Backbones pruned (%)'); ax.set_ylabel('AF2 processing burden reduced (%)'); ax.legend(frameon=False,loc='upper left')
ax.text(0.97,0.06,'BinderRanker: 15.5 min / 25,000 backbones\n1 CPU core; 0.037 s/backbone',transform=ax.transAxes,ha='right',va='bottom',fontsize=7.1,color=DARK,bbox=dict(boxstyle='round,pad=0.28',fc='white',ec='#B8C2CC',lw=.65))
ax.text(-.12,1.03,'d',transform=ax.transAxes,fontsize=12,fontweight='bold'); ax.text(-.03,1.03,'Downstream burden reduction',transform=ax.transAxes,fontsize=10,fontweight='bold')
clean_axes(ax); p3d=save_panel(fig,'Fig3d_runtime_final')
combine_grid([p3a,p3b,p3c,p3d],OUT/'Fig3_high_recall_pruning_final.png',cols=2,pad=24,row_pad=18)

# ===========================
# Figure 4
# ===========================
def basefront(label,letter,title,stem):
    fig,ax=plt.subplots(figsize=(5.1,3.65))
    styles=[('BinderRanker_v5',BLUE,'BinderRanker'),('Contact8',AMBER,'Contact8'),('Rg_massweighted',GREEN,'Rg'),('DSSP_helix_count',RED,'DSSP helix count')]
    for method,col,lab in styles[:1]+styles[2:3]:
        d=cont[(cont.label_definition==label)&(cont.method==method)].sort_values('pruned_fraction')
        ax.plot(d.pruned_fraction*100,d.macro_recall*100,lw=1.75,color=col,label=lab)
    for method,col,lab in [styles[1],styles[3]]:
        d=disc[(disc.label_definition==label)&(disc.method==method)].sort_values('pruned_fraction')
        ax.plot(d.pruned_fraction*100,d.macro_expected_recall*100,lw=1.55,color=col,label=lab)
    ax.plot([0,100],[100,0],ls='--',lw=.95,color=GREY,label='Random')
    ax.set_xlim(0,100); ax.set_ylim(0,105); ax.set_xlabel('Backbones pruned (%)'); ax.set_ylabel('Macro recall (%)')
    ax.legend(frameon=False,loc='lower left')
    ax.text(-.12,1.03,letter,transform=ax.transAxes,fontsize=12,fontweight='bold'); ax.text(-.03,1.03,title,transform=ax.transAxes,fontsize=10,fontweight='bold')
    clean_axes(ax); return save_panel(fig,stem)
p4a=basefront('loose','a','Loose labels','Fig4a_loose_final')
p4b=basefront('medium','b','Medium labels','Fig4b_medium_final')

def fixed_panel(label,letter,title,stem):
    fig,ax=plt.subplots(figsize=(5.1,3.65))
    thresholds=[.8,.9,.95,.99,1.0]; xpos=np.arange(len(thresholds))
    styles=[('BinderRanker_v5',BLUE,'BinderRanker'),('Contact8',AMBER,'Contact8'),('Rg_massweighted',GREEN,'Rg'),('DSSP_helix_count',RED,'DSSP helix count')]
    for method,col,lab in styles:
        d=fixed[(fixed.label_definition==label)&(fixed.method==method)].set_index('target_recall').loc[thresholds].reset_index()
        ax.plot(xpos,d.mean_pruned_fraction*100,marker='o',ms=4,lw=1.65,color=col,label=lab)
    ax.set_xticks(xpos,['80','90','95','99','100']); ax.set_xlabel('Required recall within each target (%)'); ax.set_ylabel('Mean pool pruned (%)'); ax.set_ylim(0,80)
    ax.legend(frameon=False,loc='upper right')
    ax.text(-.12,1.03,letter,transform=ax.transAxes,fontsize=12,fontweight='bold'); ax.text(-.03,1.03,title,transform=ax.transAxes,fontsize=10,fontweight='bold')
    clean_axes(ax); return save_panel(fig,stem)
p4c=fixed_panel('loose','c','High-recall efficiency, Loose','Fig4c_loose_final')
p4d=fixed_panel('medium','d','High-recall efficiency, Medium','Fig4d_medium_final')
combine_grid([p4a,p4b,p4c,p4d],OUT/'Fig4_simple_baselines_final.png',cols=2,pad=24,row_pad=18)

print('Final figures written to',OUT)
