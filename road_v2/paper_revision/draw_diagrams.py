"""Deterministic Arial architecture and concise data preparation figures."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
DS = OUT.parent.parent / "dataset_noto2024_v0"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42})
INK, BLUE, TEAL, AMBER = "#142b3b", "#326d9a", "#278b80", "#c68937"

def box(ax,x,y,w,h,text,edge=BLUE,fill="#edf4f9",fs=8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.006,rounding_size=0.012",fc=fill,ec=edge,lw=.85))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,color=INK,linespacing=1.4)

def arrow(ax, pts, color=INK):
    if len(pts)>2: ax.plot(*np.array(pts[:-1]).T,color=color,lw=.9)
    ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle="-|>",mutation_scale=9,lw=.9,color=color,shrinkA=0,shrinkB=1))

def canvas(w,h):
    fig=plt.figure(figsize=(w,h));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');return fig,ax

def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'): fig.savefig(OUT/f'{name}.{ext}',dpi=400,facecolor='white',bbox_inches='tight',pad_inches=.025)
    plt.close(fig)

def network():
    fig,ax=canvas(7.2,3.65)
    # One shared encoder with named F4/F3 outputs avoids duplicate long return wires.
    for kind,y,lab in [('pre',.76,'Pre-event'),('post',.53,'Post-event')]:
        a=fig.add_axes([.015,y-.072,.094,.144]);a.imshow(Image.open(DS/f'chips/{kind}/noto_00794.jpg'));a.axis('off')
        ax.text(.062,y+.094,lab,ha='center',fontsize=8)
        arrow(ax,[(.111,y),(.148,y)])
    box(ax,.155,.442,.155,.405,'Shared\nConvNeXt-T\nencoder',fs=9)
    box(ax,.365,.660,.275,.187,'Global features\nGAP + LN\nfpre, fpost, |fpost - fpre|',fs=8.3)
    arrow(ax,[(.316,.758),(.359,.758)])
    ax.text(.337,.800,'F4',ha='center',fontsize=7.5,color=BLUE)
    box(ax,.365,.435,.135,.137,'Shared\nprojection',fill='white',fs=8.5)
    box(ax,.530,.435,.130,.137,'Local\nmatching',edge='#7863a8',fill='#f6f2fb',fs=8.5)
    box(ax,.690,.435,.140,.137,'Road + global\npooling',edge=TEAL,fill='#ecf8f5',fs=8.5)
    arrow(ax,[(.316,.503),(.359,.503)])
    ax.text(.337,.546,'F3',ha='center',fontsize=7.5,color=BLUE)
    arrow(ax,[(.506,.503),(.524,.503)])
    arrow(ax,[(.666,.503),(.684,.503)])
    ax.text(.432,.392,'1 x 1 conv, L2 norm',ha='center',fontsize=7.1,color='#526575')
    ax.text(.595,.392,'3 x 3 soft alignment',ha='center',fontsize=7.1,color='#526575')
    box(ax,.690,.205,.140,.100,'Road mask',edge=TEAL,fill='white',fs=8.5)
    arrow(ax,[(.760,.311),(.760,.429)],color=TEAL)
    box(ax,.690,.690,.140,.135,'Concatenate',edge=AMBER,fill='#fff8ed',fs=8.5)
    arrow(ax,[(.646,.758),(.684,.758)])
    arrow(ax,[(.760,.578),(.760,.684)],color=TEAL)
    box(ax,.865,.710,.110,.097,'Fused head',edge=TEAL,fill='#ecf8f5',fs=8.0)
    arrow(ax,[(.836,.758),(.859,.758)])
    # Named input uses the same fpost defined in the global branch; no crossing wire.
    box(ax,.865,.588,.110,.097,'Post head',edge=TEAL,fill='#ecf8f5',fs=8.0)
    ax.text(.847,.614,'fpost',ha='right',va='center',fontsize=7.7,color=BLUE)
    arrow(ax,[(.817,.637),(.859,.637)])
    box(ax,.865,.205,.110,.140,'Mean of\nsigmoids',edge=INK,fill='white',fs=8.1)
    arrow(ax,[(.981,.758),(.993,.758),(.993,.275),(.981,.275)])
    arrow(ax,[(.920,.582),(.920,.351)])
    ax.text(.920,.165,'p(blocked)',ha='center',fontsize=8.2,color=INK)
    ax.text(.505,.915,'GLOBAL BRANCH',ha='center',fontsize=8.2,fontweight='bold',color=BLUE)
    ax.text(.595,.130,'LOCAL CHANGE BRANCH',ha='center',fontsize=8.2,fontweight='bold',color=TEAL)
    save(fig,'fig_network')

def workflow():
    fig,ax=canvas(7.2,2.9)
    xs=[.025,.275,.525,.775];w=.20
    for x,title in zip(xs,['OSM roads','GSI slope failures','GSI orthophotos','DEM deposits']):
        box(ax,x,.800,w,.130,title,edge=BLUE,fill='#edf4f9',fs=8.3)
    specs=[('1  Segment roads','50 m pieces'),('2  Select candidates','5 m buffer intersects slide'),('3  Verify visually','Blind review + recheck'),('4  Final dataset','2,750 segments\n671 blocked')]
    for x,(title,sub) in zip(xs,specs):
        box(ax,x,.260,w,.280,title+'\n\n'+sub,edge=TEAL,fill='#ecf8f5',fs=7.9)
    for a,b in zip(xs[:-1],xs[1:]):arrow(ax,[(a+w+.008,.400),(b-.008,.400)])
    for x in (.125,.375,.625):arrow(ax,[(x,.794),(x,.546)],color=BLUE)
    arrow(ax,[(.875,.794),(.875,.650),(.690,.650),(.690,.547)],color='#7863a8')
    ax.text(.802,.687,'Recheck selection',ha='center',fontsize=7.2,color='#7863a8')
    ax.text(.50,.10,'Road-centred patches  |  Verified labels  |  Masks  |  Spatial folds',ha='center',fontsize=8.3,color=INK)
    save(fig,'fig_dataprep')

if __name__=='__main__':
    network();workflow();print('Saved Arial diagrams (PNG, PDF, SVG).')
