"""Deterministic Arial architecture and concise data preparation figures."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.transforms import Bbox
from matplotlib.font_manager import FontProperties, findfont
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
    if len(pts)>2: ax.plot(*np.array(pts[:-1]).T,color=color,lw=.9,clip_on=False)
    ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle="-|>",mutation_scale=9,lw=.9,color=color,shrinkA=0,shrinkB=1,clip_on=False))

def canvas(w,h):
    fig=plt.figure(figsize=(w,h));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');return fig,ax

def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    # The full-canvas axes enlarge bbox_inches='tight' with empty margins. Crop
    # to the actual diagram artists, including the inset orthophotos and wires.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    bounds=[]
    for ax in fig.axes:
        for artist in [*ax.texts,*ax.patches,*ax.lines,*ax.images]:
            if artist.get_visible():
                bounds.append(artist.get_window_extent(renderer))
    content=Bbox.union(bounds).transformed(fig.dpi_scale_trans.inverted())
    crop=Bbox.from_extents(content.x0-.035,content.y0-.035,
                           content.x1+.035,content.y1+.035)
    for ext in ('png','pdf','svg'): fig.savefig(OUT/f'{name}.{ext}',dpi=400,facecolor='white',bbox_inches=crop)
    plt.close(fig)

def network():
    fig,ax=canvas(7.2,3.65)
    # One shared encoder with named F4/F3 outputs avoids duplicate long return wires.
    for kind,y,lab in [('pre',.76,'Pre-event'),('post',.53,'Post-event')]:
        a=fig.add_axes([.015,y-.072,.094,.144]);a.imshow(Image.open(DS/f'chips/{kind}/noto_00794.jpg'));a.axis('off')
        ax.text(.062,y+.094,lab,ha='center',fontsize=8)
        arrow(ax,[(.111,y),(.148,y)])
    box(ax,.155,.330,.155,.517,'Shared\nConvNeXt-T\nencoder',fs=9)
    box(ax,.365,.660,.275,.187,'Global features\nGAP + LN\nfpre, fpost, |fpost - fpre|',fs=8.3)
    arrow(ax,[(.316,.758),(.359,.758)])
    ax.text(.337,.800,'F4',ha='center',fontsize=7.5,color=BLUE)
    box(ax,.365,.330,.135,.137,'Shared\nprojection',fill='white',fs=8.5)
    box(ax,.530,.330,.130,.137,'Local\nmatching',edge='#7863a8',fill='#f6f2fb',fs=8.5)
    box(ax,.690,.330,.140,.137,'Road + global\npooling',edge=TEAL,fill='#ecf8f5',fs=8.5)
    arrow(ax,[(.316,.399),(.359,.399)])
    ax.text(.337,.442,'F3',ha='center',fontsize=7.5,color=BLUE)
    arrow(ax,[(.506,.399),(.524,.399)])
    arrow(ax,[(.666,.399),(.684,.399)])
    ax.text(.432,.285,'1 x 1 conv, L2 norm',ha='center',fontsize=7.1,color='#526575')
    ax.text(.595,.285,'3 x 3 soft alignment',ha='center',fontsize=7.1,color='#526575')
    box(ax,.690,.160,.140,.100,'Road mask',edge=TEAL,fill='white',fs=8.5)
    arrow(ax,[(.760,.266),(.760,.324)],color=TEAL)
    box(ax,.690,.560,.140,.115,'Concatenate',edge=AMBER,fill='#fff8ed',fs=8.5)
    arrow(ax,[(.646,.706),(.665,.706),(.665,.618),(.684,.618)])
    arrow(ax,[(.760,.473),(.760,.554)],color=TEAL)
    # Put the post head above the fused head so the global and local wires do not cross.
    box(ax,.865,.710,.110,.097,'Post head',edge=TEAL,fill='#ecf8f5',fs=8.0)
    ax.text(.751,.794,'fpost',ha='center',va='center',fontsize=7.7,color=BLUE)
    arrow(ax,[(.646,.758),(.859,.758)])
    box(ax,.865,.570,.110,.097,'Fused head',edge=TEAL,fill='#ecf8f5',fs=8.0)
    arrow(ax,[(.836,.618),(.859,.618)])
    box(ax,.865,.160,.110,.140,'Mean of\nsigmoids',edge=INK,fill='white',fs=8.1)
    arrow(ax,[(.981,.758),(1.026,.758),(1.026,.230),(.981,.230)])
    arrow(ax,[(.920,.564),(.920,.306)])
    ax.text(.920,.120,'p(blocked)',ha='center',fontsize=8.2,color=INK)
    ax.text(.505,.868,'GLOBAL BRANCH',ha='center',va='bottom',fontsize=8.2,fontweight='bold',color=BLUE)
    ax.text(.365,.506,'LOCAL CHANGE BRANCH',ha='left',fontsize=8.2,fontweight='bold',color=TEAL)
    save(fig,'fig_network')

def workflow():
    fig,ax=canvas(7.2,2.9)
    findfont(FontProperties(family='Arial'),fallback_to_default=False)
    xs=[.025,.275,.525,.775];w=.20
    for x,title in zip(xs,['OSM roads','GSI slope failures','GSI orthophotos','DEM deposits']):
        box(ax,x,.800,w,.130,title,edge=BLUE,fill='#edf4f9',fs=10.5)
    specs=[('1  Segment roads','50 m pieces'),('2  Select candidates','5 m buffer\nintersects slide'),('3  Verify visually','Blind review\n+ recheck'),('4  Final dataset','2,750 segments\n671 blocked')]
    for x,(title,sub) in zip(xs,specs):
        box(ax,x,.260,w,.280,'',edge=TEAL,fill='#ecf8f5')
        ax.text(x+w/2,.474,title,ha='center',va='center',fontsize=10.2,color=INK)
        ax.text(x+w/2,.352,sub,ha='center',va='center',fontsize=10.2,color=INK,linespacing=1.25)
    for a,b in zip(xs[:-1],xs[1:]):arrow(ax,[(a+w+.008,.400),(b-.008,.400)])
    for x in (.125,.375,.625):arrow(ax,[(x,.794),(x,.546)],color=BLUE)
    arrow(ax,[(.875,.794),(.875,.650),(.690,.650),(.690,.547)],color='#7863a8')
    ax.text(.785,.687,'Recheck selection',ha='center',fontsize=9.2,color='#7863a8')
    ax.text(.50,.10,'Road-centred patches  |  Verified labels  |  Masks  |  Spatial folds',ha='center',fontsize=10.2,color=INK)
    save(fig,'fig_dataprep')

if __name__=='__main__':
    network();workflow();print('Saved Arial diagrams (PNG, PDF, SVG).')
