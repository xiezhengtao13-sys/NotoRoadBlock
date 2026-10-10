"""Fetch official GSI tiles and show verified labels and true out-of-fold scores."""
from pathlib import Path
import concurrent.futures
import io,json,math,os,shutil,subprocess
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'paper_revision/figures'
OUT.mkdir(parents=True,exist_ok=True)
CACHE=ROOT/'paper_revision/tile_cache'
CACHE.mkdir(exist_ok=True)
DS=ROOT/'dataset_noto2024_v0'
Z=17
LAT,LON=37.2641044,136.9172309
W_M,H_M=1200,800
LAYERS=['nendophoto2020','20240102noto_anamizu_0111do']

def px(lon,lat):
    lon=np.asarray(lon);lat=np.asarray(lat)
    return np.stack([(lon+180)/360*256*2**Z,(1-np.arcsinh(np.tan(np.radians(lat)))/np.pi)/2*256*2**Z],axis=-1)

gsd=math.cos(math.radians(LAT))*2*math.pi*6378137/(256*2**Z)
centre=px(LON,LAT);x0,y0=np.rint(centre-[W_M/gsd/2,H_M/gsd/2]).astype(int)
width,height=round(W_M/gsd),round(H_M/gsd)
tiles=[(x,y) for x in range(x0//256,(x0+width-1)//256+1) for y in range(y0//256,(y0+height-1)//256+1)]

def fetch(job):
    layer,x,y=job
    target=CACHE/f'{layer}_{Z}_{x}_{y}.png'
    if target.exists():return target
    url=f'https://cyberjapandata.gsi.go.jp/xyz/{layer}/{Z}/{x}/{y}.png'
    curl_bin=os.environ.get('CURL_BIN') or shutil.which('curl')
    if not curl_bin:raise RuntimeError('Install curl or set CURL_BIN to its executable path.')
    r=subprocess.run([curl_bin,'-f','-sS','--connect-timeout','10','--max-time','35','--retry','1',url],capture_output=True)
    if r.returncode:raise RuntimeError(f'Tile unavailable: {url}; '+r.stderr.decode(errors='replace')[:120])
    # Validate before saving; no blank placeholders are accepted.
    im=Image.open(io.BytesIO(r.stdout));assert im.size==(256,256)
    target.write_bytes(r.stdout);return target

jobs=[(lay,x,y) for lay in LAYERS for x,y in tiles]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    paths=list(pool.map(fetch,jobs))
print(f'Validated {len(paths)} GSI tiles',flush=True)
images=[]
for layer in LAYERS:
    canvas=Image.new('RGB',((max(x for x,y in tiles)-x0//256+1)*256,(max(y for x,y in tiles)-y0//256+1)*256))
    for x,y in tiles:canvas.paste(Image.open(CACHE/f'{layer}_{Z}_{x}_{y}.png').convert('RGB'),((x-x0//256)*256,(y-y0//256)*256))
    canvas=canvas.crop((x0%256,y0%256,x0%256+width,y0%256+height))
    canvas.save(OUT/f'landscape_{layer}.jpg',quality=96)
    images.append(canvas)

labels=pd.read_csv(DS/'labels_verified.csv').set_index('sample_id').label.to_dict()
scores=pd.read_csv(ROOT/'results_noto/oof_ft_convnext_siamese_v2_verified.csv').set_index('sample_id').score.to_dict()
features=json.loads((DS/'samples.geojson').read_text(encoding='utf-8'))['features']
segs=[]
for f in features:
    sid=f['properties']['sample_id'];g=px(*np.array(f['geometry']['coordinates']).T)-[x0,y0]
    if sid in scores and labels.get(sid,-1)>=0 and ((g[:,0]>=0)&(g[:,0]<width)&(g[:,1]>=0)&(g[:,1]<height)).any():
        segs.append((sid,g,labels[sid],scores[sid]))
assert any(y==1 for _,g,y,s in segs)
plt.rcParams.update({'font.family':'Arial','font.size':8})
fig,axes=plt.subplots(2,2,figsize=(7.2,5.65),facecolor='white')
titles=['(a) Pre-event | 2020','(b) Post-event | 11 January 2024','(c) Reference segment labels','(d) Road-segment classification scores']
for n,(ax,title) in enumerate(zip(axes.flat,titles)):
    ax.imshow(images[0 if n==0 else 1]);ax.set(xlim=(0,width),ylim=(height,0));ax.set_xticks([]);ax.set_yticks([])
    ax.set_title(title,loc='left',fontsize=8.5,fontweight='bold',pad=5)
    for spine in ax.spines.values():spine.set_visible(False)
    if n>=2:
        for sid,g,label,score in segs:
            color=('#ce273c' if label else '#20c3ac') if n==2 else plt.get_cmap('YlOrRd')(score)
            ax.plot(g[:,0],g[:,1],color=color,lw=2.1,solid_capstyle='round',path_effects=[pe.Stroke(linewidth=3.2,foreground='white'),pe.Normal()])
    # All panels share the same true scale and extent.
    bar=100/gsd
    ax.plot([width*.055,width*.055+bar],[height*.94]*2,color='white',lw=3)
    ax.text(width*.055+bar/2,height*.91,'100 m',color='white',ha='center',fontsize=7.5,path_effects=[pe.withStroke(linewidth=2,foreground='black')])
    ax.text(width*.96,height*.08,'N',color='white',ha='center',fontsize=8,path_effects=[pe.withStroke(linewidth=2,foreground='black')])
axes[1,0].legend(handles=[Line2D([0],[0],color='#ce273c',lw=3,label='Blocked'),Line2D([0],[0],color='#20c3ac',lw=3,label='Clear')],loc='upper left',fontsize=7,framealpha=.95)
fig.subplots_adjust(left=.005,right=.995,top=.96,bottom=.105,hspace=.16,wspace=.035)
cax=fig.add_axes([.57,.06,.33,.012]);sm=plt.cm.ScalarMappable(cmap='YlOrRd',norm=plt.Normalize(0,1));cb=fig.colorbar(sm,cax=cax,orientation='horizontal');cb.set_label('p(blocked) per road segment',fontsize=7,labelpad=2);cb.ax.tick_params(labelsize=7,length=2,pad=1)
fig.text(.01,.04,'Source: GSI orthophotos | OSM roads\n17 sampled segments; one score per 50 m segment',fontsize=7,color='#526575')
fig.savefig(OUT/'fig_landscape.jpg',dpi=400,quality=96) if False else fig.savefig(OUT/'fig_landscape.jpg',dpi=400,pil_kwargs={'quality':96})
fig.savefig(OUT/'fig_landscape.pdf',dpi=300)
metadata={'centre_lat':LAT,'centre_lon':LON,'width_m':W_M,'height_m':H_M,'zoom':Z,'ground_sampling_m':gsd,'layers':LAYERS,'verified_segments':len(segs),'blocked':sum(y for _,g,y,s in segs),'prediction_unit':'one classification score per nominal 50 m road segment','colour_mapping':'YlOrRd(score), range 0 to 1, no spatial interpolation','scores':'oof_ft_convnext_siamese_v2_verified.csv','segments':[{'sample_id':sid,'label':label,'score':score} for sid,g,label,score in segs]}
(OUT/'landscape_provenance.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in metadata.items() if k!='segments'},ensure_ascii=False))
