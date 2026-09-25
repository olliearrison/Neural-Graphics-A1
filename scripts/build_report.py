"""Build the assignment's requested writeup sections from saved results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, NullLocator
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

REPO = 'https://github.com/olliearrison/Neural-Graphics-A1.git'
ARCHS = ('small', 'medium', 'large')
PROVIDED = ('gradient', 'bricks', 'clouds')
OWN = ('brick_photo', 'grass', 'gravel')
NAMES = {'gradient':'Gradient', 'bricks':'Bricks', 'clouds':'Clouds', 'brick_photo':'Photographic brick', 'grass':'Grass', 'gravel':'Gravel'}
PALETTE = {'small':'#2070b4', 'medium':'#009d88', 'large':'#df7c25'}
INK = colors.HexColor('#182f40')
TEAL = colors.HexColor('#007e79')
MUTED = colors.HexColor('#536675')
AI_DISCLOSURE = ('OpenAI Codex developed the implementation, tests and report from an initial sampler draft; sourced the additional textures; ran training and verification; and drafted the figures and analysis. Reported metrics were computed by the executed code.')
plt.rcParams.update({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False,
                     'axes.labelcolor':'#283b49', 'text.color':'#172d3e', 'axes.titleweight':'bold'})
FONT_ROOT = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
for alias, filename in [('Helvetica','DejaVuSans.ttf'), ('Helvetica-Bold','DejaVuSans-Bold.ttf'),
                        ('Helvetica-Oblique','DejaVuSans-Oblique.ttf'), ('Helvetica-BoldOblique','DejaVuSans-BoldOblique.ttf')]:
    pdfmetrics.registerFont(TTFont(alias, str(FONT_ROOT / filename)))


class Results:
    def __init__(self, root):
        self.root = root
        self.directory = root / 'results'

    def metrics(self, name, arch):
        return json.loads((self.directory / name / arch / 'metrics.json').read_text())

    def baseline(self, name):
        return json.loads((self.directory / name / 'baseline.json').read_text())

    def image(self, name, method):
        return self.directory / name / (f'{method}.png' if method in ('original','s3tc') else f'{method}/float32.png')


def save_figure(fig, directory, name):
    path = directory / (name + '.png')
    fig.savefig(path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def image_grid(results, names, methods, directory, filename, figsize):
    fig, axes = plt.subplots(len(names),len(methods),figsize=figsize,layout='constrained',squeeze=False)
    for row,name in enumerate(names):
        for col,method in enumerate(methods):
            ax = axes[row,col]
            ax.imshow(Image.open(results.image(name,method)))
            label = 'S3TC' if method == 's3tc' else method.title()
            ax.set_title(f'{NAMES[name]} / {label}',fontsize=10 if len(methods)==2 else 9)
            ax.axis('off')
    return save_figure(fig,directory,filename)


def training_plot(results,directory):
    fig, axes = plt.subplots(1,3,figsize=(7.4,2.65),layout='constrained')
    for ax,name in zip(axes,PROVIDED):
        for arch in ARCHS:
            h=results.metrics(name,arch)['training']['history']
            ax.plot([p['step'] for p in h],[p['psnr'] for p in h],color=PALETTE[arch],linewidth=1.25)
        ax.axhline(results.baseline(name)['psnr'],color='#172d3e',linestyle=':',linewidth=1)
        ax.set_title(NAMES[name],fontsize=10)
        ax.set_xlim(0,2000)
        ax.set_xticks([0,1000,2000])
        ax.set_xlabel('Training step',fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('PSNR (dB)')
    handles=[Line2D([],[],color=PALETTE[a],label=a.title()) for a in ARCHS]
    handles.append(Line2D([],[],color='#172d3e',linestyle=':',label='S3TC'))
    fig.legend(handles=handles,loc='outside lower center',ncol=4,frameon=False,fontsize=8)
    return save_figure(fig,directory,'requested_training_curves')


def size_plot(results,directory,quantized=False):
    fig, axes = plt.subplots(1,3,figsize=(7.4,2.65),layout='constrained')
    modes=[('float32','-','o')]
    if quantized:
        modes.append(('uint8','--','D'))
    for ax,name in zip(axes,PROVIDED):
        for mode,style,marker in modes:
            points=[results.metrics(name,a)[mode] for a in ARCHS]
            ax.plot([p['stored_bytes']/1024 for p in points],[p['psnr'] for p in points],color='#9eafb9',linestyle=style,linewidth=.8)
            for arch,p in zip(ARCHS,points):
                ax.scatter(p['stored_bytes']/1024,p['psnr'],color=PALETTE[arch],marker=marker,s=31,zorder=4)
        baseline=results.baseline(name)
        ax.scatter(baseline['stored_bytes']/1024,baseline['psnr'],color='#172d3e',marker='*',s=100,zorder=5)
        ax.set_title(NAMES[name],fontsize=10)
        ax.set_xscale('log')
        ax.set_xticks([25,50,100,200,400] if quantized else [50,100,200,400])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(22 if quantized else 43,440)
        ax.set_xlabel('Size (KiB)',fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('PSNR (dB)')
    handles=[Line2D([],[],color=PALETTE[a],marker='o',linestyle='',label=a.title()) for a in ARCHS]
    if quantized:
        handles.extend([Line2D([],[],color='#536675',marker='o',linestyle='-',label='Float32'),Line2D([],[],color='#536675',marker='D',linestyle='--',label='8-bit grids')])
    handles.append(Line2D([],[],color='#172d3e',marker='*',linestyle='',markersize=10,label='S3TC'))
    fig.legend(handles=handles,loc='outside lower center',ncol=len(handles),frameon=False,fontsize=8)
    return save_figure(fig,directory,'requested_quantized_size_quality' if quantized else 'requested_float_size_quality')


def own_images(results,directory):
    fig,axes=plt.subplots(1,3,figsize=(7.4,2.7),layout='constrained')
    for ax,name in zip(axes,OWN):
        ax.imshow(Image.open(results.image(name,'original')))
        ax.set_title(NAMES[name],fontsize=10)
        ax.axis('off')
    return save_figure(fig,directory,'requested_own_sources')


class Report:
    def __init__(self,path):
        self.c=canvas.Canvas(str(path),pagesize=(612,792))
        self.c.setTitle('A1: Neural Texture Compression')
        self.page=0
        self.x,self.width,self.y=44,524,683

    def new(self,kicker,title):
        if self.page:
            self.footer()
            self.c.showPage()
        self.page+=1
        self.x,self.width,self.y=44,524,683
        self.c.setFillColor(TEAL)
        self.c.rect(44,749,34,4,fill=1,stroke=0)
        self.c.setFont('Helvetica-Bold',9)
        self.c.drawString(88,748,kicker)
        self.c.setFillColor(INK)
        size=min(24,24*524/pdfmetrics.stringWidth(title,'Helvetica-Bold',24))
        self.c.setFont('Helvetica-Bold',size)
        self.c.drawString(44,711,title)

    def column(self,x,width,top):
        self.x,self.width,self.y=x,width,top

    def check(self):
        if self.y<49:
            raise RuntimeError(f'Page {self.page} overflow at y={self.y}')

    def p(self,text,size=9.5,space=8,color=INK):
        p=Paragraph(text,ParagraphStyle('p',fontName='Helvetica',fontSize=size,leading=size*1.34,textColor=color))
        _,height=p.wrap(self.width,720)
        self.y-=height
        p.drawOn(self.c,self.x,self.y)
        self.y-=space
        self.check()

    def h(self,text):
        self.y-=3
        self.c.setFillColor(TEAL)
        self.c.setFont('Helvetica-Bold',10.5)
        self.c.drawString(self.x,self.y,text)
        self.y-=18

    def figure(self,path,max_height=None):
        with Image.open(path) as image:
            width=self.width
            height=width*image.height/image.width
        if max_height and height>max_height:
            width*=max_height/height
            height=max_height
        self.y-=height
        self.c.drawImage(str(path),self.x+(self.width-width)/2,self.y,width,height,mask='auto')
        self.y-=8
        self.check()

    def table(self,rows,widths,font=8.5,pad=5):
        t=Table(rows,colWidths=widths)
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),INK),('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTNAME',(0,1),(-1,-1),'Helvetica'),
            ('FONTSIZE',(0,0),(-1,-1),font),('LEADING',(0,0),(-1,-1),font+3),
            ('TOPPADDING',(0,0),(-1,-1),pad),('BOTTOMPADDING',(0,0),(-1,-1),pad),
            ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#edf4f6'),colors.white]),
            ('ALIGN',(1,0),(-1,-1),'RIGHT'),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
        _,height=t.wrap(self.width,720)
        self.y-=height
        t.drawOn(self.c,self.x,self.y)
        self.y-=9
        self.check()

    def footer(self):
        self.c.setStrokeColor(colors.HexColor('#d7e2e7'))
        self.c.line(44,38,568,38)
        self.c.setFillColor(MUTED)
        self.c.setFont('Helvetica',8)
        self.c.drawString(44,24,'15-674 / A1: Neural Texture Compression')
        self.c.drawRightString(568,24,str(self.page))

    def save(self):
        self.footer()
        self.c.save()


def build(root,output,figures):
    data=Results(root)
    figures.mkdir(parents=True,exist_ok=True)
    p2=image_grid(data,PROVIDED,('original','s3tc'),figures,'requested_s3tc_comparisons',(4.4,6.5))
    p6=image_grid(data,PROVIDED,('original',*ARCHS),figures,'requested_neural_comparisons',(8.6,6.45))
    training=training_plot(data,figures)
    float_plot=size_plot(data,figures)
    quant_plot=size_plot(data,figures,quantized=True)
    own=own_images(data,figures)
    r=Report(output)

    r.new('P2','S3TC baseline')
    r.p(f'<b>Code:</b> <link href="{REPO}" color="#007e79">{REPO}</link>',size=9)
    r.p('All textures are 512 x 512 RGB: 768 KiB raw. 1 KiB = 1,024 bytes. Ratios are compressed/raw. PSNR uses full-image normalized RGB MSE before PNG rounding.',size=9)
    top=r.y
    r.column(44,243,top)
    r.figure(p2)
    r.column(303,265,top)
    rows=[['Texture','PSNR (dB)','Ratio']]
    for name in PROVIDED:
        b=data.baseline(name)
        rows.append([NAMES[name],f"{b['psnr']:.2f}",f"{100*b['ratio']:.2f}%"])
    r.table(rows,[93,88,84],font=8.5)
    r.p('Each S3TC texture uses 128 KiB (6.00x smaller than raw RGB).',size=9)
    r.h('Endpoint selection')
    r.p('For each 4 x 4 block, try principal-axis extremes and componentwise color bounds. Quantize to RGB565, assign nearest palette entries, and refine endpoints by least squares for eight iterations, retaining the lowest-error pair.',size=9)
    r.h('Visible limitations')
    r.p('<b>Gradient:</b> blockwise color steps interrupt the smooth ramp. <b>Bricks:</b> slight shade shifts and edge-color errors; most blocks contain few colors, so damage is mild. <b>Clouds:</b> small block and color steps appear in smooth regions. Each block has only four colors on one RGB line.',size=9)
    r.h('Why the extra green bit?')
    r.p('Human brightness perception is most sensitive to green, so its extra precision reduces visible luminance errors.',size=9)
    r.h('RGB565 approximation')
    r.p('The permitted q/31 or q/63 decode closely approximates integer bit expansion. For example, 5-bit q=3 gives 3/31 versus 24/255 after bit replication; finite-byte rounding causes the small difference.',size=9)

    r.new('P6','Neural reconstructions and results')
    r.figure(p6,max_height=322)
    r.p('Float32 results after 2,000 steps. Size = all grid and MLP parameters x 4 bytes; raw RGB is 768 KiB. File headers are excluded.',size=8.8,space=6)
    rows=[['Texture / model','PSNR (dB)','Size (KiB)','Ratio']]
    for name in PROVIDED:
        for arch in ARCHS:
            m=data.metrics(name,arch)['float32']
            rows.append([NAMES[name]+' / '+arch.title(),f"{m['psnr']:.2f}",f"{m['stored_bytes']/1024:.2f}",f"{100*m['ratio']:.2f}%"])
    r.table(rows,[202,108,108,106],font=8.5,pad=4)

    r.new('P6','Training and size-quality comparison')
    r.figure(training,max_height=185)
    r.p('Full-image PSNR every 100 steps for all nine fits; dotted lines show S3TC.',size=8.5,space=7,color=MUTED)
    r.figure(float_plot,max_height=185)
    r.p('Measured float32 size versus PSNR; stars show S3TC. Connecting lines only guide the eye.',size=8.5,space=8,color=MUTED)
    r.p('<b>Gradient:</b> Large is best at 57.23 dB, but adds only 0.44 dB over Small: the smooth signal needs little capacity. All three exceed S3TC quality at the final step.',size=9.3)
    r.p('<b>Bricks:</b> Large is best at 39.68 dB because its finer grid represents narrow mortar edges better. Medium gains 2.89 dB over Small; all final neural fits trail S3TC (42.02 dB).',size=9.3)
    r.p('<b>Clouds:</b> Large is best at 49.52 dB: its finer grid and extra features capture detail across scales. Medium gains 1.41 dB over Small. Only Large exceeds S3TC quality (42.87 dB).',size=9.3)
    r.p('<b>Size comparison for each texture:</b> Small (6.48%) and Medium (7.91%) beat S3TC\'s 16.67% ratio; Large float32 (47.04%) does not.',size=9.3)

    r.new('P7','Eight-bit grid quantization')
    r.p('Each grid uses its own uniform 8-bit quantizer; MLP parameters remain float32. Sizes include the uint8 indices and one float32 (lo, scale) pair per grid.',size=9.3)
    rows=[['Texture / model','Before dB','After dB','Loss dB','KiB','Ratio','Factor']]
    for name in PROVIDED:
        for arch in ARCHS:
            m=data.metrics(name,arch);f=m['float32'];q=m['uint8']
            rows.append([NAMES[name]+' / '+arch.title(),f"{f['psnr']:.3f}",f"{q['psnr']:.3f}",f"{f['psnr']-q['psnr']:+.3f}",f"{q['stored_bytes']/1024:.2f}",f"{100*q['ratio']:.2f}%",f"{q['factor']:.2f}x"])
    r.table(rows,[139,65,65,65,58,66,66],font=8,pad=5)
    r.figure(quant_plot,max_height=180)
    r.p('The P6 size-quality plot with quantized grids added as diamonds. All quantized models are smaller than S3TC. Quantized Large also beats S3TC PSNR on gradient and clouds.',size=9)
    r.p('Rounding perturbs the features supplied to the decoder. The largest loss is 1.105 dB for Small on the gradient, whose original error is already tiny; other losses are at most 0.401 dB. Large bricks improves by 0.004 dB because rounding happens to reduce residual error. The MLP staying float32 makes total savings smaller than 4x.',size=9)

    r.new('P8','Three additional textures')
    r.figure(own,max_height=172)
    r.p('CC0 samples from <link href="https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/_fetchers.py" color="#007e79">scikit-image v0.25.2</link>: CC0Textures Bricks25, linolafett\'s Grass 01, and CC0Textures Gravel04. The 512 x 512 grayscale sources are repeated into RGB without resizing.',size=8.8)
    rows=[['Texture','Small','Medium','Large'],['Stored size (KiB)',*[f"{data.metrics('brick_photo',a)['float32']['stored_bytes']/1024:.2f}" for a in ARCHS]]]
    for name in OWN:
        rows.append([NAMES[name],*[f"{data.metrics(name,a)['float32']['psnr']:.2f} dB" for a in ARCHS]])
    r.table(rows,[190,111,111,112],font=9,pad=6)
    r.p('Each image is a fresh fit under all three architectures. Sizes count float32 grid and MLP parameters; the raw RGB reference is 768 KiB.',size=8.8)
    r.p('<b>Photographic brick:</b> regular mortar boundaries and smooth faces are relatively easy to represent. Large restores sharp boundaries and reaches 43.38 dB; the smaller models trade edge precision for smaller storage.',size=9.5)
    r.p('<b>Grass:</b> irregular, overlapping fine blades carry high-frequency detail. Small and Medium blur this structure; Large improves to 25.89 dB but still loses fine strands. It has the lowest quality at every model size.',size=9.5)
    r.p('<b>Gravel:</b> stones form larger coherent shapes than grass strands, so quality is higher at the same sizes. Large recovers much more boundary detail at 31.04 dB; Small and Medium remain compact but visibly smoother.',size=9.5)
    r.h('AI disclosure')
    r.p(AI_DISCLOSURE,size=8.7)
    r.save()
    print(f'Created {output}: {r.page} pages')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--figures',type=Path)
    args=parser.parse_args()
    output=args.output or args.root/'writeup.pdf'
    figures=args.figures or args.root/'results'/'figures'
    output.parent.mkdir(parents=True,exist_ok=True)
    build(args.root,output,figures)
