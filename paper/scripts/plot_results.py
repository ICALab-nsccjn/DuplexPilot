"""Recreate draft plots from explicitly transcribed report data.

No raw final-run measurements are inferred from the incomplete progress CSV.
Main plots use source-group summary statistics; the transfer plot uses printed,
rounded observations from the earlier, separate matched experiment.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
with (ROOT/'data/final_report_aggregates.csv').open() as f:
    rows = list(csv.DictReader(f))

def save(fig, name):
    fig.savefig(FIG/f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIG/f'{name}.png', dpi=210, bbox_inches='tight')
    plt.close(fig)

strategies = ['A-hot', 'F-cold', 'Joint', 'Hybrid']
labels = ['Hot\nresident', 'History\nrebuild', 'Joint\noffload', 'KV offload /\naudio resident']
x = np.arange(4)
fig, ax = plt.subplots(figsize=(7.2,2.8))
for i, env in enumerate(['E1','E2']):
    rr = [next(r for r in rows if r['environment']==env and r['strategy']==s) for s in strategies]
    vals = [float(r['render_s']) for r in rr]
    bars = ax.bar(x+(i-.5)*.34, vals, .34, label=env, hatch='//' if i==0 else None)
    for bar,val in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,val+.18,f'{val:.2f}',ha='center',va='bottom',fontsize=8)
ax.set_xticks(x,labels)
ax.set_ylabel('Worklet resume to first-new-PCM render (s)')
ax.set_ylim(0,17)
ax.legend(frameon=False,ncol=2,loc='upper right')
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
save(fig,'fig_main_results')

# Separate axes/files for the distinct browser environments; no pooling.
for env in ['E1','E2']:
    fig,ax=plt.subplots(figsize=(4.5,2.95))
    for s,label,mark in zip(strategies,['Hot','Rebuild','Joint','Hybrid'],['o','s','D','^']):
        r=next(r for r in rows if r['environment']==env and r['strategy']==s)
        xx,yy=float(r['gpu1_net_reclaimed_mib']),float(r['render_s'])
        ax.scatter([xx],[yy],marker=mark,s=65,label=label)
        offset={'A-hot':(9,9),'F-cold':(-8,8),'Joint':(-9,10),'Hybrid':(9,25)}[s]
        ax.annotate(label,(xx,yy),xytext=offset,textcoords='offset points',
                    ha='left' if xx<100 else 'right',fontsize=9)
    ax.set_xlim(-65,735);ax.set_ylim(0,17.2)
    ax.set_xlabel('GPU1 net allocated memory reclaimed (MiB)')
    ax.set_ylabel('Worklet resume to first-new-PCM render (s)')
    ax.set_title(env,fontsize=11)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();save(fig,f'fig_tradeoff_{env.lower()}')
# Original advertised basename is the E2 panel, explicitly documented.
import shutil
for ext in ['pdf','png']:
    shutil.copyfile(FIG/f'fig_tradeoff_e2.{ext}',FIG/f'fig_tradeoff.{ext}')

with (ROOT/'data/transfer_report_pairs.csv').open() as f:
    pairs=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(7.2,2.8))
y=np.arange(len(pairs))
a=np.array([float(r['sync_s']) for r in pairs]);b=np.array([float(r['packed_s']) for r in pairs])
ax.hlines(y,b,a,linewidth=1.15,alpha=.45)
ax.plot(a,y,'o',label='Synchronous reference',markersize=4.5)
ax.plot(b,y,'s',label='Packed + verified',markersize=4.5)
ax.set_yticks(y,[r['source']+' / '+r['repeat'] for r in pairs],fontsize=8)
ax.invert_yaxis();ax.set_xlabel('Worklet resume to newly computed PCM (s)')
ax.set_xlim(2.65,3.98)
ax.text(3.90,6,'Reported median\npaired reduction:\n0.533 s / 16.00%',ha='right',va='center',fontsize=9)
ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.53,1.18),ncol=2,fontsize=9)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout();save(fig,'fig_transfer')

# Historical APR observations; mean ratios only, no invented uncertainty bars.
fig,ax=plt.subplots(figsize=(6.4,2.7))
for workload,values,mark in [('A',[1.0294,.9873,1.0083],'o'),('B',[.9842,1.0193,.9808],'s'),('C',[1.0047,.9762,.9240],'^')]:
    ax.plot([4,8,16],values,marker=mark,label='Workload '+workload)
ax.axhline(1,linestyle='--',linewidth=.9)
ax.set_xticks([4,8,16]);ax.set_ylim(.90,1.055)
ax.set_xlabel('Logical concurrency N (historical APR matrix)')
ax.set_ylabel('Mean paired throughput ratio\nAPR / original affinity')
ax.legend(frameon=False,ncol=3,loc='upper center',fontsize=9)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout();save(fig,'fig_apr_history')
print('Generated six experimental PDF/PNG figure variants from declared report data.')
