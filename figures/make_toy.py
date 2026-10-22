"""Figure 2 of the paper: the construction of Theorem 3.3 with four fields (computed exactly, no data needed).
Usage:  python make_toy.py   ->  figures/toy.pdf, figures/toy.png"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "pdf.fonttype": 42})
INK2="#52514e"; BLUE="#2a78d6"; ORG="#eb6834"; GREY="#c9c7c0"
# exact construction of Theorem 3.3 with F=4, k=2: a_i = 1, beta = 0.5 on two matched pairs
F=4; beta=0.5; names=["A","B","C","D"]
pairs1=[(0,1),(2,3)]; pairs2=[(0,2),(1,3)]
def U(S,pairs): return len(S)+beta*sum(1 for i,j in pairs if i in S and j in S)
import itertools
def phi(i,m,pairs):
    others=[j for j in range(F) if j!=i]; v=[U(set(T)|{i},pairs)-U(set(T),pairs) for T in itertools.combinations(others,m-1)]
    return np.mean(v)
P1=np.array([[phi(i,m,pairs1) for m in range(1,F+1)] for i in range(F)]); P2=np.array([[phi(i,m,pairs2) for m in range(1,F+1)] for i in range(F)])
assert np.allclose(P1,P2)
fig=plt.figure(figsize=(3.35,1.5))
ax0=fig.add_axes([0.07,0.22,0.27,0.62])
x=np.arange(F); w=0.36
ax0.bar(x-w/2,P1[:,1],w*0.95,color=BLUE,label="problem 1"); ax0.bar(x+w/2,P2[:,1],w*0.95,color=ORG,label="problem 2")
ax0.set_xticks(x); ax0.set_xticklabels(names); ax0.set_ylim(0,1.9); ax0.set_yticks([0,0.5,1,1.5])
ax0.set_title("importance score\n(identical in both problems)", fontsize=6.4, pad=2)
for s in ("top","right"): ax0.spines[s].set_visible(False)
ax0.tick_params(labelsize=6)
ax0.legend(fontsize=5.2, frameon=False, loc="upper center", ncol=1, handlelength=0.8, bbox_to_anchor=(0.5,1.04), labelspacing=0.15)
pos={0:(0,1),1:(1,1),2:(0,0),3:(1,0)}
for idx,(pairs,col,best,title) in enumerate([(pairs1,BLUE,(0,1),"problem 1\nbest pair {A, B}"),(pairs2,ORG,(0,2),"problem 2\nbest pair {A, C}")]):
    ax=fig.add_axes([0.40+idx*0.31,0.08,0.27,0.76]); ax.set_xlim(-0.45,1.45); ax.set_ylim(-0.45,1.45); ax.axis("off")
    for i,j in itertools.combinations(range(F),2):
        on=(i,j) in pairs
        ax.plot([pos[i][0],pos[j][0]],[pos[i][1],pos[j][1]],color=col if on else GREY,lw=2.2 if on else 0.6,ls="-" if on else (0,(2,2)),zorder=1)
        if on:
            mx,my=(pos[i][0]+pos[j][0])/2,(pos[i][1]+pos[j][1])/2
            ax.text(mx+(0.17 if pos[i][0]==pos[j][0] else 0),my+(0.1 if pos[i][1]==pos[j][1] else 0),"$+\\beta$",fontsize=6.2,color=col,ha="center",va="center")
    for i in range(F):
        hl=i in best
        ax.scatter(*pos[i],s=170,color=col if hl else "white",edgecolor=col,linewidth=1.2,zorder=3)
        ax.text(*pos[i],names[i],ha="center",va="center",fontsize=7,color="white" if hl else "#0b0b0b",zorder=4)
    ax.set_title(title,fontsize=6.4,pad=1,color=col,linespacing=1.1)
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"figures"); os.makedirs(OUT,exist_ok=True)
fig.savefig(os.path.join(OUT,"toy.pdf"),bbox_inches="tight"); fig.savefig(os.path.join(OUT,"toy.png"),dpi=220,bbox_inches="tight")
print("importance profiles (identical in both problems):"); print(P1)
