"""Install the pinned FuxiCTR source layout and the unmodified FairFS DCN files."""
from pathlib import Path
import json,shutil,subprocess
R=Path(__file__).resolve().parents[1];info=json.loads((R/'third_party/SOURCES.json').read_text())['FuxiCTR'];dst=R/'third_party/FuxiCTR'
if not dst.exists():subprocess.run(['git','clone',info['url'],str(dst)],check=True)
subprocess.run(['git','-C',str(dst),'checkout',info['commit']],check=True)
shutil.copytree(R/'third_party/fairfs/DCN',dst/'model_zoo/DCN',dirs_exist_ok=True)
print('FuxiCTR v2.1.3 and author FairFS-DCN are ready. Install requirements-fairfs.txt in its separate Python environment.')
