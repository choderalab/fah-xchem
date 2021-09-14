from pathlib import Path
import re

dir = Path('../receptors/dimer')
thiols = dir.glob('*protein.pdb')
thiolates = dir.glob('*protein-thiolate.pdb')


def get_id(x):
    return re.findall('[xP][0-9]{4}', x)[0]


lines = ['reinitialize']
lines += [f"load {x.name}, {'n'+get_id(x.stem)}" for x in thiols]
lines += [f"load {x.name}, {'c'+get_id(x.stem)}" for x in thiolates]
lines += ['hide cartoon', 'show sticks, resi 41 or resi 145']

with dir.absolute().joinpath('vis_all.pml').open('wt') as f:
    f.write('\n'.join(lines))