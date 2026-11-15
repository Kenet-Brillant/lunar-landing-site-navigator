"""Verifier les donnees avant demarrage : python preflight.py"""
from pathlib import Path
import sys
base=Path(__file__).resolve().parent
required={
 'DEM LOLA 120 m': ('data/ldem_75s_120m.img', 7624*7624*2),
 'Interface': ('static/index.html', None),
}
optional={
 'Terre historique': 'data/AVGVISIB_75S_120M_201608_EARTH.tiff',
 'Soleil historique': 'data/AVGVISIB_75S_120M_201608.tiff',
 'PSR': 'data/LPSR_85S_060M_201608.tiff',
 'Fond lunaire': 'static/lunar-background-v7.jpg',
}
ok=True
for label,(file,size) in required.items():
 p=base/file; good=p.is_file() and (size is None or p.stat().st_size==size)
 print(('OK  ' if good else 'ERREUR  ')+label+': '+str(p))
 ok &= good
for label,file in optional.items():
 p=base/file; print(('OK  ' if p.is_file() else 'ABSENT (optionnel)  ')+label+': '+str(p))
print('\n'+('Preflight OK' if ok else 'Preflight incomplet : voir les fichiers requis.'))
sys.exit(0 if ok else 1)
