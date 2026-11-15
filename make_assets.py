from pathlib import Path
import numpy as np, rasterio
from PIL import Image,ImageOps
from rasterio.enums import Resampling
BASE=Path(__file__).parent
SRC=Path('/mnt/data')
for name,src in [('earth','AVGVISIB_75S_120M_201608_EARTH.tiff'),('sun','AVGVISIB_85S_060M_201608.tiff'),('psr','LPSR_85S_060M_201608.tiff')]:
 with rasterio.open(SRC/src) as d:
  a=d.read(1,out_shape=(1100,1100),resampling=Resampling.average).astype(float)
  a[a<=-32000]=np.nan
  if name=='psr':v=np.nan_to_num(a*d.scales[0]+d.offsets[0])*255
  else:v=np.nan_to_num(a*d.scales[0])*255
  v=np.clip(v,0,255).astype('uint8')
  Image.fromarray(v,'L').save(BASE/'static'/f'{name}.png')
# downsample actual raw terrain (scaled signed 16-bit, meters = DN*0.5)
raw=np.memmap(SRC/'ldem_75s_120m(2).img',dtype='<i2',mode='r',shape=(7624,7624))
a=np.asarray(raw[::7,::7],dtype=float)*.5
p1,p99=np.percentile(a,[2,98]); v=np.uint8(np.clip((a-p1)/(p99-p1),0,1)*255)
Image.fromarray(v,'L').resize((1100,1100)).save(BASE/'static'/'terrain.png')
print('assets ready',p1,p99)
