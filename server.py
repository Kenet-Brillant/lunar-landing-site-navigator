"""Lunar Landing Site Navigator V1. Run: pip install -r requirements.txt; python server.py"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from functools import lru_cache
import json,math
import numpy as np
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
import rasterio
from pyproj import CRS,Transformer
BASE=Path(__file__).parent
app=FastAPI(title='Lunar Landing Site Navigator V1')
app.mount('/static',StaticFiles(directory=BASE/'static'),name='static')
R=1737400.0
# Spherical lunar south polar stereographic. Positive east longitude.
PROJ=CRS.from_proj4('+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs')
GEO=CRS.from_proj4('+proj=longlat +a=1737400 +b=1737400 +no_defs')
TO_XY=Transformer.from_crs(GEO,PROJ,always_xy=True)
TO_LL=Transformer.from_crs(PROJ,GEO,always_xy=True)
DEM_PATH=BASE/'data'/'ldem_75s_120m.img'
DEM=np.memmap(DEM_PATH,dtype='<i2',mode='r',shape=(7624,7624)) if DEM_PATH.exists() else None
EXTENT=457440.0; PIXEL=120.0
LAYERS={'earth':'AVGVISIB_75S_120M_201608_EARTH.tiff','sun':'AVGVISIB_85S_060M_201608.tiff','psr':'LPSR_85S_060M_201608.tiff'}
@lru_cache(maxsize=3)
def layer(name):
 p=BASE/'data'/LAYERS[name]
 return rasterio.open(p) if p.exists() else None
class Point(BaseModel):
 lat:float=Field(ge=-90,le=-75)
 lon:float=Field(ge=-360,le=360)
class Sim(Point):
 date:str
 hours:int=Field(default=72,ge=1,le=336)
 step:int=Field(default=3,ge=1,le=24)

def xy(lat,lon):return TO_XY.transform(lon,lat)
def alt_xy(x,y):
 if DEM is None:return None
 col=int((x+EXTENT)/PIXEL);row=int((EXTENT-y)/PIXEL)
 if 0<=row<7624 and 0<=col<7624:return float(DEM[row,col])*.5
 return None

def horizon(lat,lon,bearings=72,dist_km=35):
 """Approximate terrain skyline, corrected for lunar curvature; ray sampling every 240 m."""
 x,y=xy(lat,lon); h0=alt_xy(x,y)
 if h0 is None:raise HTTPException(422,'Site outside DEM')
 out=[]
 # azimuth in geographic tangent plane, computed via small coordinate displacement in projected plane
 d=10.0
 lon0,lat0=TO_LL.transform(x,y)
 for k in range(bearings):
  az=k*360/bearings;theta=math.radians(az)
  # projected grid north is not local geographic north away from central meridian.
  # use geographic small-step forward to orient ray in local true north/east.
  la=math.radians(lat);lo=math.radians(lon)
  # local north/east small angular offsets
  dlat=d*math.cos(theta)/R;dlon=d*math.sin(theta)/(R*max(abs(math.cos(la)),1e-7))
  tx,ty=xy(lat+math.degrees(dlat),lon+math.degrees(dlon))
  vx,vy=tx-x,ty-y;norm=math.hypot(vx,vy)
  if norm<1e-9:out.append(0);continue
  vx/=norm;vy/=norm
  peak=-90.0
  for meters in range(240,int(dist_km*1000)+1,240):
   h=alt_xy(x+vx*meters,y+vy*meters)
   if h is None:break
   angle=math.degrees(math.atan2(h-h0-meters*meters/(2*R),meters))
   peak=max(peak,angle)
  out.append(round(peak,3))
 return out

def sample_layers(lat,lon):
 x,y=xy(lat,lon);result={}
 for key in LAYERS:
  d=layer(key)
  if d is None:result[key]=None;continue
  try:
   a=next(d.sample([(x,y)],masked=True))[0]
   result[key]=None if np.ma.is_masked(a) or a<=-32000 else round(float(a)*d.scales[0]+d.offsets[0],5)
  except Exception:result[key]=None
 return result

def eph(lat,lon,alt_m,dates):
 try:
  from astroquery.jplhorizons import Horizons
  from astropy.time import Time
 except ImportError:
  raise HTTPException(503,'Installer astroquery et astropy : pip install -r requirements.txt')
 try:
  jd=Time([datetime.fromisoformat(d.replace('Z','+00:00')) for d in dates],scale='utc').jd.tolist()
  out={}
  for name,id in [('sun','10'),('earth','399')]:
   # Horizons: observer at lunar surface (body 301), longitude east, latitude planetocentric.
   h=Horizons(id=id,location={'lon':lon,'lat':lat,'elevation':alt_m/1000,'body':301},epochs=jd,id_type=None)
   rows=h.ephemerides(quantities='4',extra_precision=True)
   out[name]=[{'az':float(row['AZ']),'el':float(row['EL'])} for row in rows]
  return out
 except Exception as e:
  raise HTTPException(503,'Éphémérides JPL indisponibles : '+str(e)[:260])

def visible(el,az,h):
 if el is None or az is None:return None
 i=(az%360)/360*len(h);i0=int(i)%len(h);f=i-int(i)
 skyline=h[i0]*(1-f)+h[(i0+1)%len(h)]*f
 return bool(el>skyline)
@app.get('/')
def index():return FileResponse(BASE/'static'/'index.html')
@app.get('/api/status')
def status():return {'dem':DEM is not None,'layers':{k:layer(k) is not None for k in LAYERS},'horizons':'network-required'}
@app.post('/api/site')
def site(p:Point):
 x,y=xy(p.lat,p.lon)
 return {'lat':p.lat,'lon':p.lon,'x':x,'y':y,'alt_m':alt_xy(x,y),'layers':sample_layers(p.lat,p.lon)}
@app.post('/api/simulate')
def simulate(p:Sim):
 try:start=datetime.fromisoformat(p.date.replace('Z','+00:00'))
 except Exception:raise HTTPException(422,'Date ISO invalide')
 if start.tzinfo is None:start=start.replace(tzinfo=timezone.utc)
 dates=[start+timedelta(hours=i) for i in range(0,p.hours+1,p.step)]
 h=horizon(p.lat,p.lon)
 x,y=xy(p.lat,p.lon);raw=eph(p.lat,p.lon,alt_xy(x,y),[d.isoformat() for d in dates])
 series=[]
 for i,dt in enumerate(dates):
  sun=raw['sun'][i];earth=raw['earth'][i]
  series.append({'time':dt.isoformat(),'sun':{**sun,'visible':visible(sun['el'],sun['az'],h)},'earth':{**earth,'visible':visible(earth['el'],earth['az'],h)}})
 total=len(series)
 return {'site':site(Point(lat=p.lat,lon=p.lon)),'horizon':h,'series':series,'summary':{'sun_percent':round(100*sum(r['sun']['visible'] for r in series)/total,1),'earth_percent':round(100*sum(r['earth']['visible'] for r in series)/total,1),'both_percent':round(100*sum(r['sun']['visible'] and r['earth']['visible'] for r in series)/total,1)},'notes':['Estimations échantillonnées, pas de temps choisi par utilisateur','Horizon LOLA à 120 m/pixel, rayon 35 km, pas spatial 240 m','Visibilité du centre des disques; pas de modélisation de liaison radio','Pourcentage sur les instants échantillonnés, pas une mesure continue']}
if __name__=='__main__':
 import uvicorn
 uvicorn.run(app,host='127.0.0.1',port=8000)
