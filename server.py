
"""Lunar Landing Site Navigator V1.

Installation:
    pip install -r requirements.txt

Démarrage:
    python server.py
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
from functools import lru_cache
import math

import numpy as np
import rasterio

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pyproj import CRS, Transformer


# ============================================================
# 1. CONFIGURATION GÉNÉRALE
# ============================================================

BASE = Path(__file__).parent

app = FastAPI(title="Lunar Landing Site Navigator V1")

app.mount(
    "/static",
    StaticFiles(directory=BASE / "static"),
    name="static"
)

# Rayon moyen de référence de la Lune, en mètres
R = 1737400.0

# Projection stéréographique polaire Sud lunaire
PROJ = CRS.from_proj4(
    "+proj=stere "
    "+lat_0=-90 "
    "+lat_ts=-90 "
    "+lon_0=0 "
    "+a=1737400 "
    "+b=1737400 "
    "+units=m "
    "+no_defs"
)

GEO = CRS.from_proj4(
    "+proj=longlat "
    "+a=1737400 "
    "+b=1737400 "
    "+no_defs"
)

TO_XY = Transformer.from_crs(
    GEO, PROJ, always_xy=True
)

TO_LL = Transformer.from_crs(
    PROJ, GEO, always_xy=True
)


# ============================================================
# 2. CHARGEMENT DES DONNÉES LOLA
# ============================================================

DEM_PATH = BASE / "data" / "ldem_75s_120m.img"

DEM = (
    np.memmap(
        DEM_PATH,
        dtype="<i2",
        mode="r",
        shape=(7624, 7624)
    )
    if DEM_PATH.exists()
    else None
)

EXTENT = 457440.0
PIXEL = 120.0

LAYERS = {
    "earth": "AVGVISIB_75S_120M_201608_EARTH.tiff",
    "sun": "AVGVISIB_85S_060M_201608.tiff",
    "psr": "LPSR_85S_060M_201608.tiff"
}


@lru_cache(maxsize=3)
def layer(name):
    path = BASE / "data" / LAYERS[name]

    return rasterio.open(path) if path.exists() else None


# ============================================================
# 3. MODÈLES DE DONNÉES
# ============================================================

class Point(BaseModel):
    lat: float = Field(ge=-90, le=-75)
    lon: float = Field(ge=-360, le=360)


class Sim(Point):
    date: str
    hours: int = Field(default=72, ge=1, le=336)
    step: float = Field(default=0.5, ge=0.5, le=24)


# ============================================================
# 4. COORDONNÉES ET ALTITUDE
# ============================================================

def xy(lat, lon):
    """Convertit latitude/longitude en coordonnées projetées."""
    return TO_XY.transform(lon, lat)


def alt_xy(x, y):
    """Retourne l'altitude LOLA en mètres."""

    if DEM is None:
        return None

    col = int((x + EXTENT) / PIXEL)
    row = int((EXTENT - y) / PIXEL)

    if 0 <= row < 7624 and 0 <= col < 7624:
        return float(DEM[row, col]) * 0.5

    return None


# ============================================================
# 5. NOUVEAU : CALCUL DE LA PENTE LOCALE
# ============================================================

def terrain_slope(lat, lon):
    """
    Estime la pente locale du terrain lunaire en degrés.

    Méthode :
        - Échantillonnage des altitudes voisines
        - Calcul des gradients en X et Y
        - Calcul de l'angle de pente

    Résultat :
        pente en degrés, ou None si données indisponibles.

    Attention :
        Estimation à la résolution du DEM LOLA.
        Ne constitue pas une certification de sécurité.
    """

    if DEM is None:
        return None

    x, y = xy(lat, lon)

    # Altitudes voisines dans le repère projeté
    z_east = alt_xy(x + PIXEL, y)
    z_west = alt_xy(x - PIXEL, y)
    z_north = alt_xy(x, y + PIXEL)
    z_south = alt_xy(x, y - PIXEL)

    values = [
        z_east,
        z_west,
        z_north,
        z_south
    ]

    if any(z is None for z in values):
        return None

    # Gradients d'altitude
    dz_dx = (z_east - z_west) / (2 * PIXEL)
    dz_dy = (z_north - z_south) / (2 * PIXEL)

    # Correction de l'échelle locale de la projection
    # stéréographique polaire sphérique.
    radial_distance = math.hypot(x, y)

    projection_scale = (
        1.0 + (radial_distance / (2.0 * R)) ** 2
    )

    gradient = math.hypot(dz_dx, dz_dy) * projection_scale

    slope_rad = math.atan(gradient)

    return round(math.degrees(slope_rad), 3)


# ============================================================
# 6. CALCUL DE L'HORIZON TOPOGRAPHIQUE
# ============================================================

def horizon(lat, lon, bearings=72, dist_km=35):
    """
    Horizon topographique approximatif.

    - 72 directions par défaut
    - Portée de 35 km
    - Échantillonnage tous les 240 m
    - Correction approximative de courbure lunaire
    """

    x, y = xy(lat, lon)
    h0 = alt_xy(x, y)

    if h0 is None:
        raise HTTPException(
            status_code=422,
            detail="Site outside DEM"
        )

    out = []

    d = 10.0

    for k in range(bearings):

        az = k * 360 / bearings
        theta = math.radians(az)

        la = math.radians(lat)

        dlat = d * math.cos(theta) / R

        dlon = (
            d * math.sin(theta)
            / (R * max(abs(math.cos(la)), 1e-7))
        )

        tx, ty = xy(
            lat + math.degrees(dlat),
            lon + math.degrees(dlon)
        )

        vx = tx - x
        vy = ty - y

        norm = math.hypot(vx, vy)

        if norm < 1e-9:
            out.append(0)
            continue

        vx /= norm
        vy /= norm

        peak = -90.0

        for meters in range(
            240,
            int(dist_km * 1000) + 1,
            240
        ):

            h = alt_xy(
                x + vx * meters,
                y + vy * meters
            )

            if h is None:
                break

            angle = math.degrees(
                math.atan2(
                    h - h0 - meters * meters / (2 * R),
                    meters
                )
            )

            peak = max(peak, angle)

        out.append(round(peak, 3))

    return out


# ============================================================
# 7. ÉCHANTILLONNAGE DES COUCHES NASA
# ============================================================

def sample_layers(lat, lon):

    x, y = xy(lat, lon)

    result = {}

    for key in LAYERS:

        dataset = layer(key)

        if dataset is None:
            result[key] = None
            continue

        try:
            value = next(
                dataset.sample(
                    [(x, y)],
                    masked=True
                )
            )[0]

            if np.ma.is_masked(value) or value <= -32000:
                result[key] = None
            else:
                result[key] = round(
                    float(value) * dataset.scales[0]
                    + dataset.offsets[0],
                    5
                )

        except Exception:
            result[key] = None

    return result


# ============================================================
# 8. ÉPHÉMÉRIDES JPL HORIZONS
# ============================================================

def eph(lat, lon, alt_m, dates):
    """Éphémérides par petits lots pour éviter les URL Horizons trop longues."""
    try:
        from astroquery.jplhorizons import Horizons
        from astropy.time import Time
    except ImportError:
        raise HTTPException(status_code=503, detail="Installer astroquery et astropy")

    try:
        parsed_dates = [
            datetime.fromisoformat(d.replace("Z", "+00:00")) for d in dates
        ]
        jd = Time(parsed_dates, scale="utc").jd.tolist()
        out = {"sun": [], "earth": []}
        # 25 instants par requête : évite les URL excessivement longues.
        chunk_size = 25
        for name, body_id in (("sun", "10"), ("earth", "399")):
            for offset in range(0, len(jd), chunk_size):
                batch = jd[offset:offset + chunk_size]
                observer = Horizons(
                    id=body_id,
                    location={
                        "lon": lon, "lat": lat,
                        "elevation": alt_m / 1000, "body": 301
                    },
                    epochs=batch,
                    id_type=None,
                )
                rows = observer.ephemerides(
                    quantities="4", extra_precision=True
                )
                positions = [
                    {"az": float(row["AZ"]), "el": float(row["EL"])}
                    for row in rows
                ]
                if len(positions) != len(batch):
                    raise ValueError(
                        f"Horizons a renvoyé {len(positions)} lignes "
                        f"pour {len(batch)} dates ({name})"
                    )
                out[name].extend(positions)
        return out
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Éphémérides JPL indisponibles : " + str(error)[:400],
        )


# ============================================================
# 9. VISIBILITÉ SOLEIL / TERRE
# ============================================================

def visible(el, az, h):

    if el is None or az is None:
        return None

    i = (az % 360) / 360 * len(h)

    i0 = int(i) % len(h)

    f = i - int(i)

    skyline = (
        h[i0] * (1 - f)
        + h[(i0 + 1) % len(h)] * f
    )

    return bool(el > skyline)


# ============================================================
# 10. ROUTES API
# ============================================================

@app.get("/")
def index():

    return FileResponse(
        BASE / "static" / "index.html"
    )


@app.get("/api/status")
def status():

    return {
        "dem": DEM is not None,
        "layers": {
            key: layer(key) is not None
            for key in LAYERS
        },
        "horizons": "network-required"
    }


# ============================================================
# 11. ANALYSE D'UN SITE
# ============================================================

@app.post("/api/site")
def site(p: Point):

    x, y = xy(p.lat, p.lon)

    altitude = alt_xy(x, y)

    slope = terrain_slope(p.lat, p.lon)

    return {
        "lat": p.lat,
        "lon": p.lon,
        "x": x,
        "y": y,
        "alt_m": altitude,

        # NOUVEAU
        "slope_deg": slope,

        "layers": sample_layers(
            p.lat,
            p.lon
        )
    }


# ============================================================
# 12. SIMULATION TEMPORELLE
# ============================================================

@app.post("/api/simulate")
def simulate(p: Sim):

    try:
        start = datetime.fromisoformat(
            p.date.replace("Z", "+00:00")
        )

    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Date ISO invalide"
        )

    if start.tzinfo is None:
        start = start.replace(
            tzinfo=timezone.utc
        )

    dates = [
        start + timedelta(hours=i * p.step)
        for i in range(int(p.hours / p.step) + 1)
    ]

    # Horizon du site
    h = horizon(
        p.lat,
        p.lon
    )

    x, y = xy(
        p.lat,
        p.lon
    )

    altitude = alt_xy(x, y)

    # Positions Soleil / Terre
    raw = eph(
        p.lat,
        p.lon,
        altitude,
        [d.isoformat() for d in dates]
    )

    series = []

    for i, dt in enumerate(dates):

        sun = raw["sun"][i]
        earth = raw["earth"][i]

        series.append({
            "time": dt.isoformat(),

            "sun": {
                **sun,
                "visible": visible(
                    sun["el"],
                    sun["az"],
                    h
                )
            },

            "earth": {
                **earth,
                "visible": visible(
                    earth["el"],
                    earth["az"],
                    h
                )
            }
        })

    total = len(series)

    sun_percent = round(
        100 * sum(
            r["sun"]["visible"]
            for r in series
        ) / total,
        1
    )

    earth_percent = round(
        100 * sum(
            r["earth"]["visible"]
            for r in series
        ) / total,
        1
    )

    both_percent = round(
        100 * sum(
            r["sun"]["visible"]
            and r["earth"]["visible"]
            for r in series
        ) / total,
        1
    )

    return {
        "site": site(
            Point(
                lat=p.lat,
                lon=p.lon
            )
        ),

        "horizon": h,

        "series": series,

        "summary": {
            "sun_percent": sun_percent,
            "earth_percent": earth_percent,
            "both_percent": both_percent
        },

        "notes": [
            "Estimations échantillonnées, pas de temps choisi par utilisateur",
            "Horizon LOLA à 120 m/pixel, rayon 35 km, pas spatial 240 m",
            "Visibilité du centre des disques; pas de modélisation de liaison radio",
            "Pourcentage sur les instants échantillonnés, pas une mesure continue",
            "Pente locale estimée à partir du DEM LOLA; sécurité d'atterrissage non évaluée"
        ]
    }


# ============================================================
# 14. DÉMARRAGE
# ============================================================


# ============================================================
# 13. PROCHAINES TRANSITIONS DE VISIBILITÉ (EXPLORATOIRE)
# ============================================================

class EventRequest(Point):
    date: str | None = None
    days: int = Field(default=14, ge=1, le=60)


def _margin(body, skyline):
    """Élévation du centre moins horizon local, en degrés."""
    az, el = body["az"], body["el"]
    index = (az % 360) * len(skyline) / 360
    j = int(index) % len(skyline)
    f = index - int(index)
    return el - (skyline[j] * (1 - f) + skyline[(j + 1) % len(skyline)] * f)


@app.post("/api/events")
def next_events(p: EventRequest):
    if p.date:
        try:
            start = datetime.fromisoformat(p.date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=422, detail="Date ISO invalide")
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        start = start.astimezone(timezone.utc)
    else:
        start = datetime.now(timezone.utc)

    x, y = xy(p.lat, p.lon)
    altitude = alt_xy(x, y)
    if altitude is None:
        raise HTTPException(status_code=422, detail="Site hors couverture LOLA")

    skyline = horizon(p.lat, p.lon)
    coarse_step = timedelta(hours=3)
    # Réutiliser les échantillons déjà calculés à chaque extension.
    results = {"sun": None, "earth": None}
    previous_time = None
    previous_positions = None
    first_positions = None
    horizons_days = sorted(set([p.days] + [d for d in (30, 60) if d > p.days]))
    searched_days = p.days

    for target_days in horizons_days:
        end = start + timedelta(days=target_days)
        if previous_time is None:
            dates = [start + i * coarse_step for i in range(int((end - start) / coarse_step) + 1)]
        else:
            dates = []
            t = previous_time + coarse_step
            while t <= end:
                dates.append(t)
                t += coarse_step
        if dates and dates[-1] < end:
            dates.append(end)
        if not dates:
            continue

        raw = eph(p.lat, p.lon, altitude, [d.isoformat() for d in dates])
        for i, dt in enumerate(dates):
            positions = {body: raw[body][i] for body in ("sun", "earth")}
            if first_positions is None:
                first_positions = positions
            if previous_positions is not None:
                for body in ("sun", "earth"):
                    if results[body] is not None:
                        continue
                    lo_visible = _margin(previous_positions[body], skyline) > 0
                    hi_visible = _margin(positions[body], skyline) > 0
                    if lo_visible == hi_visible:
                        continue
                    lo, hi = previous_time, dt
                    # Affinage temporel à ~1 minute, sans prétendre à
                    # une précision physique équivalente.
                    for _ in range(8):
                        mid = lo + (hi - lo) / 2
                        mid_pos = eph(p.lat, p.lon, altitude, [mid.isoformat()])[body][0]
                        if (_margin(mid_pos, skyline) > 0) == lo_visible:
                            lo = mid
                        else:
                            hi = mid
                    results[body] = {
                        "type": "set" if lo_visible else "rise",
                        "time": (lo + (hi - lo) / 2).isoformat(),
                        "uncertainty_seconds_sampling": round((hi - lo).total_seconds() / 2, 1)
                    }
            previous_time = dt
            previous_positions = positions

        searched_days = target_days
        if all(value is not None for value in results.values()):
            break

    bodies = {}
    for body in ("sun", "earth"):
        bodies[body] = {
            "visible_now": _margin(first_positions[body], skyline) > 0,
            "next_event": results[body],
            "search_days": searched_days
        }

    return {
        "site": {"lat": p.lat, "lon": p.lon},
        "calculated_at": start.isoformat(),
        "search_days": searched_days,
        "bodies": bodies,
        "notes": [
            "Recherche progressive sur 14, 30 puis 60 jours selon les événements manquants",
            "Échantillonnage initial toutes les 3 heures; transitions courtes potentiellement manquées",
            "Horizon topographique approximatif et visibilité du centre des disques",
            "L'affinage temporel ne garantit pas la précision physique"
        ]
    }




# ============================================================
# 15. PRÉSÉLECTION GÉOGRAPHIQUE (SANS ÉPHÉMÉRIDES)
# ============================================================
class ScanRequest(BaseModel):
    north_limit_lat: float = Field(default=-85, ge=-90, le=-75)
    spacing_km: int = Field(default=5, ge=2, le=10)
    max_slope: float = Field(default=10, ge=0, le=45)
    weight_earth: float = Field(default=33, ge=0, le=100)
    weight_terrain: float = Field(default=34, ge=0, le=100)
    weight_sun: float = Field(default=33, ge=0, le=100)

@app.post("/api/scan")
def scan_pole(p: ScanRequest):
    """Balayage exploratoire LOLA + visibilité terrestre historique.

    Le Soleil n'est PAS noté tant que sa couverture complète est absente.
    Les scores sont provisoires, pas des certifications d'atterrissage.
    """
    if DEM is None:
        raise HTTPException(status_code=503, detail="DEM LOLA absent")
    earth_ds = layer("earth")
    if earth_ds is None:
        raise HTTPException(status_code=503, detail="Couche historique Terre absente")
    if p.weight_earth + p.weight_terrain <= 0:
        raise HTTPException(status_code=422, detail="Donner un poids positif à la Terre ou au terrain ; le Soleil n'est pas encore noté")
    step = p.spacing_km * 1000.0
    # Disque centré sur le pôle Sud : limite nord configurable (-85° par défaut).
    max_r = 2 * R * math.tan(math.radians(90 + p.north_limit_lat) / 2)
    axis = np.arange(-max_r, max_r + 1, step)
    gx, gy = np.meshgrid(axis, axis)
    xs, ys = gx.ravel(), gy.ravel()
    keep = xs * xs + ys * ys <= max_r * max_r
    xs, ys = xs[keep], ys[keep]
    col = np.floor((xs + EXTENT) / PIXEL).astype(int)
    row = np.floor((EXTENT - ys) / PIXEL).astype(int)
    valid = (row > 0) & (row < DEM.shape[0]-1) & (col > 0) & (col < DEM.shape[1]-1)
    xs, ys, row, col = xs[valid], ys[valid], row[valid], col[valid]
    east = DEM[row, col+1].astype(float) * .5
    west = DEM[row, col-1].astype(float) * .5
    north = DEM[row-1, col].astype(float) * .5
    south = DEM[row+1, col].astype(float) * .5
    scale = 1 + (np.hypot(xs, ys) / (2 * R)) ** 2
    slope = np.degrees(np.arctan(np.hypot((east-west)/(2*PIXEL), (north-south)/(2*PIXEL))*scale))
    good = np.isfinite(slope) & (slope <= p.max_slope)
    xs, ys, slope, row, col = xs[good], ys[good], slope[good], row[good], col[good]
    # La lecture rasterio s'effectue dans le système projeté de la couche.
    values = list(earth_ds.sample(zip(xs.tolist(), ys.tolist()), masked=True))
    candidates = []
    w_e = p.weight_earth / (p.weight_earth + p.weight_terrain)
    w_t = p.weight_terrain / (p.weight_earth + p.weight_terrain)
    for x, y, sl, r, c, val in zip(xs, ys, slope, row, col, values):
        v = val[0]
        if np.ma.is_masked(v):
            continue
        earth = float(v) * earth_ds.scales[0] + earth_ds.offsets[0]
        if not (0 <= earth <= 1):
            continue
        terrain_score = max(0., 1 - float(sl)/max(p.max_slope, 0.1))
        score = 100 * (w_e * earth + w_t * terrain_score)
        lon, lat = TO_LL.transform(float(x), float(y))
        candidates.append({"lat": round(float(lat),5), "lon": round(float(lon),5),
            "x": round(float(x),1), "y": round(float(y),1),
            "alt_m": round(float(DEM[r,c])*.5,1), "slope_deg": round(float(sl),2),
            "earth_historical_percent": round(earth*100,1), "score": round(score,1)})
    candidates.sort(key=lambda z: z["score"], reverse=True)
    # Empêche 20 points presque identiques d'occuper tout le classement.
    top = []
    for candidate in candidates:
        if all(math.hypot(candidate["x"]-other["x"],candidate["y"]-other["y"]) >= max(step*2, 10000) for other in top):
            top.append(candidate)
            if len(top) == 20:
                break
    return {"sites": top, "sampled_points": int(len(keep.nonzero()[0])),
        "eligible_points": len(candidates), "spacing_km": p.spacing_km, "north_limit_lat": p.north_limit_lat,
        "max_slope": p.max_slope, "scoring_weights": {"earth": round(w_e,3),"terrain": round(w_t,3)},
        "sun_not_scored": True,
        "method": "Présélection exploratoire sur grille LOLA, pente locale et visibilité Terre historique; pas de simulation JPL sur tous les points.",
        "warnings": ["Le Soleil n'entre PAS dans ce classement faute de couverture homogène.",
          "Les pixels entre les points de grille ne sont pas évalués.",
          "La pente LOLA 120 m n'établit pas la sécurité d'un alunissage.",
          "L'ensoleillement et la communication à une date donnée restent à simuler pour chaque candidat."]}

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
