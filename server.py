
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
    "sun": "AVGVISIB_75S_120M_201608.tiff",
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
    max_slope: float = Field(default=10, ge=0.1, le=45)
    weight_earth: float = Field(default=33, ge=0, le=100)
    weight_terrain: float = Field(default=34, ge=0, le=100)
    weight_sun: float = Field(default=33, ge=0, le=100)
    exclude_psr: bool = True

@app.post("/api/scan")
def scan_pole(p: ScanRequest):
    """Classement exploratoire à 3 critères, avec contrôle des données absentes."""
    if DEM is None:
        raise HTTPException(status_code=503, detail="DEM LOLA absent")
    if p.weight_earth + p.weight_terrain + p.weight_sun <= 0:
        raise HTTPException(status_code=422, detail="Choisir au moins un poids positif")
    datasets = {key: layer(key) for key in ("sun", "earth", "psr")}
    weights = {"sun": p.weight_sun, "earth": p.weight_earth, "terrain": p.weight_terrain}
    # Ne jamais redistribuer silencieusement un poids demandé vers les autres critères.
    missing = [key for key in ("sun", "earth") if weights[key] > 0 and datasets[key] is None]
    if missing:
        raise HTTPException(status_code=503, detail="Données requises absentes : " + ", ".join(missing))
    if p.exclude_psr and datasets["psr"] is None:
        raise HTTPException(status_code=503, detail="Couche PSR absente : désactiver explicitement le filtre pour continuer")
    step = p.spacing_km * 1000.0
    max_r = 2 * R * math.tan(math.radians(90 + p.north_limit_lat) / 2)
    axis = np.arange(-max_r, max_r + 1, step)
    gx, gy = np.meshgrid(axis, axis)
    xs, ys = gx.ravel(), gy.ravel()
    inside = xs * xs + ys * ys <= max_r * max_r
    xs, ys = xs[inside], ys[inside]
    sampled = len(xs)
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
    coords = list(zip(xs.tolist(), ys.tolist()))
    def samples(key):
        ds = datasets[key]
        if ds is None:
            return [None] * len(coords)
        # Rasterio's sample returns raw stored values; apply scale and offset.
        result = []
        for arr in ds.sample(coords, masked=True):
            v = arr[0]
            if np.ma.is_masked(v) or not np.isfinite(float(v)):
                result.append(None)
            else:
                result.append(float(v) * ds.scales[0] + ds.offsets[0])
        return result
    sun_values = samples("sun") if weights["sun"] > 0 else [None] * len(coords)
    earth_values = samples("earth") if weights["earth"] > 0 else [None] * len(coords)
    psr_values = samples("psr") if p.exclude_psr else [None] * len(coords)
    weight_total = sum(weights.values())
    normalized = {k: v/weight_total for k, v in weights.items()}
    candidates = []
    excluded_psr = 0
    missing_coverage = 0
    for i, (x, y, sl, r, c) in enumerate(zip(xs, ys, slope, row, col)):
        sun, earth, psr = sun_values[i], earth_values[i], psr_values[i]
        if (weights["sun"] > 0 and (sun is None or not 0 <= sun <= 1)) or (weights["earth"] > 0 and (earth is None or not 0 <= earth <= 1)):
            missing_coverage += 1
            continue
        if p.exclude_psr:
            if psr is None:
                missing_coverage += 1
                continue
            if psr >= .5:
                excluded_psr += 1
                continue
        components = {"sun": (sun*100 if sun is not None else None),
                      "earth": (earth*100 if earth is not None else None),
                      "terrain": 100*max(0., 1-float(sl)/p.max_slope)}
        contributions = {k: round((components[k] or 0)*normalized[k], 2) for k in weights}
        score = sum(contributions.values())
        lon, lat = TO_LL.transform(float(x), float(y))
        candidates.append({"lat": round(float(lat),5), "lon": round(float(lon),5),
            "x": round(float(x),1), "y": round(float(y),1),
            "alt_m": round(float(DEM[r,c])*.5,1), "slope_deg": round(float(sl),2),
            "sun_historical_percent": round(sun*100,1) if sun is not None else None,
            "earth_historical_percent": round(earth*100,1) if earth is not None else None,
            "terrain_score": round(components["terrain"],1),
            "score_components": {k: (round(v,1) if v is not None else None) for k,v in components.items()},
            "score_contributions": contributions, "score": round(score,1),
            "psr_excluded": p.exclude_psr})
    candidates.sort(key=lambda z: z["score"], reverse=True)
    top = []
    for candidate in candidates:
        if all(math.hypot(candidate["x"]-other["x"],candidate["y"]-other["y"]) >= max(step*2,10000) for other in top):
            top.append(candidate)
            if len(top) == 20:
                break
    warnings = ["Classement de présélection, pas une certification d'alunissage.",
        "Le Soleil et la Terre sont des moyennes historiques, pas des prévisions pour la date de mission.",
        "Pente locale LOLA 120 m/pixel : obstacles et risques à petite échelle non évalués.",
        "Les zones entre points de grille ne sont pas analysées."]
    if not p.exclude_psr:
        warnings.append("Filtre PSR désactivé : des zones d'ombre permanente peuvent figurer parmi les candidats.")
    return {"sites": top, "sampled_points": sampled, "eligible_points": len(candidates),
        "excluded_psr_points": excluded_psr, "missing_coverage_points": missing_coverage,
        "spacing_km": p.spacing_km, "north_limit_lat": p.north_limit_lat,
        "max_slope": p.max_slope, "scoring_weights": {k: round(v,3) for k,v in normalized.items()},
        "sun_not_scored": weights["sun"] == 0,
        "method": "Classement pondéré Soleil/Terre/terrain sur grille LOLA ; exclusion PSR optionnelle ; sans simulation JPL par candidat.",
        "warnings": warnings}


# V8: carte topographique polaire géoréférencée
import io
from fastapi.responses import Response
from PIL import Image

POLAR_EXT = 2 * R * math.tan(math.radians(2.5))

@lru_cache(maxsize=1)
def build_polar_map():
    if DEM is None:
        raise HTTPException(status_code=503, detail="DEM LOLA absent")
    n = 768
    a = (np.arange(n) + .5) * (2 * POLAR_EXT / n) - POLAR_EXT
    xx, yy = np.meshgrid(a, a[::-1])
    valid = np.hypot(xx, yy) <= POLAR_EXT
    col = np.clip(np.floor((xx + EXTENT) / PIXEL).astype(int), 1, 7622)
    row = np.clip(np.floor((EXTENT - yy) / PIXEL).astype(int), 1, 7622)
    z = DEM[row, col].astype(np.float32) * .5
    gy, gx = np.gradient(z, 2 * POLAR_EXT / n)
    lo, hi = np.percentile(z[valid], [3, 97])
    v = np.clip((z - lo) / max(hi - lo, 1), 0, 1)
    shade = np.clip(.8 - .5 * gx + .35 * gy, .3, 1.2)
    lum = np.clip((45 + 165 * v) * shade, 0, 255)
    rgb = np.zeros((n, n, 3), dtype=np.uint8)
    for channel, factor, outside in [(0,.92,4),(1,.97,13),(2,1.05,27)]:
        rgb[:,:,channel] = np.where(valid, np.clip(lum * factor, 0, 255), outside).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(output, format="PNG")
    return output.getvalue()

@app.get("/api/polar-map")
def polar_map():
    return Response(build_polar_map(), media_type="image/png")


# V11 — couches scientifiques reprojetées sur la même grille polaire que LOLA
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds

V11_LAYERS = {
    "earth": ("AVGVISIB_75S_120M_201608_EARTH.tiff", "Visibilité historique de la Terre", "%"),
    "sun": ("AVGVISIB_75S_120M_201608.tiff", "Éclairement solaire historique", "%"),
    "psr": ("LPSR_85S_060M_201608.tiff", "Régions d’ombre permanente", "classe"),
}

@app.get("/api/map-layer/{kind}")
def v11_map_layer(kind: str):
    if kind not in ("slope", *V11_LAYERS):
        raise HTTPException(status_code=404, detail="Couche inconnue")
    return Response(_v11_png(kind), media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})

@lru_cache(maxsize=4)
def _v11_png(kind: str):
    n = 768
    pix = 2 * POLAR_EXT / n
    a = (np.arange(n) + .5) * pix - POLAR_EXT
    xx, yy = np.meshgrid(a, a[::-1])
    disk = np.hypot(xx, yy) <= POLAR_EXT
    rgba = np.zeros((n, n, 4), dtype=np.uint8)
    if kind == "slope":
        if DEM is None:
            raise HTTPException(status_code=503, detail="DEM absent")
        c = np.floor((xx + EXTENT) / PIXEL).astype(int)
        r = np.floor((EXTENT - yy) / PIXEL).astype(int)
        valid = disk & (c >= 1) & (c < 7623) & (r >= 1) & (r < 7623)
        c = np.clip(c, 1, 7622); r = np.clip(r, 1, 7622)
        dzdx = (DEM[r,c+1].astype(np.float32)-DEM[r,c-1].astype(np.float32)) * .5 / (2*PIXEL)
        dzdy = (DEM[r-1,c].astype(np.float32)-DEM[r+1,c].astype(np.float32)) * .5 / (2*PIXEL)
        v = np.degrees(np.arctan(np.hypot(dzdx,dzdy)))
        t = np.clip(v/20., 0, 1)
        rgba[:,:,0] = (55+200*t).astype('uint8')
        rgba[:,:,1] = (205-140*t).astype('uint8')
        rgba[:,:,2] = (140-100*t).astype('uint8')
    else:
        filename,_,_ = V11_LAYERS[kind]
        path = BASE/'data'/filename
        if not path.exists():
            raise HTTPException(status_code=503, detail=f"Fichier manquant : {filename}")
        with rasterio.open(path) as src:
            raw = np.full((n,n), -32768, dtype=np.float32)
            target = from_bounds(-POLAR_EXT,-POLAR_EXT,POLAR_EXT,POLAR_EXT,n,n)
            reproject(source=rasterio.band(src,1),destination=raw,src_transform=src.transform,
                src_crs=src.crs,dst_transform=target,dst_crs=PROJ,
                src_nodata=src.nodata,dst_nodata=-32768,resampling=Resampling.nearest)
            valid = disk & np.isfinite(raw) & (raw != -32768)
            val = raw * src.scales[0] + src.offsets[0]
        if kind == 'psr':
            # 0 = non-PSR, 1 = PSR ; afficher uniquement les pixels PSR
            valid &= val >= .5
            rgba[:,:,:3] = [180, 75, 245]
        else:
            t = np.clip(val,0,1)
            if kind == 'earth':
                rgba[:,:,0] = (35+35*t).astype('uint8')
                rgba[:,:,1] = (85+145*t).astype('uint8')
                rgba[:,:,2] = (160+85*t).astype('uint8')
            else:
                rgba[:,:,0] = (110+145*t).astype('uint8')
                rgba[:,:,1] = (50+160*t).astype('uint8')
                rgba[:,:,2] = (35+55*t).astype('uint8')
    rgba[:,:,3] = np.where(valid, 220, 0).astype('uint8')
    out=io.BytesIO();Image.fromarray(rgba,'RGBA').save(out,format='PNG')
    return out.getvalue()

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )


# V16 — fenêtre locale LOLA en projection stéréographique, grille décimée.
@app.get("/api/terrain-3d")
def terrain_3d(lat: float = -86, lon: float = 0, size_km: float = 12, n: int = 65):
    if DEM is None:
        raise HTTPException(status_code=503, detail="DEM LOLA absent : fichier ldem_75s_120m.img requis")
    if not (-90 <= lat <= -85) or not (-180 <= lon <= 180):
        raise HTTPException(status_code=422, detail="La V16 couvre uniquement 85°S–90°S")
    if not (2 <= size_km <= 30) or not (25 <= n <= 101):
        raise HTTPException(status_code=422, detail="Taille ou résolution hors limites")
    x, y = xy(lat, lon)
    offsets = np.linspace(-size_km * 500, size_km * 500, n)
    xx = x + offsets[None, :]
    yy = y + offsets[::-1, None]
    cols = np.floor((xx + EXTENT) / PIXEL).astype(int)
    rows = np.floor((EXTENT - yy) / PIXEL).astype(int)
    rows, cols = np.broadcast_arrays(rows, cols)
    valid = (rows >= 0) & (rows < 7624) & (cols >= 0) & (cols < 7624)
    z = np.full((n,n), np.nan, dtype=np.float32)
    z[valid] = DEM[rows[valid], cols[valid]].astype(np.float32) * .5
    # Hors bande 85°S–90°S : ne pas inventer de relief.
    polar_r = 2 * R * math.tan(math.radians(2.5))
    coverage = np.hypot(np.broadcast_to(xx,(n,n)), np.broadcast_to(yy,(n,n))) <= polar_r
    z[~coverage] = np.nan
    if not np.isfinite(z).any():
        raise HTTPException(status_code=422, detail="Aucune donnée LOLA dans cette fenêtre")
    zmin = float(np.nanmin(z)); zmax = float(np.nanmax(z))
    return {"lat":lat,"lon":lon,"size_km":size_km,"n":n,
            "step_m":size_km*1000/(n-1),"min_m":round(zmin,1),"max_m":round(zmax,1),
            "valid_fraction":round(float(np.isfinite(z).mean()),3),
            "heights": [[round(float(v),1) if np.isfinite(v) else None for v in row] for row in z]}
