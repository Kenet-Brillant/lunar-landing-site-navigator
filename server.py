
"""Lunar Landing Site Navigator.

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
from pyproj import CRS, Transformer, Geod


# ============================================================
# 1. CONFIGURATION GÉNÉRALE
# ============================================================

BASE = Path(__file__).parent

app = FastAPI(title="Lunar Landing Site Navigator")

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
    step: int = Field(default=1, ge=1, le=24)
    horizon_bearings: int = Field(default=144, ge=36, le=360)
    horizon_step_m: int = Field(default=120, ge=60, le=480)
    horizon_distance_km: int = Field(default=35, ge=1, le=60)
    horizon_convergence: bool = False


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

LUNAR_GEOD = Geod(a=R, b=R)


def _dem_bilinear(x, y):
    """Bilinear altitude in meters at DEM pixel centers; None outside coverage."""
    if DEM is None:
        return None
    c = (x + EXTENT) / PIXEL - 0.5
    r = (EXTENT - y) / PIXEL - 0.5
    c0, r0 = math.floor(c), math.floor(r)
    if not (0 <= c0 < DEM.shape[1] - 1 and 0 <= r0 < DEM.shape[0] - 1):
        return None
    fc, fr = c - c0, r - r0
    z = DEM[r0:r0+2, c0:c0+2].astype(float) * 0.5
    return float((1-fr)*((1-fc)*z[0,0]+fc*z[0,1]) + fr*((1-fc)*z[1,0]+fc*z[1,1]))


def _horizon_profile(lat, lon, bearings=144, dist_km=35, step_m=120):
    """Return (horizon angles, diagnostics); missing DEM coverage is never silently accepted."""
    if DEM is None:
        raise HTTPException(status_code=503, detail="DEM LOLA indisponible")
    if not (36 <= bearings <= 360 and 60 <= step_m <= 480 and 1 <= dist_km <= 60):
        raise HTTPException(status_code=422, detail="Paramètres d'horizon hors limites")
    x0, y0 = xy(lat, lon)
    z0 = _dem_bilinear(x0, y0)
    if z0 is None:
        raise HTTPException(status_code=422, detail="Site hors couverture LOLA exploitable")
    angles = []
    incomplete = []
    max_distance_m = int(dist_km * 1000)
    # Sample at the requested spacing and always include the last distance.
    distances = list(range(step_m, max_distance_m + 1, step_m))
    if not distances or distances[-1] != max_distance_m:
        distances.append(max_distance_m)
    for k in range(bearings):
        az = 360.0 * k / bearings
        # Geodesic forward avoids longitude singularities at the south pole.
        lon1, lat1, _ = LUNAR_GEOD.fwd(lon, lat, az, 10.0)
        x1, y1 = xy(lat1, lon1)
        vx, vy = x1-x0, y1-y0
        norm = math.hypot(vx, vy)
        if norm < 1e-10:
            raise HTTPException(status_code=422, detail="Direction géographique indéterminée")
        vx, vy = vx/norm, vy/norm
        peak = -90.0
        for meters in distances:
            z = _dem_bilinear(x0 + vx*meters, y0 + vy*meters)
            if z is None:
                incomplete.append({"azimuth_deg": round(az, 3), "last_valid_distance_m": meters-step_m})
                break
            # Approximate lunar curvature; projected distances approximate ground distances.
            corrected_dz = z - z0 - meters*meters/(2*R)
            peak = max(peak, math.degrees(math.atan2(corrected_dz, meters)))
        angles.append(round(peak, 3) if not incomplete or incomplete[-1]["azimuth_deg"] != round(az, 3) else None)
    if incomplete:
        raise HTTPException(status_code=422, detail={
            "message": "Horizon incomplet : couverture DEM insuffisante pour le rayon demandé",
            "incomplete_directions": len(incomplete),
            "examples": incomplete[:5],
            "suggestion": "Réduire horizon_distance_km ou choisir un autre site"
        })
    return angles, {
        "bearings": bearings, "azimuth_spacing_deg": 360.0/bearings,
        "distance_km": dist_km, "sample_step_m": step_m,
        "dem_native_pixel_m": PIXEL, "interpolation": "bilinear",
        "coverage": "complete", "method": "sampled topographic skyline with approximate lunar curvature",
        "limitations": "DEM 120 m; interpolation does not resolve subpixel obstacles; horizon is approximate"
    }


def horizon(lat, lon, bearings=144, dist_km=35, step_m=120):
    """Compatibility wrapper: frontend expects an array of angles."""
    return _horizon_profile(lat, lon, bearings, dist_km, step_m)[0]


def horizon_convergence(lat, lon, bearings=144, dist_km=35, step_m=120):
    """Compare a profile with twice the angular sampling and finer distance sampling."""
    coarse, info = _horizon_profile(lat, lon, bearings, dist_km, step_m)
    fine_bearings = min(360, bearings*2)
    fine_step = max(60, step_m//2)
    refined, refined_info = _horizon_profile(lat, lon, fine_bearings, dist_km, fine_step)
    # Evaluate refined profile at original azimuths with periodic interpolation.
    differences = []
    for i, base in enumerate(coarse):
        pos = i*fine_bearings/bearings
        j = int(pos)
        f = pos-j
        estimate = refined[j % fine_bearings]*(1-f) + refined[(j+1)%fine_bearings]*f
        differences.append(abs(base-estimate))
    info["convergence"] = {
        "reference_bearings": fine_bearings, "reference_step_m": fine_step,
        "max_abs_difference_deg": round(max(differences), 3),
        "median_abs_difference_deg": round(float(np.median(differences)), 3),
        "p90_abs_difference_deg": round(float(np.percentile(differences,90)), 3),
        "interpretation": "Numerical sampling sensitivity, not measurement uncertainty"
    }
    return coarse, info


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

    try:
        from astroquery.jplhorizons import Horizons
        from astropy.time import Time

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Installer astroquery et astropy : "
                "pip install -r requirements.txt"
            )
        )

    try:
        parsed_dates = [
            datetime.fromisoformat(
                d.replace("Z", "+00:00")
            )
            for d in dates
        ]

        jd = Time(
            parsed_dates,
            scale="utc"
        ).jd.tolist()

        out = {}

        for name, body_id in [
            ("sun", "10"),
            ("earth", "399")
        ]:

            observer = Horizons(
                id=body_id,
                location={
                    "lon": lon,
                    "lat": lat,
                    "elevation": alt_m / 1000,
                    "body": 301
                },
                epochs=jd,
                id_type=None
            )

            rows = observer.ephemerides(
                quantities="4",
                extra_precision=True
            )

            out[name] = [
                {
                    "az": float(row["AZ"]),
                    "el": float(row["EL"])
                }
                for row in rows
            ]

        return out

    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Éphémérides JPL indisponibles : "
                + str(error)[:260]
            )
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



def horizon_elevation(az, h):
    """Elevation de l'horizon dans l'azimut choisi (interpolation circulaire)."""
    if az is None or not h:
        return None
    index = (az % 360) * len(h) / 360.0
    left = int(index) % len(h)
    fraction = index - int(index)
    return h[left] * (1 - fraction) + h[(left + 1) % len(h)] * fraction


def operational_windows(series, horizon_profile):
    """Fenêtres par interpolation linéaire des franchissements d'horizon.

    Les éphémérides ne sont PAS recalculées aux transitions : les heures
    obtenues sont des estimations limitées par le pas temporel initial.
    """
    if not series:
        return {"windows": {}, "next_events": {}, "method": "linear_clearance_interpolation"}

    times = [datetime.fromisoformat(row["time"].replace("Z", "+00:00")) for row in series]
    if any(times[i + 1] <= times[i] for i in range(len(times) - 1)):
        raise ValueError("La série temporelle doit être strictement croissante")

    def clearance(row, body):
        data = row[body]
        skyline = horizon_elevation(data.get("az"), horizon_profile)
        if skyline is None or data.get("el") is None:
            raise ValueError("Éphéméride ou horizon manquant")
        return float(data["el"]) - skyline

    def transitions(values):
        # Liste de segments (début, fin, actif) ; un franchissement est
        # estimé entre deux échantillons par interpolation linéaire.
        edges = [times[0]]
        for i in range(len(values) - 1):
            a, b = values[i], values[i + 1]
            if (a > 0) != (b > 0):
                f = max(0.0, min(1.0, -a / (b - a))) if b != a else 0.5
                crossing = times[i] + (times[i + 1] - times[i]) * f
                if times[0] < crossing < times[-1]:
                    edges.append(crossing)
        edges.append(times[-1])
        edges = sorted(set(edges))
        result = []
        for left, right in zip(edges, edges[1:]):
            midpoint = left + (right - left) / 2
            # Interpoler la clairance au milieu de chaque segment.
            from bisect import bisect_right
            k = min(bisect_right(times, midpoint) - 1, len(times) - 2)
            f = (midpoint - times[k]).total_seconds() / (times[k + 1] - times[k]).total_seconds()
            val = values[k] * (1 - f) + values[k + 1] * f
            result.append((left, right, val > 0))
        return result

    sun = [clearance(row, "sun") for row in series]
    earth = [clearance(row, "earth") for row in series]
    sun_segments = transitions(sun)
    earth_segments = transitions(earth)

    def build(kind):
        if kind == "sun":
            segments = sun_segments
        elif kind == "earth":
            segments = earth_segments
        else:
            edges = sorted(set([times[0], times[-1]] +
                [v for a, b, _ in sun_segments for v in (a, b)] +
                [v for a, b, _ in earth_segments for v in (a, b)]))
            def active_at(parts, t):
                return any(a <= t < b and state for a, b, state in parts)
            segments = [(a, b, active_at(sun_segments, a + (b-a)/2)
                         and active_at(earth_segments, a + (b-a)/2))
                        for a, b in zip(edges, edges[1:])]
        merged = []
        for a, b, active in segments:
            if not active or b <= a:
                continue
            if merged and merged[-1][1] == a:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        items = [{"start": a.isoformat(), "end": b.isoformat(),
                  "duration_hours": round((b-a).total_seconds()/3600, 3),
                  "starts_before_interval": a == times[0],
                  "continues_after_interval": b == times[-1]}
                 for a, b in merged]
        total_seconds = sum((b-a).total_seconds() for a, b in merged)
        span_seconds = (times[-1] - times[0]).total_seconds()
        return {"intervals": items, "count": len(items),
                "total_hours": round(total_seconds/3600, 3),
                "fraction_percent": round(100*total_seconds/span_seconds, 2) if span_seconds else None,
                "longest_hours": round(max((b-a).total_seconds() for a,b in merged)/3600, 3) if merged else 0.0}

    windows = {key: build(key) for key in ("sun", "earth", "both")}
    events = {}
    for key in ("sun", "earth"):
        values = sun if key == "sun" else earth
        segments = sun_segments if key == "sun" else earth_segments
        # Les transitions à l'intérieur de la période, pas les limites artificielles.
        changes = []
        for (a, b, state), (c, d, next_state) in zip(segments, segments[1:]):
            if state != next_state:
                changes.append({"time": b.isoformat(),
                                "type": "rise" if next_state else "set",
                                "hours_from_start": round((b-times[0]).total_seconds()/3600, 3)})
        events[key] = {"initially_visible": values[0] > 0,
                       "transitions": changes,
                       "next_rise": next((e for e in changes if e["type"] == "rise"), None),
                       "next_set": next((e for e in changes if e["type"] == "set"), None)}
    return {"windows": windows, "next_events": events,
            "period_start": times[0].isoformat(), "period_end": times[-1].isoformat(),
            "method": "linear_clearance_interpolation",
            "limitations": [
                "Transitions estimées par interpolation entre éphémérides ; aucune recherche itérative de l'instant exact.",
                "L'erreur dépend du pas temporel et de la variation non linéaire de l'azimut et de l'horizon.",
                "Une fenêtre plus courte que le pas temporel peut être entièrement manquée.",
                "Visibilité du centre des disques uniquement ; aucun bilan de puissance ou de liaison radio.",
                "Les fenêtres touchant les bornes de la simulation peuvent se prolonger hors de la période étudiée."
            ]}

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


# Analyse spatiale exploratoire, à la résolution native du DEM (120 m).
class TerrainArea(Point):
    radius_m: int = Field(default=500, ge=120, le=5000)
    max_slope_deg: float = Field(default=10.0, ge=0, le=90)

@app.post("/api/terrain-area")
def terrain_area(p: TerrainArea):
    """Native DEM-pixel screening; connected components are 4-neighbour groups.

    Pixel-centre inclusion, projected-area estimates, no independent 25 m samples.
    """
    if DEM is None:
        raise HTTPException(status_code=503, detail="DEM LOLA indisponible sur le serveur")
    x0, y0 = xy(p.lat, p.lon)
    radius = float(p.radius_m)
    # A one-pixel halo is needed for central differences.
    cmin = max(1, int(math.floor((x0-radius+EXTENT)/PIXEL))-1)
    cmax = min(7622, int(math.ceil((x0+radius+EXTENT)/PIXEL))+1)
    rmin = max(1, int(math.floor((EXTENT-y0-radius)/PIXEL))-1)
    rmax = min(7622, int(math.ceil((EXTENT-y0+radius)/PIXEL))+1)
    if cmax < cmin or rmax < rmin:
        raise HTTPException(status_code=422, detail="Zone hors couverture LOLA")
    rows = np.arange(rmin, rmax+1)
    cols = np.arange(cmin, cmax+1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    xx = (cc + .5)*PIXEL-EXTENT
    yy = EXTENT-(rr + .5)*PIXEL
    inside = (xx-x0)**2+(yy-y0)**2 <= radius**2
    # Terrain elevation uses 0.5 m per DEM integer count.
    dzdx = (DEM[rr,cc+1].astype(float)-DEM[rr,cc-1].astype(float))*(.5/(2*PIXEL))
    dzdy = (DEM[rr-1,cc].astype(float)-DEM[rr+1,cc].astype(float))*(.5/(2*PIXEL))
    scale = 1+(np.hypot(xx,yy)/(2*R))**2
    slope = np.degrees(np.arctan(np.hypot(dzdx,dzdy)*scale))
    valid = inside & np.isfinite(slope)
    vals = slope[valid]
    if not vals.size:
        raise HTTPException(status_code=422, detail="Aucun centre de pixel LOLA dans le disque demandé : agrandir le rayon")
    good = valid & (slope <= p.max_slope_deg)
    # Four-neighbour connected components, no diagonals.
    seen = np.zeros_like(good, dtype=bool)
    regions = []
    H, W = good.shape
    region_labels = np.zeros((H,W), dtype=int)
    for j in range(H):
        for i in range(W):
            if not good[j,i] or seen[j,i]:
                continue
            stack=[(j,i)]; seen[j,i]=True; pixels=[]
            while stack:
                r,c=stack.pop(); pixels.append((r,c))
                for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr,nc=r+dr,c+dc
                    if 0<=nr<H and 0<=nc<W and good[nr,nc] and not seen[nr,nc]:
                        seen[nr,nc]=True; stack.append((nr,nc))
            js=np.array([v[0] for v in pixels]); ii=np.array([v[1] for v in pixels])
            region_slopes=slope[js,ii]
            region_id=len(regions)+1
            region_labels[js,ii]=region_id
            regions.append({"id":region_id,"pixels":len(pixels),"area_m2_projected":round(len(pixels)*PIXEL**2),
                "width_m_projected":round((ii.max()-ii.min()+1)*PIXEL),
                "height_m_projected":round((js.max()-js.min()+1)*PIXEL),
                "median_slope_deg":round(float(np.median(region_slopes)),2),
                "centroid_dx_m":round(float(np.mean(xx[js,ii]-x0))),
                "centroid_dy_m":round(float(np.mean(yy[js,ii]-y0)))})
    regions.sort(key=lambda r:r["pixels"],reverse=True)
    n=int(vals.size)
    # Classification stability under a user-facing illustrative +/- 2 degree band.
    # This is NOT a measured DEM uncertainty or a confidence interval.
    band=2.0
    uncertain=int(np.count_nonzero(valid & (np.abs(slope-p.max_slope_deg)<=band)))
    ratio=PIXEL/radius
    warnings=[]
    if n<30: warnings.append("Moins de 30 pixels natifs : statistiques peu représentatives.")
    if ratio>=.25: warnings.append("Résolution grossière par rapport au rayon analysé.")
    if uncertain/n>.2: warnings.append("Plus de 20 % des pixels sont proches du seuil (±2° illustratifs).")
    warnings.append("Les pentes sont calculées sur une maille de 120 m ; les obstacles sous-maille restent invisibles.")
    return {
        "lat":p.lat,"lon":p.lon,"radius_m":p.radius_m,"grid_size":W,
        "dem_pixel_m":PIXEL,"sample_spacing_m":PIXEL,
        "threshold_deg":p.max_slope_deg,
        "slope_grid_deg":[[round(float(slope[j,i]),2) if valid[j,i] else None for i in range(W)] for j in range(H)],
        "native_grid":{"rows":H,"cols":W,"pixel_m":PIXEL,"native_pixels":n,
            "origin_dx_m":round(float(xx[0,0]-x0),2),"origin_dy_m":round(float(yy[0,0]-y0),2)},
        "statistics":{"samples":n,"min_deg":round(float(np.min(vals)),2),
            "median_deg":round(float(np.median(vals)),2),
            "p90_deg":round(float(np.percentile(vals,90)),2),
            "max_deg":round(float(np.max(vals)),2),
            "below_threshold_percent":round(100*int(good.sum())/n,1)},
        "region_id_grid":region_labels.tolist(),
        "regions":{"count":len(regions),"largest":regions[0] if regions else None,"top":regions[:5],"all":regions,
            "connectivity":"4-neighbour","area_note":"Aire et dimensions approximatives en coordonnées projetées, non ellipse certifiée."},
        "uncertainty":{"kind":"qualitative_resolution_and_threshold_sensitivity",
            "illustrative_band_deg":band,"near_threshold_pixels":uncertain,
            "near_threshold_percent":round(100*uncertain/n,1),
            "pixel_to_radius_ratio":round(ratio,3),"warnings":warnings,
            "note":"Bande ±2° utilisée uniquement comme test de sensibilité, pas comme incertitude mesurée ni intervalle de confiance."},
        "limitations":"Pixels LOLA natifs ~120 m ; zones contiguës au sens 4-voisins. Pas de détection fiable des rochers, petits cratères, portance ou stabilité du sol."
    }

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

    # Pas temporel entier, compatible avec les choix de l’interface (1, 3, 6, 12 h).
    exact_end = start + timedelta(hours=p.hours)
    n_steps = math.floor(p.hours / p.step)
    dates = [
        start + timedelta(hours=i * p.step)
        for i in range(n_steps + 1)
    ]
    # Garantir que la simulation se termine exactement à l'heure demandée.
    if dates[-1] < exact_end:
        dates.append(exact_end)
    elif dates[-1] > exact_end:
        dates[-1] = exact_end

    # Horizon du site
    h, horizon_info = (
        horizon_convergence(p.lat, p.lon, p.horizon_bearings, p.horizon_distance_km, p.horizon_step_m)
        if p.horizon_convergence else
        _horizon_profile(p.lat, p.lon, p.horizon_bearings, p.horizon_distance_km, p.horizon_step_m)
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
        "horizon_info": horizon_info,

        "operational_windows": operational_windows(series, h),

        "series": series,

        "summary": {
            "sun_percent": sun_percent,
            "earth_percent": earth_percent,
            "both_percent": both_percent
        },

        "notes": [
            "Estimations échantillonnées, pas de temps choisi par utilisateur",
            "Horizon LOLA ~120 m/pixel, interpolation bilinéaire et paramètres dans horizon_info",
            "Visibilité du centre des disques; pas de modélisation de liaison radio",
            "Pourcentage historique sur les instants échantillonnés; operational_windows donne une estimation temporelle interpolée",
            "Pente locale estimée à partir du DEM LOLA; sécurité d'atterrissage non évaluée"
        ]
    }


# ============================================================
# COMPARAISON INTÉGRÉE SITE–DATE
# ============================================================

class CompareRequest(BaseModel):
    a: Sim
    b: Sim


def _scenario_summary(result):
    site_data = result["site"]
    windows = result["operational_windows"]["windows"]
    return {
        "lat": site_data["lat"], "lon": site_data["lon"],
        "date_start": result["series"][0]["time"],
        "date_end": result["series"][-1]["time"],
        "alt_m": site_data["alt_m"],
        "slope_deg": site_data["slope_deg"],
        "historical_layers": site_data["layers"],
        "sampled_percent": result["summary"],
        "windows": {key: {
            "total_hours": windows[key]["total_hours"],
            "longest_hours": windows[key]["longest_hours"],
            "fraction_percent": windows[key]["fraction_percent"],
            "intervals": windows[key]["intervals"],
        } for key in ("sun", "earth", "both")},
        "horizon_info": result["horizon_info"],
    }


@app.post("/api/compare-scenarios")
def compare_scenarios(request: CompareRequest):
    """Compare deux couples site/date sur la même durée et au même pas temporel."""
    a, b = request.a, request.b
    if a.hours != b.hours or a.step != b.step:
        raise HTTPException(status_code=422, detail="Durée et pas temporel doivent être identiques pour comparer les scénarios.")
    left = _scenario_summary(simulate(a))
    right = _scenario_summary(simulate(b))
    def difference(metric):
        x, y = metric(left), metric(right)
        return round(y-x, 3) if x is not None and y is not None else None
    return {
        "a": left, "b": right,
        "comparison_type": "same_site_different_dates" if (a.lat == b.lat and a.lon == b.lon) else "different_sites",
        "delta_b_minus_a": {
            "slope_deg": difference(lambda s: s["slope_deg"]),
            "sun_hours": difference(lambda s: s["windows"]["sun"]["total_hours"]),
            "earth_hours": difference(lambda s: s["windows"]["earth"]["total_hours"]),
            "both_hours": difference(lambda s: s["windows"]["both"]["total_hours"]),
            "longest_both_hours": difference(lambda s: s["windows"]["both"]["longest_hours"]),
        },
        "limitations": [
            "Fenêtres estimées par interpolation entre échantillons ; transitions courtes potentiellement manquées.",
            "Énergie : disponibilité solaire géométrique, sans puissance photovoltaïque ni batterie.",
            "Communications : visibilité géométrique de la Terre, sans bilan radio.",
            "Pente LOLA ~120 m : ne certifie pas la sécurité de l'atterrissage.",
            "La comparaison ne fournit pas de score global arbitraire ; le compromis dépend de la mission.",
        ]
    }



# ============================================================
# AIDE À LA DÉCISION ET RECHERCHE DE DATES
# ============================================================

class DecisionCriteria(BaseModel):
    min_sun_hours: float = Field(default=24, ge=0, le=336)
    min_earth_hours: float = Field(default=12, ge=0, le=336)
    min_both_hours: float = Field(default=8, ge=0, le=336)
    max_slope_deg: float = Field(default=10, ge=0, le=90)

class DecisionRequest(BaseModel):
    a: Sim
    b: Sim
    criteria: DecisionCriteria = Field(default_factory=DecisionCriteria)


def _decision_evaluation(s, criteria):
    windows = s["windows"]
    checks = {
        "sun": {"actual": windows["sun"]["total_hours"], "required": criteria.min_sun_hours, "unit": "h"},
        "earth": {"actual": windows["earth"]["total_hours"], "required": criteria.min_earth_hours, "unit": "h"},
        "both": {"actual": windows["both"]["total_hours"], "required": criteria.min_both_hours, "unit": "h"},
        "slope": {"actual": s["slope_deg"], "required": criteria.max_slope_deg, "unit": "deg"},
    }
    for name, check in checks.items():
        check["passed"] = None if check["actual"] is None else (
            check["actual"] <= check["required"] if name == "slope" else check["actual"] >= check["required"]
        )
    status = "indeterminate" if any(c["passed"] is None for c in checks.values()) else (
        "meets_screening_criteria" if all(c["passed"] for c in checks.values()) else "does_not_meet_criteria"
    )
    return {"status": status, "checks": checks, "failed": [k for k,v in checks.items() if v["passed"] is False],
            "unknown": [k for k,v in checks.items() if v["passed"] is None]}


@app.post("/api/decision")
def decision(request: DecisionRequest):
    compared = compare_scenarios(CompareRequest(a=request.a, b=request.b))
    a = _decision_evaluation(compared["a"], request.criteria)
    b = _decision_evaluation(compared["b"], request.criteria)
    return {"a": {**compared["a"], "evaluation": a},
            "b": {**compared["b"], "evaluation": b},
            "delta_b_minus_a": compared["delta_b_minus_a"],
            "criteria": request.criteria.model_dump() if hasattr(request.criteria,"model_dump") else request.criteria.dict(),
            "conclusion": ("Les deux scénarios satisfont les seuils exploratoires." if a["status"] == b["status"] == "meets_screening_criteria" else
                           "A seul satisfait les seuils exploratoires." if a["status"] == "meets_screening_criteria" and b["status"] == "does_not_meet_criteria" else
                           "B seul satisfait les seuils exploratoires." if b["status"] == "meets_screening_criteria" and a["status"] == "does_not_meet_criteria" else
                           "Aucun des deux scénarios ne satisfait tous les seuils, ou certaines données manquent."),
            "limitations": compared["limitations"] + ["Les seuils sont choisis par l'utilisateur ; leur satisfaction ne prouve pas la sécurité d'alunissage."]}


class WindowSearchRequest(Point):
    start_date: str
    search_days: int = Field(default=7, ge=1, le=14)
    mission_hours: int = Field(default=72, ge=6, le=168)
    step_hours: int = Field(default=6, ge=3, le=12)
    criteria: DecisionCriteria = Field(default_factory=DecisionCriteria)


@app.post("/api/search-windows")
def search_windows(p: WindowSearchRequest):
    """Recherche sur une seule grille d'éphémérides, sans répéter les appels JPL."""
    try:
        start = datetime.fromisoformat(p.start_date.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=422, detail="Date de début ISO invalide")
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    start = start.astimezone(timezone.utc)
    if p.criteria.min_sun_hours > p.mission_hours or p.criteria.min_earth_hours > p.mission_hours or p.criteria.min_both_hours > p.mission_hours:
        raise HTTPException(status_code=422, detail="Un seuil horaire dépasse la durée de mission")
    h, h_info = _horizon_profile(p.lat, p.lon)
    s = site(Point(lat=p.lat, lon=p.lon))
    if s["alt_m"] is None:
        raise HTTPException(status_code=422, detail="Altitude LOLA indisponible")
    # Candidate starts every step_hours; every window ends at the exact requested duration.
    offsets = list(range(0, p.search_days*24+1, p.step_hours))
    all_offsets = set()
    for offset in offsets:
        for hour in range(0, p.mission_hours+1, p.step_hours):
            all_offsets.add(offset+hour)
        all_offsets.add(offset+p.mission_hours)
    times = sorted(all_offsets)
    dates = [start+timedelta(hours=i) for i in times]
    raw = eph(p.lat, p.lon, s["alt_m"], [d.isoformat() for d in dates])
    rows = [{"time": d.isoformat(), "sun": {**raw["sun"][i]}, "earth": {**raw["earth"][i]}} for i,d in enumerate(dates)]
    index = {t:i for i,t in enumerate(times)}
    candidates = []
    for offset in offsets:
        selected = [offset+v for v in range(0,p.mission_hours+1,p.step_hours)]
        if selected[-1] != offset+p.mission_hours:
            selected.append(offset+p.mission_hours)
        subset = [rows[index[t]] for t in selected]
        w = operational_windows(subset, h)["windows"]
        summary = {k: w[k]["total_hours"] for k in ("sun","earth","both")}
        checks = {
            "sun": summary["sun"] >= p.criteria.min_sun_hours,
            "earth": summary["earth"] >= p.criteria.min_earth_hours,
            "both": summary["both"] >= p.criteria.min_both_hours,
            "slope": s["slope_deg"] is not None and s["slope_deg"] <= p.criteria.max_slope_deg,
        }
        candidates.append({"start":subset[0]["time"], "end":subset[-1]["time"],
                           "hours":summary,"longest_both_hours":w["both"]["longest_hours"],
                           "meets_criteria":all(checks.values()), "checks":checks})
    # Rank compliant periods first, then simultaneous visibility, then sun and earth.
    ranked = sorted(candidates, key=lambda c:(c["meets_criteria"], c["hours"]["both"], c["hours"]["sun"], c["hours"]["earth"]), reverse=True)
    return {"site": {"lat":p.lat,"lon":p.lon,"slope_deg":s["slope_deg"]},
            "searched_starts":len(candidates), "matching_starts":sum(c["meets_criteria"] for c in candidates),
            "best":ranked[:10], "criteria": p.criteria.model_dump() if hasattr(p.criteria,"model_dump") else p.criteria.dict(),
            "method": "single JPL ephemeris grid; sliding windows with interpolated horizon crossings",
            "horizon_info":h_info,
            "limitations": ["Départs explorés uniquement aux pas indiqués ; pas de recherche continue du meilleur instant.",
                            "Événements interpolés ; les fenêtres courtes peuvent être manquées.",
                            "Soleil et Terre visibles géométriquement ; ni énergie réelle, ni budget de liaison radio.",
                            "Pente LOLA 120 m ; aucun contrôle des dangers sous-pixel ni certification d'alunissage."]}

# ============================================================
# 13. DÉMARRAGE
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
