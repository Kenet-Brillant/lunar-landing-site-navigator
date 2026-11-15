
"""
Lunar Landing Site Navigator
Moteur de classement multicritère des sites lunaires.

Ce module classe les sites à partir de scores déjà calculés.
Il ne calcule pas lui-même les éphémérides ou la topographie.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ============================================================
# 1. MODÈLES DE DONNÉES
# ============================================================

class Weights(BaseModel):
    solar: float = Field(default=1.0, ge=0)
    earth: float = Field(default=1.0, ge=0)
    terrain: float = Field(default=1.0, ge=0)


class SiteScores(BaseModel):
    name: str
    lat: float
    lon: float

    solar: Optional[float] = Field(default=None, ge=0, le=100)
    earth: Optional[float] = Field(default=None, ge=0, le=100)
    terrain: Optional[float] = Field(default=None, ge=0, le=100)


# ============================================================
# 2. CALCUL DU SCORE GLOBAL
# ============================================================

def calculate_score(site: SiteScores, weights: Weights):
    """
    Calcule une moyenne pondérée des trois critères.

    Un score global n'est disponible que si tous les
    critères ayant un poids positif sont renseignés.
    """

    criteria = {
        "solar": (site.solar, weights.solar),
        "earth": (site.earth, weights.earth),
        "terrain": (site.terrain, weights.terrain)
    }

    total_weight = sum(
        weight for _, weight in criteria.values()
    )

    if total_weight <= 0:
        return None

    for value, weight in criteria.values():
        if weight > 0 and value is None:
            return None

    score = sum(
        value * weight
        for value, weight in criteria.values()
        if weight > 0
    ) / total_weight

    return round(score, 2)


# ============================================================
# 3. CLASSEMENT DES SITES
# ============================================================

def rank_sites(sites, weights=None):
    """
    Classe les sites par score décroissant.

    Les sites ayant des données insuffisantes
    apparaissent en fin de classement.
    """

    if weights is None:
        weights = Weights()

    results = []

    for site in sites:
        score = calculate_score(site, weights)

        results.append({
            "name": site.name,
            "lat": site.lat,
            "lon": site.lon,
            "solar": site.solar,
            "earth": site.earth,
            "terrain": site.terrain,
            "global_score": score,
            "status": (
                "evaluated"
                if score is not None
                else "insufficient_data"
            )
        })

    results.sort(
        key=lambda item: (
            item["global_score"] is None,
            -item["global_score"]
            if item["global_score"] is not None
            else 0
        )
    )

    for index, result in enumerate(results):
        result["rank"] = (
            index + 1
            if result["global_score"] is not None
            else None
        )

    return results


# ============================================================
# 4. TESTS DE DÉMONSTRATION
# ============================================================

if __name__ == "__main__":

    # Scores fictifs : uniquement pour tester le classement.
    # Ils ne représentent pas des mesures lunaires.

    sites = [
        SiteScores(
            name="Site A",
            lat=-85.0,
            lon=0.0,
            solar=95,
            earth=35,
            terrain=70
        ),
        SiteScores(
            name="Site B",
            lat=-85.2,
            lon=10.0,
            solar=78,
            earth=82,
            terrain=85
        ),
        SiteScores(
            name="Site C",
            lat=-86.0,
            lon=20.0,
            solar=55,
            earth=95,
            terrain=75
        )
    ]

    print("\nCLASSEMENT EQUILIBRE")
    print("--------------------")

    for site in rank_sites(sites):
        print(
            site["rank"],
            site["name"],
            site["global_score"]
        )

    print("\nCLASSEMENT PRIORITE SOLEIL")
    print("--------------------------")

    solar_weights = Weights(
        solar=0.8,
        earth=0.1,
        terrain=0.1
    )

    for site in rank_sites(sites, solar_weights):
        print(
            site["rank"],
            site["name"],
            site["global_score"]
        )
