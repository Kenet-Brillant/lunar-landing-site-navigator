# Lunar Landing Site Navigator — V1 P0

Prototype web **local** en français pour comparer des sites dans la région **75°S–90°S** de la Lune.

## Démarrage

1. Installer **Python 3.10+**.
2. Ouvrir un terminal dans le dossier `lunar_navigator_v1`.
3. Exécuter `python -m pip install -r requirements.txt`.
4. Exécuter `python server.py`.
5. Ouvrir **http://127.0.0.1:8000** dans le navigateur.

Une connexion Internet est requise pour les éphémérides JPL Horizons. Le DEM et les couches NASA incluses sont disponibles hors ligne.

## Fonctionnalités

- Carte du pôle Sud construite à partir du DEM LOLA 120 m/pixel, coordonnées personnalisées et sélection sur carte.
- Couches NASA : visibilité terrestre moyenne (75°S–90°S), éclairement solaire moyen et ombres permanentes (85°S–90°S).
- Lecture des altitudes réelles depuis le fichier brut LOLA (signé 16 bits, facteur 0,5 m).
- Simulation temporelle Soleil/Terre avec astroquery JPL Horizons (si service accessible).
- Profil d'horizon calculé depuis le DEM sur 35 km, 72 azimuts et échantillonnage de 240 m.
- Visibilité du **centre** du Soleil et de la Terre, courbes, tableau chronologique, fractions d'échantillons visibles, comparaison des sites.
- Indication de la prochaine transition visible→masqué détectée à la résolution temporelle choisie.

## Limitations scientifiques et techniques importantes

- **Version expérimentale** : les calculs d'horizon, la projection et les conventions d'azimut JPL doivent être comparés à des cas de référence avant usage scientifique. La courbure lunaire est approchée localement, le rayon de calcul de l'horizon est limité à 35 km, les effets de disque solaire/terrestre sont ignorés.
- Les moyennes historiques NASA **ne décrivent pas la date choisie**.
- Les valeurs d'énergie ne sont **pas** des estimations de puissance électrique ; la visibilité terrestre ne garantit **pas** la qualité d'une liaison radio.
- La « durée » calculée par le site correspond à des **fractions d'instants échantillonnés** ; les transitions peuvent survenir entre deux échantillons.
- Les points A/B/C sont des **points d'essai**, pas des sites approuvés par la NASA.
- Les sites Apollo et la couverture mondiale sont hors périmètre de cette V1 polaire.
- Les appels Horizons peuvent être lents ou échouer selon le réseau, la disponibilité du service ou ses contraintes.
- Le moteur Horizons utilise la position de surface de la Lune (body 301) et l'altitude relative à la sphère de référence ; vérifier les conventions pour toute étude avancée.

## Provenance des données

- LOLA LDEM_75S_120M (Lunar Reconnaissance Orbiter), fichier binaire fourni par l'utilisateur.
- NASA Goddard PGDA Lunar Polar Illumination : `AVGVISIB_75S_120M_201608_EARTH.tiff`, `AVGVISIB_85S_060M_201608.tiff`, `LPSR_85S_060M_201608.tiff`.
- JPL Horizons via astroquery pour les positions temporelles.

## Étape suivante

Tester un appel Horizons réel, vérifier la convention de coordonnées et valider les profils d'horizon sur plusieurs emplacements avec des données de référence indépendantes. Ajouter ensuite un catalogue vérifié de régions candidates.
