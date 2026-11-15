# Lunar Landing Site Navigator — V18

## Contenu
V17 préservée (modes Débutant/Expert et analyses V1–V16) + tests automatiques, vérification des données, dépendances et procédure de publication. **Ce paquet n'inclut pas les rasters NASA volumineux** ni les fonds graphiques. La publication en ligne n'est pas effectuée.

## Installation sur Windows
1. Faire une copie de sauvegarde du dossier V17.
2. Copier `server.py`, `static/index.html`, `requirements.txt`, `preflight.py` et `tests/` depuis ce ZIP dans le dossier existant `lunar_navigator_v1`. **Conserver** `data/` et les images déjà présentes dans `static/`.
3. Ouvrir PowerShell dans ce dossier, puis exécuter :

```powershell
python -m pip install -r requirements.txt
python preflight.py
python -m unittest discover -s tests -v
python -m uvicorn server:app --reload
```

4. Ouvrir `http://127.0.0.1:8000` et `http://127.0.0.1:8000/docs`.

## Vérification manuelle V18
- [ ] Mode Débutant : guide affiché, coordonnées et site modifiables.
- [ ] Mode Expert : paramètres V13 et V12 accessibles.
- [ ] Changement de mode : ne perd pas les coordonnées ni la simulation déjà calculée.
- [ ] Explorer : carte LOLA et couches Terre/Soleil/PSR chargées.
- [ ] `/api/site` : altitude, pente et couches plausibles pour un site couvert.
- [ ] `/api/simulate` : exécuter sur une période courte ; JPL Horizons requiert Internet et peut être indisponible.
- [ ] V12 : bilan énergétique change lorsque la charge électrique augmente.
- [ ] V13 : modifier les pondérations change le classement, sans modifier les données historiques.
- [ ] V14 : comparaison et export CSV.
- [ ] V15 : atlas des missions, filtres et sélection.
- [ ] V16 : relief 3D, rotation, zoom et exagération 1×/8×.
- [ ] Mobile : navigation et champs lisibles.

## Déploiement public — procédure
1. Choisir un hébergeur capable d'exécuter **FastAPI/Python avec GDAL/rasterio** et de stocker le DEM (~116 Mo) et les TIFF. Une plateforme statique seule (GitHub Pages) ne suffit pas.
2. Placer `data/` sur le disque persistant du service (ou dans l'image de déploiement si la licence et la taille le permettent). Ne pas supposer que le ZIP contient ces fichiers.
3. Copier les images existantes dans `static/`. Configurer l'installation de `requirements.txt` et la commande de démarrage :

```bash
uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}
```

4. Configurer HTTPS, limites de ressources et redémarrage automatique. Éviter de publier le serveur de développement `--reload`.
5. Exécuter `python preflight.py`, les tests et un essai depuis un **autre appareil** avant de partager l'URL.
6. Pour un trafic public, ajouter limitation de débit et cache aux routes coûteuses `/api/events`, `/api/simulate` et `/api/scan` : le prototype actuel n'a pas de protection suffisante contre les appels intensifs.

## Limites scientifiques à afficher au jury
- LOLA : 120 m/pixel dans cette version, ne résout pas les petits obstacles.
- Visibilité historique : moyenne de long terme, **pas** prévision datée.
- Soleil visible : ne garantit pas une production électrique effective.
- Terre visible : ne garantit pas une liaison radio.
- Scores et pentes : présélection exploratoire, **pas** certification d'un site d'alunissage.
- Relief 3D : exagération visuelle, ne pas interpréter les pentes à partir de la vue amplifiée.
- La vérification géodésique indépendante, la validation mission et les tests de charge restent à réaliser.

## Démonstration conseillée (3 minutes)
0:00–0:25 : objectif et région du pôle Sud.
0:25–1:05 : Explorer, sélectionner un site, couches NASA.
1:05–1:45 : simulation Soleil/Terre, limites et bilan énergétique.
1:45–2:20 : classement et comparaison des sites.
2:20–2:45 : relief 3D et modes Débutant/Expert.
2:45–3:00 : limites et prochaines validations.
