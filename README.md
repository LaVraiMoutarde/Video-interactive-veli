# Prototype interactif Webcam -> Vidéo (OpenCV & MediaPipe)

Ce prototype permet de contrôler la timeline d'une vidéo avec son placement physique par rapport à la caméra, et de changer de vidéo grâce aux gestes de la main.

## Interactions
- **Pre-phase guidee (avant demarrage)** : une interface claire affiche les etapes a suivre (presence, zone distance, 2 mains levees, maintien du geste).
- **Lancement vidéo 1 (2 mains levées)** : autorisé uniquement entre **3.0 m et 3.7 m**.
  - Le geste doit etre maintenu pendant **3 secondes** avec une barre de progression.
  - Au-dela de 3.7 m : lancement refuse (trop loin).
  - En-dessous de 3.0 m : lancement refuse (trop proche).
  - L'indicateur visuel suit la zone stabilisee (hysteresis), pas seulement le seuil brut.
- **Scrubbing spatial** : plus vous vous approchez de la webcam, plus la vidéo avance. Plus vous vous reculez, plus elle retourne vers le début.
  *L'interaction est gérée par la distance entre vos deux épaules, lissée via un filtre (EMA) pour un effet fluide et stable.*
- **Bascule vers vidéo 2 (main droite levée)** : prise en compte uniquement dans la fenetre distance configurée.
  - Le geste est valide apres maintien avec barre de progression dediee.
  - En-dessous de **1.2 m**, le trigger est ignore.
  - La barre de progression et le badge de zone suivent l'etat stabilise de la logique.
- **Robustesse bords de zones** : une hysteresis est appliquee sur la fenetre de lancement et sur la fenetre du trigger video 2 pour eviter les oscillations a la limite.
- **Perte de tracking** : countdown de **2 secondes**. Si l'utilisateur ne revient pas, l'experience revient au debut (attente + debut vidéo 1).
- **Mono-utilisateur** : un seul utilisateur est verrouille par session.

## Installation

Dans un terminal, placez-vous dans ce dossier :
```bash
cd "c:\Users\Admin\Documents\ecole\projet véli\video_interaction"
```

Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Démarrage

Au lancement, un menu de demarrage type "jeu video" permet de choisir rapidement:
- Lancer l'experience
- Mode calibration
- Test camera
- Quitter

Navigation menu:
- `W/S` ou fleches haut/bas: naviguer
- `Entree` ou `Espace`: valider
- `Q`: quitter

**1. Avoir des vidéos dans le dossier :**
Le script s'attend à lire `video 1.mp4` et `video 2.mp4`.
Si vous n'en avez pas, vous pouvez générer rapidement deux vidéos de démonstration avec :
```bash
python generate_dummy_videos.py
```

**2. Lancer l'application :**
```bash
python main.py
```

Au demarrage, un ecran d'initialisation est affiche pendant au moins 3 secondes avec le texte `initation de l'idee`.
Cela masque le chargement des videos lourdes avant l'entree dans l'experience.

Au premier lancement, un fichier `calibration.json` est cree automatiquement.
Les valeurs de calibration sont rechargees a chaque demarrage.

Raccourci clavier de l'app :
- **q** : Quitter le programme.
- **c** : Sauvegarder la calibration courante dans `calibration.json`.
- **k** : Basculer en mode calibration visuel (fichier separe `src/calibration_mode.py`).
- **b** : Sortir des videos et revenir au mode detection.
- **f** : Basculer plein ecran/fenetre pour les 2 fenetres runtime.

## Affichage double ecran (PC debug + projecteur clean)
- Le runtime ouvre maintenant **2 fenetres**:
  - `Projector Experience` : rendu clean pour le projecteur.
  - `Debug Monitor` : overlays techniques (etat, distance, skeleton, commandes) pour le PC.
- Les deux fenetres se lancent en **plein ecran**.
- Positions d'ecran configurables dans `calibration.json`:
  - `debug_monitor_x`, `debug_monitor_y`
  - `projector_monitor_x`, `projector_monitor_y`

Exemple courant sous Windows (ecran PC a gauche, projecteur a droite):
```json
"debug_monitor_x": 0,
"debug_monitor_y": 0,
"projector_monitor_x": 1920,
"projector_monitor_y": 0
```

## Mode calibration visuel
- **Objectif** : regler les distances/seuils directement sur l'espace d'installation avec retour camera + jauge des zones.
- **Touches** :
  - `[` / `]` : parametre precedent/suivant
  - `+` / `-` : ajustement fin
  - `{` / `}` : ajustement large
  - `g` : assistant guide 4 etapes
  - `space` : capturer la distance courante dans l'assistant
  - `x` : annuler l'assistant guide
  - `s` : sauvegarder dans `calibration.json`
  - `r` : recharger `calibration.json`
  - `k` : quitter le mode calibration
  - Parametres utiles pour stabilite aux bords : `launch hyst (m)` et `v2 trigger hyst (m)`

### Assistant guide (4 etapes)
1. Capturer la distance minimale de lancement video 1.
2. Capturer la distance maximale de lancement video 1.
3. Capturer la distance de bascule video 1 -> video 2.
4. Capturer la distance minimale d'acceptation du geste main droite pour video 2.

L'assistant applique ensuite la coherence des seuils automatiquement.

librairies principales : 
open cv
mediapipe
