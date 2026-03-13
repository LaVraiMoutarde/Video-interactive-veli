# Prototype interactif Webcam -> Vidéo (OpenCV & MediaPipe)

Ce prototype permet de contrôler la timeline d'une vidéo avec son placement physique par rapport à la caméra, et de changer de vidéo grâce aux gestes de la main.

## Interactions
- **Scrubbing spatial** : Plus vous vous approchez de la webcam, plus la vidéo avance. Plus vous vous reculez, plus elle retourne vers le début. 
  *L'interaction est gérée par la distance entre vos deux épaules, lissée via un filtre (EMA) pour un effet fluide et une vraie stabilité.*
- **Changement de canal** : Levez une main au-dessus de la hauteur de votre épaule, et la seconde vidéo prendra le relai après un cooldown de 1 seconde.

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

**1. Avoir des vidéos dans le dossier :**
Le script s'attend à lire `video1.mp4` et `video2.mp4`. Si vous n'en avez pas, vous pouvez générer rapidement deux vidéos de démonstration avec :
```bash
python generate_dummy_videos.py
```

**2. Lancer l'application :**
```bash
python main.py
```

Raccourci clavier de l'app :
- **q** : Quitter le programme.

librairies principales : 
onpen cv
mediapipe
