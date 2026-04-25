#  Installation Vidéo Interactive

Bienvenue dans votre projet d'installation interactive "Paradoxe de l'echelle"! Ce guide va vous expliquer simplement comment tout mettre en place le projet.

## C'est quoi ce projet ?
C'est une expérience où **votre corps devient la télécommande**. En vous déplaçant devant la caméra, vous faites avancer ou reculer le temps dans la vidéo. Un simple geste de la main permet de changer de scène !

---

##  Matériel Nécessaire

Pour que l'installation fonctionne parfaitement, vous avez besoin de :

1.  **Un Ordinateur** : Sous Windows ou Mac, avec une puissance correcte.
2.  **Une Webcam** : Une webcam USB externe standard (HD de préférence) ou une caméra du pc portable.
3.  **Un Vidéoprojecteur ou un Grand Écran** : Pour diffuser l'expérience.

## ⚡ Guide de Démarrage Rapide

1.  **Vérifiez vos vidéos** : Assurez-vous d'avoir deux fichiers nommés `video 1.mp4` et `video 2.mp4` dans ce dossier.
2.  **Lancez l'application** : Double-cliquez sur le fichier `run.bat`.
3.  **Jouez** : Cliquez sur le bouton "demarrer l'experience" pour lancer l'expérience.
4.  **Relancez l'experience** : Cliquez sur la touche "B" ou quittez le champ de vision de la camera.

---

## 🏛️ Installation Physique (Conseils)

Pour garantir une interaction fluide, voici comment organiser votre espace :

### 1. Position de la caméra
- Placez la webcam en dessous de la zone de projoection, face à l'utilisateur à une hauteur sufisante pour que la camera puisse voir un utilisateur de la tete au bassin.

### 2. Espace de jeu
- Prévoyez un recul de **3 à 4 mètres** entre l'utilisateur et la caméra.
- Gardez la zone de passage dégagée : l'ordinateur ne doit suivre qu'une seule personne à la fois.

### 3. Éclairage
- Privilégiez un éclairage **homogène** si possible. Évitez d'avoir une fenêtre lumineuse ou un projecteur qui éblouit directement la caméra.
- L'utilisateur doit être bien visible.

---

##  Installation Technique (Première fois)

Si vous installez le projet sur un nouvel ordinateur :

### 1. Installer Python
- [Téléchargez Python ici](https://www.python.org/downloads/windows/) (Cochez bien la case **"Add Python to PATH"** pendant l'installation).

### 2. Installer les outils
Ouvrez un terminal (tapez "cmd" dans Windows) et tapez :
```bash
pip install -r requirements.txt
```
ou ouvrerz le terminal et tapez :
pip install opencv-python mediapipe numpy
---

##  Raccourcis Utiles

-   **F** : Passer en Plein Écran.
-   **Q** : Quitter l'application.
-   **B** : Réinitialise l'expérience.

