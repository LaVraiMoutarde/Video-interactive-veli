import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
import os
import urllib.request

class VideoPlayer:
    """Gestionnaire qui encadre le chargement des vidéos et le scrub fluide."""
    def __init__(self, video_paths):
        self.video_paths = video_paths
        self.current_video_idx = 0
        self.cap = cv2.VideoCapture(self.video_paths[self.current_video_idx])
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def switch_video(self):
        """Passe à la vidéo suivante dans la liste."""
        self.current_video_idx = (self.current_video_idx + 1) % len(self.video_paths)
        self.cap.release()
        self.cap = cv2.VideoCapture(self.video_paths[self.current_video_idx])
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Bascule sur la vidéo : {self.video_paths[self.current_video_idx]}")
        
    def get_frame_at_progress(self, progress):
        """Récupère une frame de la vidéo cible selon une progression entre 0.0 et 1.0"""
        if self.total_frames <= 0:
            return None
            
        target_frame = int(progress * (self.total_frames - 1))
        # deplace le curseur de lecture de l'objet videocaptur de opencv
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = self.cap.read()
        
        # En cas de perte de frames à la fin de la vidéo
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, self.total_frames - 2))
            ret, frame = self.cap.read()
            if not ret:
                return None
        return frame

def download_model_if_needed(script_dir):
    """Télécharge le modèle IA automatiquement pour un vrai 'Plug & Play'"""
    model_path = os.path.join(script_dir, "pose_landmarker_lite.task")
    if not os.path.exists(model_path):
        print("Téléchargement du modèle IA (MediaPipe Pose Lite) en cours, veuillez patienter...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, model_path)
        print("Téléchargement terminé !")
    return model_path

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_paths = [os.path.join(script_dir, "video1.mp4"), os.path.join(script_dir, "video2.mp4")]
    
    # Vérification de l'existence des vidéos
    for path in video_paths:
        if not os.path.exists(path):
            print(f"Erreur: La vidéo '{path}' est introuvable.")
            print("Lancez d'abord la création de vidéos de test : python generate_dummy_videos.py")
            return

    # Configuration de la nouvelle API MediaPipe "Tasks"
    model_path = download_model_if_needed(script_dir)
    with open(model_path, 'rb') as f:
        model_data = f.read()
    
    base_options = python.BaseOptions(model_asset_buffer=model_data)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    detector = vision.PoseLandmarker.create_from_options(options)

    # Initialisation de la webcam (ID=0)
    cap_webcam = cv2.VideoCapture(0)
    player = VideoPlayer(video_paths)
    
    # Paramétrage pour la distance : "largeur d'épaules" relative
    min_shoulder_dist = 0.15 
    max_shoulder_dist = 0.55
    
    smooth_dist = 0.0
    alpha = 0.1
    
    last_switch_time = 0
    cooldown_seconds = 1.0

    print("=== Démarrage ===")
    print("1) Avancez ou reculez le visage pour 'scrubber' la vidéo.")
    print("2) Levez une main au dessus de l'épaule pour changer de vidéo.")
    print("3) Appuyez sur la touche 'q' de la fenêtre OpenCV pour quitter.")

    while True:
        ret, web_frame = cap_webcam.read()
        if not ret:
            print("Incapable de lire le flux de la webcam.")
            break
            
        web_frame = cv2.flip(web_frame, 1)
        rgb_frame = cv2.cvtColor(web_frame, cv2.COLOR_BGR2RGB)
        
        # Analyse du spectateur via l'API Tasks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        
        progress = 0.0
        
        # si humain détecté
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0] # 1ère personne
            
            # Les IDs des points clefs: 11(épaule G), 12(épaule D), 15(poignet G), 16(poignet D)
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            
            # Draw call des articulations sur l'image webcam
            h, w, _ = web_frame.shape
            for point in [left_shoulder, right_shoulder, left_wrist, right_wrist]:
                cv2.circle(web_frame, (int(point.x * w), int(point.y * h)), 8, (0, 255, 255), -1)
            
            # 1)gesion de la distance (Avancer/Reculer la vidéo)
            dist = np.sqrt((left_shoulder.x - right_shoulder.x)**2 + (left_shoulder.y - right_shoulder.y)**2)
            
            if smooth_dist == 0.0:
                 smooth_dist = dist
            else:
                 smooth_dist = alpha * dist + (1 - alpha) * smooth_dist
                 
            progress = (smooth_dist - min_shoulder_dist) / (max_shoulder_dist - min_shoulder_dist)
            progress = np.clip(progress, 0.0, 1.0)
            
            # 2) gestion du geste lever la main(Changer la vidéo)
            hand_raised = False
            if (left_wrist.y < left_shoulder.y - 0.1) or (right_wrist.y < right_shoulder.y - 0.1):
                hand_raised = True
                
            current_time = time.time()
            if hand_raised and (current_time - last_switch_time > cooldown_seconds):
                player.switch_video()
                last_switch_time = current_time

        video_frame = player.get_frame_at_progress(progress)
        
        # Injection de la vidéo 
        if video_frame is not None:
            # Fixation de la taille :
            vw, vh = 800, 600
            video_frame = cv2.resize(video_frame, (vw, vh))
            
            # Webcam miniature en bas à droite
            ww, wh = int(vw / 3), int(vh / 3)
            web_frame_small = cv2.resize(web_frame, (ww, wh))
            video_frame[vh - wh - 10 : vh - 10, vw - ww - 10 : vw - 10] = web_frame_small
            
            # Overlay par dessus
            percent = int(progress * 100)
            cv2.putText(video_frame, f"Timeline : {percent}%", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(video_frame, "- Levez la main pour Switch", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
                        
            # progress bar
            padding_bar = 20
            cv2.rectangle(video_frame, (padding_bar, vh - 40), (vw - ww - 30, vh - 15), (50, 50, 50), -1)
            progress_px = int(progress * ((vw - ww - 30) - padding_bar))
            cv2.rectangle(video_frame, (padding_bar, vh - 40), (padding_bar + progress_px, vh - 15), (0, 255, 0), -1)

            cv2.imshow("Projet Interaction Video", video_frame)
        else:
            # Cas de fallback pour ne pas crasher
            cv2.imshow("Projet Interaction Video", web_frame)

        # Rafraichissement et détection de la sortie
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup strict des ressources
    cap_webcam.release()
    player.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
