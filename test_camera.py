import cv2

def test_cameras():
    print("--- DIAGNOSTIC CAMÉRAS ---")
    available_ids = []
    
    # Test des index de 0 à 10
    for i in range(11):
        # On teste avec et sans DirectShow
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"[OK] Caméra trouvée à l'ID : {i}")
                available_ids.append(i)
            cap.release()
        else:
            cap.release()
            
    if not available_ids:
        print("[ERREUR] Aucune caméra détectée. Vérifiez vos branchements.")
        print("CONSEIL : Vérifiez aussi 'Paramètres de confidentialité' sur Windows -> Caméra -> Autoriser les applications.")
    else:
        print(f"\nRésumé : Vous pouvez utiliser l'ID {available_ids[1] if len(available_ids) > 1 else available_ids[0]}")

if __name__ == "__main__":
    test_cameras()
