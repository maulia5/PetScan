import cv2
import numpy as np
import pickle
import os
import time
import math
import shutil
import mediapipe as mp
import traceback # Tambahan untuk melihat detail error
from pathlib import Path
from insightface.app import FaceAnalysis

# ================= KONFIGURASI =================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path(".").resolve()

# 1. Folder Dataset BARU (Terpisah dari yang lama)
DATASET_DIR = BASE_DIR / "Dataset_Wajah_Baru"
TRAIN_DIR = DATASET_DIR / "Data_Foto"
DB_PATH = BASE_DIR / "new_embeddings.pkl"

# 2. Model AI
MODEL_NAME = "buffalo_s" # MobileFaceNet

# 3. Threshold (Sesuai Permintaan: 70%)
SIMILARITY_THRESHOLD = 0.70

# 4. Konfigurasi KEDIP (Liveness)
EAR_THRESHOLD = 0.16        # Batas mata tertutup
BLINK_CONSEC_FRAMES = 2     # Harus merem 2 frame berturut-turut
BLINK_COOLDOWN = 4.0        # Jeda antar kedipan

# Nama Jendela (Penting untuk Fokus)
WINDOW_NAME = "Sistem Presensi AI"

class FaceSystem:
    def __init__(self):
        print("\n[INIT] Menyiapkan Sistem AI & Detektor Kedip...")
        
        TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        
        self.app = FaceAnalysis(name=MODEL_NAME, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        self.db_embs = None
        self.db_names = None
        self.load_database()

    def load_database(self):
        if DB_PATH.exists():
            with open(DB_PATH, "rb") as f:
                data = pickle.load(f)
                self.db_embs = data["embs"]
                self.db_names = data["names"]
            print(f"[DATA] Database dimuat: {len(self.db_names)} user.")
            return True
        print("[DATA] Database baru/kosong.")
        return False

    def euclidean_dist(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def calculate_ear(self, landmarks, indices, w, h):
        coords = []
        for idx in indices:
            lm = landmarks[idx]
            coords.append((int(lm.x * w), int(lm.y * h)))

        v1 = self.euclidean_dist(coords[1], coords[5])
        v2 = self.euclidean_dist(coords[2], coords[4])
        h_dist = self.euclidean_dist(coords[0], coords[3])

        if h_dist == 0: return 0.0
        return (v1 + v2) / (2.0 * h_dist)

    def detect_blink(self, frame, blink_counter):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        ear_avg = 0.0
        triggered = False
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lm = face_landmarks.landmark
                left_ear = self.calculate_ear(lm, self.LEFT_EYE, w, h)
                right_ear = self.calculate_ear(lm, self.RIGHT_EYE, w, h)
                ear_avg = (left_ear + right_ear) / 2.0
                
                if ear_avg < EAR_THRESHOLD:
                    blink_counter += 1
                else:
                    if blink_counter >= BLINK_CONSEC_FRAMES:
                        triggered = True
                    blink_counter = 0
                    
        return triggered, blink_counter, ear_avg

    def perform_countdown(self, cap, seconds=3, message=""):
        """Hitung mundur dengan check tombol Q agar bisa keluar kapan saja"""
        start_time = time.time()
        
        # Setup Window sekali saja di awal timer agar aman
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

        while int(time.time() - start_time) < seconds:
            ret, frame = cap.read()
            if not ret: return None
            frame = cv2.flip(frame, 1)
            
            elapsed = int(time.time() - start_time)
            remaining = seconds - elapsed
            
            h, w, _ = frame.shape
            center_x, center_y = w // 2, h // 2
            
            # Efek Gelap
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # Lingkaran
            cv2.circle(frame, (center_x, center_y), 80, (0, 255, 255), 4)
            cv2.circle(frame, (center_x, center_y), 70, (0, 0, 0), -1)
            
            # Angka Timer
            text_size = cv2.getTextSize(str(remaining), cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
            text_x = int(center_x - text_size[0] / 2)
            text_y = int(center_y + text_size[1] / 2)
            cv2.putText(frame, str(remaining), (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
            
            if message:
                cv2.putText(frame, message, (center_x - 150, center_y + 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow(WINDOW_NAME, frame)
            
            # FORCE FOCUS: Terus menerus set TopMost saat timer
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

            # Cek tombol Q saat timer berjalan
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return "EXIT" # Kode khusus untuk keluar
        
        ret, frame = cap.read()
        if ret:
            return cv2.flip(frame, 1)
        return None

    def capture_phase(self, user_name):
        print(f"\n[PHASE 1] Daftar User: {user_name}")
        cap = cv2.VideoCapture(0)
        
        # Setup Window agar selalu di atas dan posisi fixed
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 640, 480)
        cv2.moveWindow(WINDOW_NAME, 100, 100)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        
        captured_frame = None
        mode = "preview"
        blink_cnt = 0
        
        print("Silakan BERKEDIP untuk memulai timer foto...")

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            if mode == "preview":
                is_blink, blink_cnt, ear_val = self.detect_blink(frame, blink_cnt)
                status_text = "Silakan KEDIP untuk Foto"
                color = (0, 255, 255)
                
                if is_blink:
                    print("Kedipan Terdeteksi! Memulai Timer...")
                    captured_frame = self.perform_countdown(cap, 3, "Bersiap...")
                    
                    # Fix Crash: Cek dulu apakah captured_frame itu string "EXIT"
                    if isinstance(captured_frame, str) and captured_frame == "EXIT": 
                        cap.release()
                        cv2.destroyAllWindows()
                        return None
                    elif captured_frame is not None:
                        mode = "review"
                
                cv2.putText(frame, f"DAFTAR: {user_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"EAR: {ear_val:.2f}", (500, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow(WINDOW_NAME, frame)
                # FORCE FOCUS HACK: Set properti di setiap frame
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    cap.release()
                    cv2.destroyAllWindows()
                    return None

            elif mode == "review":
                display = captured_frame.copy()
                cv2.rectangle(display, (0, 0), (display.shape[1], 100), (0,0,0), -1)
                cv2.putText(display, "Apakah foto ini bagus?", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(display, "[ENTER] Simpan  |  [R] Ulangi", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow(WINDOW_NAME, display)
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

                key = cv2.waitKey(0)
                if key == 13: # ENTER
                    cap.release()
                    cv2.destroyAllWindows()
                    return captured_frame
                elif key == ord('r') or key == ord('R'):
                    mode = "preview"
                    blink_cnt = 0
                elif key == ord('q') or key == ord('Q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return None

    def save_and_train(self, frame, user_name):
        print(f"\n[PHASE 2] Menyimpan data {user_name}...")
        user_folder = TRAIN_DIR / user_name
        user_folder.mkdir(parents=True, exist_ok=True)
        
        filename = f"{user_name}_blink_01.jpg"
        save_path = user_folder / filename
        cv2.imwrite(str(save_path), frame)
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.app.get(img_rgb)
        
        if len(faces) == 0:
            print("[ERROR] Wajah tidak terdeteksi! Coba foto ulang.")
            return False
            
        face = max(faces, key=lambda a: (a.bbox[2]-a.bbox[0]) * (a.bbox[3]-a.bbox[1]))
        new_emb = face.embedding / np.linalg.norm(face.embedding)
        
        if self.db_embs is None:
            self.db_embs = np.array([new_emb], dtype=np.float32)
            self.db_names = np.array([user_name])
        else:
            self.db_embs = np.vstack([self.db_embs, new_emb])
            self.db_names = np.append(self.db_names, user_name)
            
        data = {"embs": self.db_embs, "names": self.db_names}
        with open(DB_PATH, "wb") as f:
            pickle.dump(data, f)
            
        print(f"[TRAIN] Berhasil! Total User Database: {len(set(self.db_names))}")
        return True

    def live_test_phase(self):
        print("\n[PHASE 3] Live Test Mode (TRIGGER KEDIP)")
        
        cap = cv2.VideoCapture(0)
        
        # Setup Window agar selalu di atas
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 640, 480)
        cv2.moveWindow(WINDOW_NAME, 100, 100)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

        blink_cnt = 0
        last_check_time = 0
        
        display_name = "Menunggu Kedip..."
        display_color = (200, 200, 200)
        display_score = 0.0
        display_result_text = "..."
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            curr_time = time.time()
            
            is_blink, blink_cnt, ear_val = self.detect_blink(frame, blink_cnt)
            
            if is_blink and (curr_time - last_check_time > BLINK_COOLDOWN):
                print("Kedipan Test Terdeteksi! Memulai Timer...")
                
                # --- TIMER 3 DETIK (Bisa dicancel tombol Q) ---
                frame_clean = self.perform_countdown(cap, 3, "Validasi Wajah...")
                
                # Fix Crash: Cek dulu apakah frame_clean itu string "EXIT"
                if isinstance(frame_clean, str) and frame_clean == "EXIT":
                    break # Keluar loop utama
                elif frame_clean is not None:
                    # --- PROSES RECOGNITION ---
                    img_rgb = cv2.cvtColor(frame_clean, cv2.COLOR_BGR2RGB)
                    faces = self.app.get(img_rgb)
                    
                    if len(faces) > 0:
                        face = max(faces, key=lambda a: (a.bbox[2]-a.bbox[0]) * (a.bbox[3]-a.bbox[1]))
                        emb = face.embedding / np.linalg.norm(face.embedding)
                        
                        if self.db_embs is not None:
                            scores = np.dot(self.db_embs, emb)
                            best_idx = np.argmax(scores)
                            max_score = scores[best_idx]
                            
                            # Logika Threshold 70% (0.70)
                            if max_score > SIMILARITY_THRESHOLD:
                                display_name = self.db_names[best_idx]
                                display_result_text = "ABSENSI BERHASIL"
                                display_color = (0, 255, 0)
                            else:
                                display_name = "User"
                                display_result_text = "TIDAK DIKENALI"
                                display_color = (0, 0, 255)
                            display_score = max_score
                        else:
                            display_name = "DB Kosong"
                    else:
                        display_name = "Wajah Tidak Jelas"
                        display_result_text = "Error"
                        display_color = (0, 255, 255)
                        
                    last_check_time = time.time()

            # UI
            cv2.rectangle(frame, (0,0), (frame.shape[1], 100), (0,0,0), -1)
            cv2.putText(frame, f"MODE TEST: Kedip untuk Cek (Threshold {SIMILARITY_THRESHOLD})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"EAR: {ear_val:.2f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)

            cv2.putText(frame, f"{display_result_text}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, display_color, 3)
            if display_score > 0:
                cv2.putText(frame, f"{display_name} ({display_score:.1%})", (300, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            cv2.imshow(WINDOW_NAME, frame)
            
            # FORCE ON TOP TERUS MENERUS
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def cleanup_dataset():
    print("\n[CLEANUP] Membersihkan dataset dan cache...")
    if DATASET_DIR.exists():
        try:
            shutil.rmtree(DATASET_DIR)
            print(f"[DELETE] Folder {DATASET_DIR} berhasil dihapus.")
        except Exception as e:
            print(f"[ERROR] Gagal menghapus folder dataset: {e}")
    if DB_PATH.exists():
        try:
            os.remove(DB_PATH)
            print(f"[DELETE] File {DB_PATH} berhasil dihapus.")
        except Exception as e:
            print(f"[ERROR] Gagal menghapus file database: {e}")

def main():
    system = FaceSystem()
    try:
        while True:
            print("\n=== MENU UTAMA (DATASET BARU) ===")
            name = input("Masukkan Nama User Baru (atau 'skip' untuk tes): ").strip()
            
            if name.lower() != 'skip' and name != "":
                photo = system.capture_phase(name)
                if photo is not None:
                    success = system.save_and_train(photo, name)
                    if success:
                        print("Lanjut ke Live Test...")
                        time.sleep(1)
                else:
                    print("Dibatalkan.")
            
            system.live_test_phase()
            
            if input("\nReset/Daftar lagi? (y/n): ").lower() != 'y':
                break
    except KeyboardInterrupt:
        print("\n[INFO] Program dihentikan paksa.")
    except Exception as e:
        # CATCH ERROR AGAR KITA TAHU PENYEBAB CRASH
        print("\n[CRASH] Terjadi Error Serius:")
        print(traceback.format_exc())
    finally:
        cleanup_dataset()
        print("[INFO] Program Selesai.")

if __name__ == "__main__":
    main()