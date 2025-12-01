import cv2
import os

# ==========================================
# EXTRACT FRAMES FROM VIDEO
# ==========================================

########################
## 1. SETTINGS
########################
VIDEO_PATH = "video/4.mp4"     
OUTPUT_FOLDER = "frames_4"    


##########################
## 2. EXTRACTION CODE
##########################
def extract_frames():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Could not open video file")
        return

    frame_index = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # video finished
        # if frame_index % 2 != 0:
        #     frame_index += 1
            # continue
        filename = f"{frame_index}.jpeg"
        filepath = os.path.join(OUTPUT_FOLDER, filename)

        cv2.imwrite(filepath, frame)  # save frame
        print(f"[SAVED] {filepath}")

        frame_index += 1

    cap.release()
    print("\n✔ Extraction Complete!")



#########################
## 3. RUN
#########################
if __name__ == "__main__":
    extract_frames()
