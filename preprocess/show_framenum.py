import cv2

def count_frames(video_path):
    # ビデオファイルの読み込み
    cap = cv2.VideoCapture(video_path)

    # 総フレーム数を取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ビデオファイルの解放
    cap.release()

    return total_frames

# テスト用のビデオファイルパスを指定して呼び出し
video_path = "../solos_data/25fps_video/Trombone/Rka-Gpys4bA.mp4"
total_frames = count_frames(video_path)
print(f"Total frames in {video_path}: {total_frames}")
