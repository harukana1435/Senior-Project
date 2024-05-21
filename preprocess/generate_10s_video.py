import os
import cv2
import json

def generate_10s_video(folder_path, output_path, frames_per_segment):
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            # 楽器フォルダのパス
            instrument_folder = os.path.join(root, dir_name)
            # 出力フォルダのパス
            output_folder = os.path.join(output_path, dir_name)
            # 出力フォルダが存在しない場合は作成する
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            video_number = 0
            video_info = {}

            # 楽器フォルダ内のファイルに対して処理を行う
            for file_name in os.listdir(instrument_folder):
                # 拡張子が ".mp4" の動画ファイルに対してのみ処理を行う
                if file_name.endswith(".mp4"):
                    # ファイルの絶対パス
                    file_path = os.path.join(instrument_folder, file_name)
                    cap = cv2.VideoCapture(file_path)

                    # ビデオのプロパティを取得
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    loop_per_video = frame_count // frames_per_segment
                    frame_index = 0

                    for i in range(loop_per_video):
                        # 出力ビデオファイル名
                        output_video_path = os.path.join(output_folder, f"video_{video_number:05d}.mp4")              
                        # 出力ビデオを作成
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(output_video_path, fourcc, frames_per_segment/10.0, (width, height))
                        # ビデオ情報を保存
                        video_info[f"video_{video_number:05d}"] = {
                            "video_id": file_name[:-4],
                            "start_frame": frame_index
                        }
                        print(f"video_{video_number:05d} {file_name[:-4]} {frame_index}")
                        # 指定されたフレーム数ごとにセグメントを保存する
                        for _ in range(frames_per_segment):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            out.write(frame)
                            frame_index += 1
                        out.release()
                        video_number += 1
                    
                    cap.release()
            # JSONファイルにビデオ情報を保存
            with open(os.path.join("../solos_data", f"{dir_name}_info.json"), "w") as json_file:
                json.dump(video_info, json_file, indent=4)           

# テスト用のフォルダパスと出力先フォルダパスを指定して呼び出し
folder_path = "../solos_data/25fps_video/"
output_path = "../solos_data/10s_video/"
generate_10s_video(folder_path, output_path, 215)
