import os
import json
import shutil

def concatenate_audio_with_video_for_instruments(video_info_folder, audio_folder, video_folder, output_folder):
    # 25fps_audioフォルダ内の各楽器フォルダについて処理を行う
    for instrument_folder_name in os.listdir(audio_folder):
        # 各楽器のJSONファイルのパスを取得
        instrument_json_path = os.path.join(video_info_folder, f"{instrument_folder_name}_info.json")
        
        # 各楽器のJSONファイルからビデオ情報を読み取る
        with open(instrument_json_path, 'r') as f:
            video_info = json.load(f)

        # 出力フォルダが存在しない場合は作成する
            if not os.path.exists(os.path.join(output_folder, instrument_folder_name)):
                os.makedirs(os.path.join(output_folder, instrument_folder_name))

        # 各ビデオについて処理を行う
        for video_name, info in video_info.items():
            # ビデオのIDと開始フレームを取得
            video_id = info['video_id']
            start_frame = info['start_frame']
            if start_frame == 0:
                start_pos = 0
            else:
                start_pos = start_frame/21.5
            # 終了フレームを計算（次のビデオの開始フレームまで）
            end_pos = start_pos+10.0

            # 音声ファイルのパスを構築
            audio_file_path = os.path.join(audio_folder, instrument_folder_name, f"{video_id}.wav")

            # 出力ビデオファイルのパスを構築
            video_file_path = os.path.join(video_folder, instrument_folder_name, f"{video_name}.mp4")

            output_file_path = os.path.join(output_folder, instrument_folder_name, f"{video_name}.mp4")

            # 開始フレームから終了フレームまでの音声を切り出して結合
            extract_audio_command = f"ffmpeg -y -i {audio_file_path} -ss {start_pos} -to {end_pos} -c copy temp_audio.wav"
            os.system(extract_audio_command)

            # 映像に音声を結合
            combine_audio_command = f"ffmpeg -y -i {video_file_path} -i temp_audio.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 {output_file_path}"
            os.system(combine_audio_command)


        # 一時音声ファイルを削除
        os.remove("temp_audio.wav")

# テスト用のファイルパスを指定して呼び出し
video_info_folder = "../solos_data"
audio_folder = "../solos_data/0.86x_audio"
video_folder = "../solos_data/10s_video"
output_folder = "../solos_data/preprocessed_video"
concatenate_audio_with_video_for_instruments(video_info_folder, audio_folder, video_folder, output_folder)
