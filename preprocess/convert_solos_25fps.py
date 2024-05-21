import os

def convert_to_25fps(folder_path, output_path):
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            # 楽器フォルダのパス
            instrument_folder = os.path.join(root, dir_name)
            if not os.path.exists(os.path.join(output_path, dir_name)):
                        os.makedirs(os.path.join(output_path, dir_name))
                    
            # 楽器フォルダ内のファイルに対して処理を行う
            for file_name in os.listdir(instrument_folder):
                # ファイルの絶対パス
                file_path = os.path.join(instrument_folder, file_name)
                # 拡張子が ".mp4" の動画ファイルに対してのみ処理を行う
                if file_name.endswith(".mp4"):
                    # 出力ファイル名
                    output_file_path = os.path.join(output_path, dir_name, f"{file_name}")
                    # FFmpeg コマンドの実行
                    os.system(f"ffmpeg -i {file_path} -r 25 -c:v libx264 -crf 18 -preset medium -c:a copy {output_file_path}")
                    print(f"Converted {file_name} to 25fps in {dir_name} folder.")

# テスト用のフォルダパスを指定して呼び出し
folder_path = "../solos_data/original_video/"
output_path = "../solos_data/25fps_video/"
convert_to_25fps(folder_path, output_path)