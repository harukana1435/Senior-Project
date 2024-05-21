import os

def extract_audio(folder_path, output_path):
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            # 楽器フォルダのパス
            instrument_folder = os.path.join(root, dir_name)
            # 出力フォルダのパス
            output_folder = os.path.join(output_path, dir_name)
            # 出力フォルダが存在しない場合は作成する
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # 楽器フォルダ内のファイルに対して処理を行う
            for file_name in os.listdir(instrument_folder):
                # ファイルの絶対パス
                file_path = os.path.join(instrument_folder, file_name)
                # 拡張子が ".mp4" の動画ファイルに対してのみ処理を行う
                if file_name.endswith(".wav"):
                    # 出力ファイル名（拡張子を ".wav" に変更）
                    output_file_path = os.path.join(output_folder, f"{file_name[:-4]}.wav")
                    # FFmpeg コマンドの実行（テンポを 0.86 倍に変更して保存）
                    os.system(f"ffmpeg -y -i {file_path} -af atempo=0.86 {output_file_path}")
                    print(f"Adjusted tempo of {file_name} in {dir_name} folder.")

# テスト用のフォルダパスと出力先フォルダパスを指定して呼び出し
folder_path = "../solos_data/25fps_audio/"
output_path = "../solos_data/0.86x_audio/"
extract_audio(folder_path, output_path)
