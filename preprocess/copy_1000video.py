import os
import shutil

def copy_files_to_new_folder(folder_path, output_path, max_files=1000):
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            # 楽器フォルダのパス
            instrument_folder = os.path.join(root, dir_name)
            # 出力フォルダのパス
            output_instrument_folder = os.path.join(output_path, dir_name)
            # 出力フォルダが存在しない場合は作成する
            if not os.path.exists(output_instrument_folder):
                os.makedirs(output_instrument_folder)

            # 最大ファイル数までのファイルをコピー
            file_count = 0
            for file_name in os.listdir(instrument_folder):
                if file_count >= max_files:
                    break
                
                # ファイルの絶対パス
                file_path = os.path.join(instrument_folder, file_name)
                # 出力ファイル名
                output_file_path = os.path.join(output_instrument_folder, file_name)
                
                # ファイルをコピー
                shutil.copyfile(file_path, output_file_path)
                print(f"Copied {file_name} to {output_instrument_folder}.")
                
                file_count += 1

# テスト用のフォルダパスを指定して呼び出し
folder_path = "../solos_data/preprocessed_video/"
output_path = "../data/solos/videos/"
copy_files_to_new_folder(folder_path, output_path)