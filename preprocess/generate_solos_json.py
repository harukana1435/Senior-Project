import os
import json

def generate_json(folder_path, output_path):
    result = {}

    # 指定されたフォルダ内の各サブフォルダに対して処理を行う
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            # サブフォルダのパス
            subfolder_path = os.path.join(root, dir_name)
            # サブフォルダ内の".mp4"ファイルを取得
            mp4_files = [file_name[:-4] for file_name in os.listdir(subfolder_path) if file_name.endswith(".mp4")]

            # ".mp4"ファイル名から拡張子を除いた部分をJSONに追加
            result[dir_name] = mp4_files

    # 結果を JSON ファイルとして保存
    with open(output_path, 'w') as json_file:
        json.dump(result, json_file, indent=4)

# テスト用のフォルダパスと出力先のJSONファイルパスを指定して呼び出し
folder_path = "../solos_data/original_video"
output_path = "../solos_data/video_id.json"
generate_json(folder_path, output_path)
