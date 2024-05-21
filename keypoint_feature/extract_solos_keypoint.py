import os
import json
import numpy as np

def extract_solos_keypoint(video_info_folder, keypoints, output_folder, skeleton_dict):
    # 25fps_audioフォルダ内の各楽器フォルダについて処理を行う
    for instrument_folder_name in ["Cello", "Violin", "Clarinet", "Flute", "Trombone", "Horn"]:
        # 各楽器のJSONファイルのパスを取得
        instrument_json_path = os.path.join(video_info_folder, f"{instrument_folder_name}_info.json")
        print(instrument_json_path)
        # 各楽器のJSONファイルからビデオ情報を読み取る
        with open(instrument_json_path, 'r') as f:
            video_info = json.load(f)

        # 出力フォルダが存在しない場合は作成する
        instrument_output_folder = os.path.join(output_folder, instrument_folder_name)
        if not os.path.exists(instrument_output_folder):
            os.makedirs(instrument_output_folder)

        print(instrument_output_folder)
 
        count = 0
        # 各ビデオについて処理を行う
        for video_name, info in video_info.items():
            if(count == 1000):break

            # ビデオのIDと開始フレームを取得
            video_id = info['video_id']
            start_frame = skeleton_dict[video_id][0]+info['start_frame']
            end_frame = start_frame+215

            video_keypoints = keypoints[start_frame:end_frame]
            temp_keypoints =[]
            for i in range(215):
                temp = [video_keypoints[i]]
                for j in range(7):
                    if(i-j<0):
                        temp.insert(0, video_keypoints[i])
                    else:
                        temp.insert(0, video_keypoints[i-j])
                for j in range(7):
                    if(i+j>214):
                        temp.append(video_keypoints[i])
                    else:
                        temp.append(video_keypoints[i+j])
                temp_keypoints.append(temp)
            
            npy_data = np.array(temp_keypoints)
            print(npy_data.shape)
            npy_data = npy_data.transpose(0, 2, 1, 3)
            print(npy_data.shape)

            # npyファイルの生成
            npy_output_path = os.path.join(instrument_output_folder, f"{video_name}.npy")
            np.save(npy_output_path, npy_data)
            print(npy_output_path)
            count += 1

# テスト用のファイルパスを指定して呼び出し
keypoints = np.memmap('skeleton_npy_padded.npy', dtype=np.float32, mode='r', shape=(5976615, 3, 47))
video_info_folder = "../solos_data"
output_folder = "../solos_data/keypoints_data"
# JSONファイルからビデオ情報を読み取る
with open("skeleton_dict.json", 'r') as f:
    skeleton_dict = json.load(f)
extract_solos_keypoint(video_info_folder, keypoints, output_folder, skeleton_dict)
