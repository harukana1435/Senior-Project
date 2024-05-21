import os

# ファイルパス
data_dir = "../data"
instruments = ["Violin", "Cello", "Clarinet", "Flute", "Trombone", "Horn"]
valid_file = os.path.join(data_dir, "solos_valid.txt")
train_combined_file = os.path.join(data_dir, "solos_train.txt")

# 楽器ごとにファイルを読み込んで書き込む
with open(valid_file, 'w', encoding='utf-8') as valid_output:
    # video_00800からvideo_00999までを楽器名をつけて書き込む
    for instrument in instruments:
        for idx in range(800, 1000):
            valid_output.write(f"{instrument}/video_{idx:05d}\n")

with open(train_combined_file, 'w', encoding='utf-8') as train_output:
    # video_00000からvideo_00799までを書き込む
    for instrument in instruments:
        for idx in range(800):
            train_output.write(f"{instrument}/video_{idx:05d}\n")