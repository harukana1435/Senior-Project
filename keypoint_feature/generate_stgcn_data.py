import numpy as np
import os

def load_and_label_files(instrument_dirs, instrument_labels):
    train_data_list = []
    test_data_list = []
    train_labels_list = []
    test_labels_list = []

    for instrument, label in zip(instrument_dirs, instrument_labels):
        base_dir = instrument
        file_pattern = 'video_{:05d}.npy'

        for i in range(1000):
            file_name = file_pattern.format(i)
            file_path = os.path.join(base_dir, file_name)
            if os.path.exists(file_path):
                if i <= 799:
                    if i % 40 == 1:
                        data = np.load(file_path)
                        train_data_list.append(data)
                        train_labels_list.append(np.full(215, label))
                else:
                    if i % 40 == 1:
                        data = np.load(file_path)
                        test_data_list.append(data)
                        test_labels_list.append(np.full(215, label))
    
    train_data = np.concatenate(train_data_list, axis=0)
    test_data = np.concatenate(test_data_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    test_labels = np.concatenate(test_labels_list, axis=0)

    return train_data, train_labels, test_data, test_labels

def save_data(train_data, train_labels, test_data, test_labels, output_dir, temp_labels):
    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)
    print(temp_labels.shape)
    np.save(os.path.join(output_dir, 'train_data.npy'), train_data)
    np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(output_dir, 'test_data.npy'), test_data)
    np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels)
    np.save(os.path.join(output_dir, 'temp_labels.npy'), temp_labels)
    print(f"Data saved to {output_dir}")

def main():
    instrument_dirs = [
        '../solos_data/keypoints_data/Cello',
        '../solos_data/keypoints_data/Violin',
        '../solos_data/keypoints_data/Clarinet',
        '../solos_data/keypoints_data/Flute',
        '../solos_data/keypoints_data/Horn',
        '../solos_data/keypoints_data/Trombone'
    ]
    instrument_labels = [0, 1, 2, 3, 4, 5]  # 各楽器に対するラベル

    train_data, train_labels, test_data, test_labels = load_and_label_files(instrument_dirs, instrument_labels)

    temp_labels = np.full(215, 0)

    output_dir = './'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_data(train_data, train_labels, test_data, test_labels, output_dir, temp_labels)

if __name__ == "__main__":
    main()
