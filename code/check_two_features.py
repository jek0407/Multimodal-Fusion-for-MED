import os
import pandas as pd

cnn3d_path = './data/cnn3d'
snf_path = './data/snf_conv5'

cnn3d_files = {os.path.splitext(filename)[0] for filename in os.listdir(cnn3d_path) if filename.endswith('.pkl')}
snf_files = {os.path.splitext(filename)[0] for filename in os.listdir(snf_path) if filename.endswith('.csv')}

different_files = cnn3d_files.symmetric_difference(snf_files)

for file in different_files:
    file_path = os.path.join(cnn3d_path, file + '.pkl')
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted '{file_path}'")
    else:
        print(f"File '{file_path}' not found, cannot delete.")

train_val_label_file = './data/labels/train_val.csv'
df = pd.read_csv(train_val_label_file)
# different_files에 있는 이름들을 제외하고 나머지만 필터링
filtered_df = df[~df['Id'].str.replace('.csv', '').isin(different_files)]

# 필터링된 데이터프레임을 다시 CSV로 저장
filtered_df.to_csv(train_val_label_file, index=False)
print(f"Updated '{train_val_label_file}' with filtered data.")

