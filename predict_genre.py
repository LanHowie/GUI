# 本腳本使用訓練好的 PyTorch CNN 模型預測音頻檔案的音樂類型。
# 可用在GUI幫助使用者預測音頻檔案的音樂類型。
# 請確保已運行 'my_script.py' 以生成以下檔案：
# - 'music_genre_cnn.pth'：訓練好的模型參數
# - 'scaler.pkl'：特徵標準化器
# - 'label_encoder.pkl'：類型標籤編碼器
#
# 使用說明：
# 1. 安裝所需庫：
#    pip install torch numpy librosa scikit-learn
# 2. 在 main() 函數中更新 'audio_file' 為您想要預測的音頻檔案路徑。
# 3. 在 Jupyter Notebook 單元格中運行本腳本。
#
# 注意事項：
# - 音頻檔案支援 .au、.wav、.mp3 等格式，建議與 GTZAN 數據集的 .au 格式一致。
# - 音頻長度需至少 3 秒，多個片段將預測並返回最常見的類型。
# - 確保模型和預處理檔案位於工作目錄。

import os
import numpy as np
import librosa
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# 定義音樂類型（必須與訓練腳本一致）
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# 定義 PyTorch CNN 模型（必須與訓練腳本一致）
class MusicGenreCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MusicGenreCNN, self).__init__()
        # 第一層：線性層、ReLU 激活、20% 丟棄率
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # 第二層：線性層、ReLU 激活、20% 丟棄率
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # 第三層：線性層、ReLU 激活、20% 丟棄率
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # 第四層：線性層、ReLU 激活、20% 丟棄率
        self.layer4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # 輸出層：映射到類型數量
        self.output = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # 前向傳播：依次通過各層
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output(x)
        return x

# 提取音頻特徵的函數（與訓練腳本一致）
def extract_features(file_path, segment_duration=3):
    try:
        # 加載音頻檔案，採樣率設為 22050 Hz
        audio, sr = librosa.load(file_path, sr=22050)
        # 計算每個片段的樣本數
        samples_per_segment = int(segment_duration * sr)
        # 計算片段數量
        num_segments = int(len(audio) / samples_per_segment)
        
        features = []
        for seg in range(num_segments):
            # 提取當前片段的音頻數據
            start_sample = samples_per_segment * seg
            end_sample = start_sample + samples_per_segment
            segment = audio[start_sample:end_sample]
            
            # 提取 20 個 MFCC 係數
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20).mean(axis=1)
            # 提取 12 個色度特徵
            chroma = librosa.feature.chroma_stft(y=segment, sr=sr).mean(axis=1)
            # 提取 7 個頻譜對比特徵
            contrast = librosa.feature.spectral_contrast(y=segment, sr=sr).mean(axis=1)
            # 提取頻譜質心
            centroid = librosa.feature.spectral_centroid(y=segment, sr=sr).mean()
            # 提取頻譜滾降
            rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr).mean()
            # 提取零交叉率
            zcr = librosa.feature.zero_crossing_rate(y=segment).mean()
            
            # 合併特徵為 42 維向量
            feature_vector = np.concatenate([mfcc, chroma, contrast, [centroid, rolloff, zcr]])
            features.append(feature_vector)
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

# 預測單個音頻檔案類型的函數
def predict_genre(file_path, model, scaler, label_encoder, device):
    # 提取特徵
    features = extract_features(file_path)
    if not features:
        return None
    
    # 轉為 NumPy 陣列
    features = np.array(features)
    
    # 標準化特徵
    features_scaled = scaler.transform(features)
    
    # 轉為 PyTorch 張量並移到指定設備
    features_tensor = torch.FloatTensor(features_scaled).to(device)
    
    # 設置模型為評估模式
    model.eval()
    predictions = []
    with torch.no_grad():
        # 進行預測
        outputs = model(features_tensor)
        _, predicted = torch.max(outputs, 1)
        predictions = predicted.cpu().numpy()
    
    # 將預測標籤轉為類型名稱
    genres = label_encoder.inverse_transform(predictions)
    
    # 返回最常見的類型
    unique, counts = np.unique(genres, return_counts=True)
    most_common_genre = unique[np.argmax(counts)]
    return most_common_genre
'''
# 主函數
def main():
    # 指定音頻檔案路徑
    # 請替換為您的音頻檔案路徑，例如 'your_music.au' 或 'path/to/your_music.mp3'
    audio_file = 'gui_test_music/pop_test1.mp3'

    # 檢查檔案是否存在
    if not os.path.exists(audio_file):
        print(f"File {audio_file} does not exist. Please update the path.")
        return

    # 設置設備（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加載標準化器和標籤編碼器
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
    except FileNotFoundError:
        print("Scaler or LabelEncoder not found. Please train the model using 'my_script.py' first.")
        return

    # 初始化模型
    input_dim = 42  # 特徵數量（MFCC、色度等）
    num_classes = len(GENRES)
    model = MusicGenreCNN(input_dim, num_classes).to(device)

    # 加載模型參數
    try:
        model.load_state_dict(torch.load('music_genre_cnn.pth', map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model file 'music_genre_cnn.pth' not found. Please train the model first.")
        return

    # 執行類型預測
    predicted_genre = predict_genre(audio_file, model, scaler, label_encoder, device)
    if predicted_genre:
        print(f"Predicted genre: {predicted_genre}")
    else:
        print("Unable to predict genre due to processing error.")

if __name__ == "__main__":
    main()'''