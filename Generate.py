import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import soundfile as sf
import os

class CausalConv1d(nn.Module):
    """因果卷積層 - 確保生成時不會看到未來的信息"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        # 移除右側的padding以保持因果性
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class ResidualBlock(nn.Module):
    """WaveNet的核心殘差塊"""

    def __init__(self, residual_channels, gate_channels, skip_channels,
                 kernel_size, dilation, condition_channels=None):
        super(ResidualBlock, self).__init__()

        self.dilated_conv = CausalConv1d(residual_channels, gate_channels,
                                        kernel_size, dilation)

        # 條件輸入（如音樂風格）
        self.condition_channels = condition_channels
        if condition_channels:
            self.condition_conv = nn.Conv1d(condition_channels, gate_channels, 1)

        # 門控機制
        self.output_conv = nn.Conv1d(gate_channels // 2, residual_channels, 1)
        self.skip_conv = nn.Conv1d(gate_channels // 2, skip_channels, 1)

    def forward(self, x, condition=None):
        # 殘差連接的輸入
        residual = x

        # 膨脹卷積
        x = self.dilated_conv(x)

        # 添加條件信息
        if condition is not None and self.condition_channels:
            condition = self.condition_conv(condition)
            x = x + condition

        # 門控激活：tanh * sigmoid
        tanh_out, sigmoid_out = torch.chunk(x, 2, dim=1)
        x = torch.tanh(tanh_out) * torch.sigmoid(sigmoid_out)

        # 輸出分支
        skip = self.skip_conv(x)
        output = self.output_conv(x)

        return (output + residual), skip


class WaveNet(nn.Module):
    """WaveNet音樂生成模型"""

    def __init__(self,
                 layers=10,           # 每個stack的層數
                 stacks=3,            # stack數量
                 residual_channels=64, # 殘差通道數
                 gate_channels=128,    # 門控通道數
                 skip_channels=64,     # 跳躍連接通道數
                 output_channels=256,  # 輸出通道數（量化級別）
                 kernel_size=2,        # 卷積核大小
                 condition_channels=None,  # 條件通道數
                 num_genres=10):       # 音樂類型數量

        super(WaveNet, self).__init__()

        self.layers = layers
        self.stacks = stacks
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.output_channels = output_channels
        self.num_genres = num_genres

        # 計算感受野大小
        self.receptive_field = self.calculate_receptive_field()

        # 輸入卷積層（將one-hot編碼的音頻映射到殘差通道）
        self.start_conv = CausalConv1d(output_channels, residual_channels, 1)

        # 類型嵌入層
        self.genre_embedding = nn.Embedding(num_genres, residual_channels)

        # 殘差塊
        self.residual_blocks = nn.ModuleList()
        for stack in range(stacks):
            for layer in range(layers):
                dilation = 2 ** layer
                block = ResidualBlock(
                    residual_channels=residual_channels,
                    gate_channels=gate_channels,
                    skip_channels=skip_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    condition_channels=residual_channels if condition_channels else None
                )
                self.residual_blocks.append(block)

        # 最終輸出層
        self.end_conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.end_conv2 = nn.Conv1d(skip_channels, output_channels, 1)

    def calculate_receptive_field(self):
        """計算模型的感受野大小"""
        receptive_field = 1
        for stack in range(self.stacks):
            for layer in range(self.layers):
                dilation = 2 ** layer
                receptive_field += dilation
        return receptive_field

    def forward(self, x, genre=None):
        # x shape: (batch_size, quantization_levels, seq_len) - one-hot編碼
        # genre shape: (batch_size,) - 類型標籤

        batch_size, _, seq_len = x.size()

        # 輸入卷積
        x = self.start_conv(x)

        # 準備類型條件
        condition = None
        if genre is not None:
            # 獲取類型嵌入並擴展到序列長度
            genre_emb = self.genre_embedding(genre)  # (batch_size, residual_channels)
            genre_emb = genre_emb.unsqueeze(2).expand(-1, -1, seq_len)  # (batch_size, residual_channels, seq_len)
            condition = genre_emb

        # 累積跳躍連接
        skip_connections = []

        # 通過所有殘差塊
        for block in self.residual_blocks:
            x, skip = block(x, condition)
            skip_connections.append(skip)

        # 合併跳躍連接
        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0)

        # 最終輸出層
        x = F.relu(skip_sum)
        x = F.relu(self.end_conv1(x))
        x = self.end_conv2(x)

        return x


class MusicWaveNetDataset(Dataset):
    """音樂生成的數據集類"""

    def __init__(self, processed_data, sequence_length=8000, quantization_levels=256):
        self.sequence_length = sequence_length
        self.quantization_levels = quantization_levels

        # 從處理後的數據中獲取原始音頻
        self.audio_data = []
        self.genre_labels = []

        # 保存標籤編碼器和類型名稱
        self.label_encoder = processed_data['label_encoder']
        # 確保 genre_names 是列表而不是 numpy 數組
        if hasattr(self.label_encoder.classes_, 'tolist'):
            self.genre_names = self.label_encoder.classes_.tolist()
        else:
            self.genre_names = list(self.label_encoder.classes_)

        # 重新加載原始音頻文件用於生成
        self.load_audio_for_generation(processed_data)

    def load_audio_for_generation(self, processed_data):
        """重新加載原始音頻數據用於生成"""
        print("為WaveNet加載原始音頻數據...")

        file_info = processed_data['file_info']
        label_encoder = processed_data['label_encoder']

        # 限制文件數量以避免內存問題
        max_files = min(50, len(file_info))

        for info in tqdm(file_info[:max_files], desc="加載音頻"):
            try:
                # 加載原始音頻，縮短長度以節省內存
                audio, sr = librosa.load(info['path'], sr=22050, duration=10)  # 只用10秒

                # 量化音頻到指定級別
                audio_quantized = self.quantize_audio(audio)

                # 分割成序列
                sequences = self.create_sequences(audio_quantized)

                # 獲取類型標籤
                genre = info['genre']
                genre_idx = label_encoder.transform([genre])[0]

                for seq in sequences:
                    self.audio_data.append(seq)
                    self.genre_labels.append(genre_idx)

            except Exception as e:
                print(f"處理文件出錯 {info['file']}: {e}")

        print(f"共生成 {len(self.audio_data)} 個訓練序列")
        print(f"類型分佈: {dict(zip(*np.unique(self.genre_labels, return_counts=True)))}")

    def quantize_audio(self, audio):
        """將音頻量化到指定級別"""
        # 歸一化到[-1, 1]
        audio = np.clip(audio, -1, 1)

        # 量化到[0, quantization_levels-1]
        audio_quantized = np.round((audio + 1) * (self.quantization_levels - 1) / 2)
        audio_quantized = audio_quantized.astype(np.int64)

        return audio_quantized

    def create_sequences(self, audio_quantized):
        """將音頻分割成訓練序列"""
        sequences = []

        # 滑動窗口創建序列
        step_size = max(1, self.sequence_length // 4)  # 增加重疊以獲得更多數據
        for i in range(0, len(audio_quantized) - self.sequence_length, step_size):
            seq = audio_quantized[i:i + self.sequence_length]
            if len(seq) == self.sequence_length:
                sequences.append(seq)

        return sequences

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        sequence = self.audio_data[idx]
        genre = self.genre_labels[idx]

        # 輸入和目標（目標是輸入向右偏移一位）
        x = torch.LongTensor(sequence[:-1])
        y = torch.LongTensor(sequence[1:])

        return x, y, genre


class MusicGenerator:
    """音樂生成器"""

    def __init__(self, model, quantization_levels=256, sample_rate=22050, genre_names=None):
        self.model = model
        self.quantization_levels = quantization_levels
        self.sample_rate = sample_rate

        # 處理 genre_names，確保它是一個列表
        if genre_names is None:
            self.genre_names = [f"Genre_{i}" for i in range(model.num_genres)]
        else:
            # 如果是 numpy 數組，轉換為列表
            if hasattr(genre_names, 'tolist'):
                self.genre_names = genre_names.tolist()
            else:
                self.genre_names = list(genre_names)

        self.device = next(model.parameters()).device

    def generate(self, length=22050, temperature=1.0, genre=None, seed=None):
        """生成音頻序列"""
        self.model.eval()

        if seed is not None:
            torch.manual_seed(seed)

        # 初始化序列
        generated = torch.zeros(1, 1, self.model.receptive_field,
                              dtype=torch.long, device=self.device)

        # 準備類型條件
        genre_tensor = None
        if genre is not None:
            genre_tensor = torch.LongTensor([genre]).to(self.device)

        generated_samples = []

        genre_name = self.genre_names[genre] if genre is not None else "無條件"
        print(f"生成 {length} 個樣本 (類型: {genre_name})...")

        with torch.no_grad():
            for i in tqdm(range(length)):
                # 準備輸入
                x = self.one_hot_encode(generated)  # (batch, channels, time)

                # 前向傳播
                logits = self.model(x, genre_tensor)

                # 獲取最後一個時間步的預測
                logits = logits[:, :, -1] / temperature

                # 采樣
                probs = F.softmax(logits, dim=1)
                sample = torch.multinomial(probs, 1).squeeze()

                # 添加到生成序列
                generated_samples.append(sample.item())

                # 更新輸入序列
                next_sample = sample.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                generated = torch.cat([generated[:, :, 1:], next_sample], dim=2)

        # 反量化到音頻
        audio = self.dequantize_audio(generated_samples)
        return audio

    def generate_all_genres(self, length=22050, temperature=1.0, seed=None, output_dir="generated_music"):
        """為所有類型生成音樂"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        generated_files = []

        print(f"開始為 {len(self.genre_names)} 個類型生成音樂...")

        for genre_idx, genre_name in enumerate(self.genre_names):
            print(f"\n正在生成 {genre_name} 類型音樂...")

            # 為每個類型使用不同的種子
            current_seed = seed + genre_idx if seed is not None else None

            # 生成音頻
            audio = self.generate(
                length=length,
                temperature=temperature,
                genre=genre_idx,
                seed=current_seed
            )

            # 保存文件
            filename = os.path.join(output_dir, f"{genre_name}_generated.wav")
            self.save_audio(audio, filename)
            generated_files.append(filename)

        print(f"\n✅ 已為所有類型生成音樂，文件保存在: {output_dir}")
        return generated_files

    def one_hot_encode(self, x):
        """將量化的音頻轉換為one-hot編碼"""
        # x shape: (batch, 1, time)
        batch_size, _, time_steps = x.size()

        # 創建one-hot編碼
        one_hot = torch.zeros(batch_size, self.quantization_levels, time_steps,
                             device=x.device)
        one_hot.scatter_(1, x, 1)

        return one_hot

    def dequantize_audio(self, quantized_samples):
        """將量化的樣本轉換回音頻"""
        # 從[0, quantization_levels-1]轉換到[-1, 1]
        audio = np.array(quantized_samples, dtype=np.float32)
        audio = (audio * 2 / (self.quantization_levels - 1)) - 1

        return audio

    def save_audio(self, audio, filename, sample_rate=None):
        """保存生成的音頻"""
        if sample_rate is None:
            sample_rate = self.sample_rate

        sf.write(filename, audio, sample_rate)
        print(f"音頻已保存到: {filename}")


class WaveNetTrainer:
    """WaveNet訓練器"""

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None

        print(f"WaveNet訓練器初始化完成")
        print(f"設備: {self.device}")
        print(f"模型參數數量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"感受野大小: {self.model.receptive_field}")

    def setup_training(self, learning_rate=0.001, weight_decay=1e-5):
        """設置訓練參數"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    def train_epoch(self, train_loader):
        """訓練一個epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Training WaveNet')
        for batch_idx, (x, y, genre) in enumerate(pbar):
            x, y, genre = x.to(self.device), y.to(self.device), genre.to(self.device)

            # 轉換為one-hot編碼
            x_onehot = self.one_hot_encode(x)

            self.optimizer.zero_grad()

            # 前向傳播（帶類型條件）
            logits = self.model(x_onehot, genre)

            # 重塑輸出用於損失計算
            logits = logits.permute(0, 2, 1).contiguous()
            logits = logits.view(-1, logits.size(-1))
            y = y.view(-1)

            # 計算損失
            loss = self.criterion(logits, y)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 統計
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        return total_loss / len(train_loader), 100. * correct / total

    def one_hot_encode(self, x):
        """將量化的音頻轉換為one-hot編碼"""
        batch_size, seq_len = x.size()

        # 創建one-hot編碼
        one_hot = torch.zeros(batch_size, self.model.output_channels, seq_len,
                             device=x.device)
        one_hot.scatter_(1, x.unsqueeze(1), 1)

        return one_hot

    def train(self, train_loader, epochs=10, save_interval=5):
        """完整訓練流程"""
        if self.optimizer is None:
            self.setup_training()

        train_losses = []
        train_accs = []

        print(f"\n開始訓練WaveNet {epochs} 個epochs...")

        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')

            # 訓練
            train_loss, train_acc = self.train_epoch(train_loader)

            train_losses.append(train_loss)
            train_accs.append(train_acc)

            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

            # 定期保存模型
            if (epoch + 1) % save_interval == 0:
                torch.save(self.model.state_dict(), f'wavenet_epoch_{epoch+1}.pth')
                print(f'模型已保存: wavenet_epoch_{epoch+1}.pth')

        # 保存最終模型
        torch.save(self.model.state_dict(), 'wavenet_final.pth')

        return {
            'train_losses': train_losses,
            'train_accs': train_accs
        }

'''
# 使用示例
def main():
    # 載入處理後的數據
    print("載入數據...")
    with open('gtzan_processed_data.pkl', 'rb') as f:
        processed_data = pickle.load(f)

    # 創建WaveNet數據集 - 使用較小的序列長度
    dataset = MusicWaveNetDataset(processed_data, sequence_length=2000)

    # 創建數據加載器 - 減小批次大小
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 獲取類型數量
    num_genres = len(dataset.genre_names)
    print(f"檢測到 {num_genres} 個音樂類型: {dataset.genre_names}")

    # 創建WaveNet模型 - 使用較小的參數
    model = WaveNet(
        layers=6,              # 減少層數
        stacks=2,              # 減少堆疊數
        residual_channels=32,  # 減少通道數
        gate_channels=64,      # 減少通道數
        skip_channels=32,      # 減少通道數
        output_channels=256,
        kernel_size=2,
        num_genres=num_genres
    )

    print(f"數據集大小: {len(dataset)}")
    print(f"批次數: {len(train_loader)}")

    # 訓練模型
    trainer = WaveNetTrainer(model)
    history = trainer.train(train_loader, epochs=5)  # 減少訓練輪數

    # 生成音樂
    print("\n開始為每個類型生成音樂...")
    generator = MusicGenerator(model, genre_names=dataset.genre_names)

    # 為每個類型生成3秒的音頻
    generated_files = generator.generate_all_genres(
        length=22050 * 3,  # 3秒
        temperature=0.8,
        seed=42,
        output_dir="generated_music_by_genre"
    )

    print(f"\n✅ 音樂生成完成！生成了 {len(generated_files)} 個文件")
    for file in generated_files:
        print(f"  - {file}")'''


def generate_from_trained_model(model_path='wavenet_final.pth',
                               data_path='gtzan_processed_data.pkl'):
    """從已訓練的模型生成音樂"""
    # 載入數據以獲取類型信息
    with open(data_path, 'rb') as f:
        processed_data = pickle.load(f)

    dataset = MusicWaveNetDataset(processed_data, sequence_length=2000)
    num_genres = len(dataset.genre_names)

    # 重建模型結構
    model = WaveNet(
        layers=6,
        stacks=2,
        residual_channels=32,
        gate_channels=64,
        skip_channels=32,
        output_channels=256,
        kernel_size=2,
        num_genres=num_genres
    )

    # 載入訓練好的權重
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    print(f"已載入訓練好的模型: {model_path}")

    # 生成音樂
    generator = MusicGenerator(model, genre_names=dataset.genre_names)
    generated_files = generator.generate_all_genres(
        length=22050 * 5,  # 5秒
        temperature=0.8,
        seed=42,
        output_dir="generated_music_by_genre"
    )

    return generated_files