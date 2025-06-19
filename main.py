import os
import threading
import flet as ft
import pygame
from datetime import datetime
from Generate import *
from predict_genre import *

class MusicClassifierGUI:
    def __init__(self):
        self.selected_file = None
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.device = None
        self.is_playing = False
        pygame.mixer.init()
        
    def load_model_components(self):
        """載入模型相關組件"""
        try:
            # 設置設備
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 載入標準化器和標籤編碼器
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open('label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # 初始化並載入模型
            input_dim = 42
            num_classes = len(GENRES)
            self.model = MusicGenreCNN(input_dim, num_classes).to(self.device)
            self.model.load_state_dict(torch.load('music_genre_cnn.pth', map_location=self.device))
            
            return True, "模型載入成功！"
        except Exception as e:
            return False, f"模型載入失敗: {str(e)}"

class MusicGeneratorGUI:
    def __init__(self):
        self.model = None
        self.device = None
        self.is_playing = False
        self.generated_audio = None
        self.temp_file_path = None
        pygame.mixer.init()
        
        # 音樂類型選項
        self.music_genres = {
            'Blues': 'blues',
            'Classical': 'classical', 
            'Country': 'country',
            'Disco': 'disco',
            'Hip-hop': 'hiphop',
            'Jazz': 'jazz',
            'Metal': 'metal',
            'Pop': 'pop',
            'Regga': 'reggae',
            'Rock': 'rock'
        }

def main(page: ft.Page):
    def handle_nav_change(e):
        if e.control.selected_index == 0:
            page.controls.clear()
            page.title = "音樂分類器"
            page.window_width = 600
            page.window_height = 400
            page.theme_mode = ft.ThemeMode.LIGHT
            page.vertical_alignment = ft.MainAxisAlignment.START
            
            # 初始化分類器
            classifier = MusicClassifierGUI()
            
            # UI 組件
            title = ft.Text(
                "🎵 音樂分類器",
                size=32,
                weight=ft.FontWeight.BOLD,
                color=ft.Colors.BLUE_700
            )
            
            # 模型狀態顯示
            model_status = ft.Text(
                "正在載入模型...",
                size=14,
                color=ft.Colors.GREY_600
            )
            
            # 檔案選擇區域
            selected_file_text = ft.Text(
                "尚未選擇檔案",
                size=16,
                color=ft.Colors.GREY_600
            )
            
            def on_file_picked(e: ft.FilePickerResultEvent):
                if e.files:
                    classifier.selected_file = e.files[0].path
                    selected_file_text.value = f"已選擇: {os.path.basename(classifier.selected_file)}"
                    selected_file_text.color = ft.Colors.GREEN_700
                    play_button.disabled = False
                    classify_button.disabled = False
                else:
                    selected_file_text.value = "尚未選擇檔案"
                    selected_file_text.color = ft.Colors.GREY_600
                    play_button.disabled = True
                    classify_button.disabled = True
                page.update()
            
            file_picker = ft.FilePicker(on_result=on_file_picked)
            page.overlay.append(file_picker)
            
            pick_file_button = ft.ElevatedButton(
                "選擇音樂檔案",
                icon=ft.Icons.FOLDER_OPEN,
                on_click=lambda _: file_picker.pick_files(
                    dialog_title="選擇音樂檔案",
                    file_type=ft.FilePickerFileType.AUDIO
                ),
                style=ft.ButtonStyle(
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.BLUE_600
                )
            )
            
            # 音樂控制區域
            def toggle_music(e):
                if not classifier.selected_file:
                    return
                    
                if not classifier.is_playing:
                    try:
                        pygame.mixer.music.load(classifier.selected_file)
                        pygame.mixer.music.play()
                        classifier.is_playing = True
                        play_button.text = "停止播放"
                        play_button.icon = ft.Icons.STOP
                    except Exception as ex:
                        result_text.value = f"播放錯誤: {str(ex)}"
                        result_text.color = ft.Colors.RED
                else:
                    pygame.mixer.music.stop()
                    classifier.is_playing = False
                    play_button.text = "試聽音樂"
                    play_button.icon = ft.Icons.PLAY_ARROW
                page.update()
            
            play_button = ft.ElevatedButton(
                "試聽音樂",
                icon=ft.Icons.PLAY_ARROW,
                on_click=toggle_music,
                disabled=True,
                style=ft.ButtonStyle(
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.GREEN_600
                )
            )
            
            # 分類結果區域
            result_text = ft.Text(
                "",
                size=18,
                weight=ft.FontWeight.BOLD,
                text_align=ft.TextAlign.CENTER
            )
            
            progress_ring = ft.ProgressRing(visible=False)
            
            def classify_music(e):
                if not classifier.selected_file or not classifier.model:
                    return
                
                # 顯示載入動畫
                progress_ring.visible = True
                classify_button.disabled = True
                result_text.value = "正在分析音樂..."
                result_text.color = ft.Colors.BLUE_600
                page.update()
                
                def classify_thread():
                    try:
                        # 執行分類
                        predicted_genre = predict_genre(
                            classifier.selected_file,
                            classifier.model,
                            classifier.scaler,
                            classifier.label_encoder,
                            classifier.device
                        )
                        
                        if predicted_genre:
                            # 音樂類型中英文對照
                            genre_translation = {
                                'blues': '藍調',
                                'classical': '古典',
                                'country': '鄉村',
                                'disco': '迪斯可',
                                'hiphop': '嘻哈',
                                'jazz': '爵士',
                                'metal': '金屬',
                                'pop': '流行',
                                'reggae': '雷鬼',
                                'rock': '搖滾'
                            }
                            
                            chinese_genre = genre_translation.get(predicted_genre, predicted_genre)
                            result_text.value = f"🎵 預測結果: {chinese_genre} ({predicted_genre.upper()})"
                            result_text.color = ft.Colors.GREEN_700
                        else:
                            result_text.value = "❌ 分類失敗，請檢查音樂檔案"
                            result_text.color = ft.Colors.RED
                            
                    except Exception as ex:
                        result_text.value = f"❌ 分類錯誤: {str(ex)}"
                        result_text.color = ft.Colors.RED
                    
                    # 隱藏載入動畫
                    progress_ring.visible = False
                    classify_button.disabled = False
                    page.update()
                
                # 在背景執行分類
                threading.Thread(target=classify_thread, daemon=True).start()
            
            classify_button = ft.ElevatedButton(
                "開始分類",
                icon=ft.Icons.ANALYTICS,
                on_click=classify_music,
                disabled=True,
                style=ft.ButtonStyle(
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.ORANGE_600
                )
            )
            
            # 載入模型
            def load_model():
                success, message = classifier.load_model_components()
                model_status.value = message
                if success:
                    model_status.color = ft.Colors.GREEN_700
                else:
                    model_status.color = ft.Colors.RED
                page.update()
            
            # 在背景載入模型
            threading.Thread(target=load_model, daemon=True).start()
            
            # 頁面佈局
            page.add(
                ft.Container(
                    content=ft.Row([
                        ft.Container(
                            content=ft.Column([
                                # 標題區域
                                ft.Container(
                                    content=title,
                                    alignment=ft.alignment.center,
                                    padding=ft.padding.only(bottom=20)
                                )
                            ]
                            )
                        ),
                        
                        # 檔案選擇區域
                        ft.Card(
                            content=ft.Container(
                                content=ft.Column([
                                    ft.Text("📁 選擇音樂檔案", size=20, weight=ft.FontWeight.BOLD),
                                    pick_file_button,
                                    selected_file_text
                                ], spacing=15),
                                padding=15,
                                alignment=ft.alignment.center,
                                height=160
                            ),
                            elevation=3
                        ),
                        
                        # 音樂控制區域
                        ft.Card(
                            content=ft.Container(
                                content=ft.Column([
                                    ft.Text("🎧 音樂試聽", size=20, weight=ft.FontWeight.BOLD),
                                    play_button
                                ], spacing=15),
                                padding=15,
                                alignment=ft.alignment.center,
                                height=160
                            ),
                            elevation=3
                        ),
                        
                        # 分類區域
                        ft.Card(
                            content=ft.Container(
                                content=ft.Column([
                                    ft.Text("🤖 音樂分類", size=20, weight=ft.FontWeight.BOLD),
                                    classify_button,
                                    ft.Row([
                                        progress_ring,
                                        ft.Container(result_text, expand=True)
                                    ], alignment=ft.MainAxisAlignment.CENTER)
                                ], spacing=15),
                                padding=15,
                                alignment=ft.alignment.center,
                                height=160
                            ),
                            elevation=3
                        )
                        
                    ], spacing=20),
                    padding=30
                )
            )
        elif e.control.selected_index == 1:
            page.controls.clear()
            page.title = "音樂生成器"
            page.window_width = 600
            page.window_height = 400
            page.theme_mode = ft.ThemeMode.LIGHT
            page.vertical_alignment = ft.MainAxisAlignment.START
            
            # 初始化生成器
            generator = MusicGeneratorGUI()
            
            # UI 組件
            title = ft.Text(
                "🎵音樂生成器",
                size=32,
                weight=ft.FontWeight.BOLD,
                color=ft.Colors.BLUE_700
            )
            
            # 音樂類型選擇
            selected_genre = ft.Text(
                "請選擇音樂類型",
                size=16,
                color=ft.Colors.GREY_600
            )
            
            def on_genre_change(e):
                selected_genre.value = f"已選擇: {e.control.value}"
                selected_genre.color = ft.Colors.GREEN_700
                generate_button.disabled = False
                generator.generated_audio = rf'generated_music_by_genre\{e.control.value}_generated.wav'
                generator.temp_file_path = rf'generated_music_by_genre\{e.control.value}_generated.wav'
                page.update()
            
            genre_dropdown = ft.Dropdown(
                label="選擇音樂類型",
                options=[
                    ft.dropdown.Option(key=value, text=key) 
                    for key, value in generator.music_genres.items()
                ],
                on_change=on_genre_change,
                width=300
            )
            
            # 音樂控制區域
            def toggle_music(e):
                if not generator.generated_audio or not generator.temp_file_path:
                    return
                    
                if not generator.is_playing:
                    try:
                        pygame.mixer.music.load(generator.temp_file_path)
                        pygame.mixer.music.play()
                        generator.is_playing = True
                        play_button.text = "停止播放"
                        play_button.icon = ft.Icons.STOP
                    except Exception as ex:
                        result_text.value = f"播放錯誤: {str(ex)}"
                        result_text.color = ft.Colors.RED
                else:
                    pygame.mixer.music.stop()
                    generator.is_playing = False
                    play_button.text = "試聽音樂"
                    play_button.icon = ft.Icons.PLAY_ARROW
                page.update()
            
            play_button = ft.ElevatedButton(
                "試聽音樂",
                icon=ft.Icons.PLAY_ARROW,
                on_click=toggle_music,
                disabled=False,
                style=ft.ButtonStyle(
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.GREEN_600
                )
            )
            
            # 下載功能
            def download_music(e):
                if not generator.generated_audio or not generator.temp_file_path:
                    return
                def save_file(e: ft.FilePickerResultEvent):
                    if e.path:
                        try:
                            # 複製臨時檔案到選擇的位置
                            import shutil
                            shutil.copy2(generator.temp_file_path, e.path)
                            result_text.value = f"✅ 音樂已保存至: {os.path.basename(e.path)}"
                            result_text.color = ft.Colors.GREEN_700
                        except Exception as ex:
                            result_text.value = f"❌ 保存失敗: {str(ex)}"
                            result_text.color = ft.Colors.RED
                        page.update()
                
                save_dialog = ft.FilePicker(on_result=save_file)
                page.overlay.append(save_dialog)
                page.update()
                
                # 生成預設檔名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                genre_name = genre_dropdown.value if genre_dropdown.value else "music"
                default_name = f"generated_{genre_name}_{timestamp}.wav"
                
                save_dialog.save_file(
                    dialog_title="保存生成的音樂",
                    file_name=default_name,
                    file_type=ft.FilePickerFileType.AUDIO
                )
            
            download_button = ft.ElevatedButton(
                "下載音樂",
                icon=ft.Icons.DOWNLOAD,
                on_click=download_music,
                disabled=False,
                style=ft.ButtonStyle(
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.PURPLE_600
                )
            )
            
            # 生成結果區域
            result_text = ft.Text(
                "",
                size=18,
                weight=ft.FontWeight.BOLD,
                text_align=ft.TextAlign.CENTER
            )
            
            progress_ring = ft.ProgressRing(visible=False)
            
            def generate_music(e):
                generate_from_trained_model()
                
            generate_button = ft.ElevatedButton(
                "開始生成",
                icon=ft.Icons.MUSIC_NOTE,
                on_click=generate_music,
                disabled=True,
                style=ft.ButtonStyle(
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.ORANGE_600
                )
            )
            
            # 頁面佈局
            page.add(
                ft.Container(
                    content=ft.Row([
                        # 標題區域
                        ft.Container(
                            content=title,
                            alignment=ft.alignment.center,
                            padding=ft.padding.only(bottom=20)
                        ),
                        # 音樂類型選擇區域（對應原來的檔案選擇）
                        ft.Card(
                            content=ft.Container(
                                content=ft.Column([
                                    ft.Text("🎼 選擇音樂類型", size=20, weight=ft.FontWeight.BOLD),
                                    genre_dropdown,
                                    generate_button,
                                    ft.Row([
                                        progress_ring,
                                        ft.Container(result_text, expand=True)
                                    ], alignment=ft.MainAxisAlignment.CENTER)
                                ], spacing=15),
                                padding=15,
                                alignment=ft.alignment.center,
                                height=180
                            ),
                            elevation=3
                        ),
                        
                        # 音樂控制區域（對應原來的試聽區域）
                        ft.Card(
                            content=ft.Container(
                                content=ft.Column([
                                    ft.Text("🎧 音樂試聽", size=20, weight=ft.FontWeight.BOLD),
                                    ft.Row([
                                        play_button,
                                        download_button
                                    ], spacing=10, alignment=ft.MainAxisAlignment.CENTER)
                                ], spacing=15),
                                padding=15,
                                alignment=ft.alignment.center,
                                height=180
                            ),
                            elevation=3
                        )
                        
                    ], spacing=20),
                    padding=30
                )
            )
        elif e.control.selected_index == 2:
            page.controls.clear()
            page.title = "images"
            page.scroll = ft.ScrollMode.ALWAYS

            def image_with_caption(img_src, caption):
                return ft.Column(
                    [
                        ft.Image(
                            src=img_src,
                            width=450,
                            fit=ft.ImageFit.COVER
                        ),
                        ft.Text(
                            caption,
                            size=14,
                            text_align=ft.TextAlign.CENTER
                        ),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=10
                )

            page.add(
                ft.Column(
                    [
                        ft.Row(
                            [
                                image_with_caption(r"img\cnn_accuracy_curves.png", "classification accuracy curves"),
                                image_with_caption(r"img\cnn_loss_curves.png", "classification loss curves"),
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                            spacing=40
                        ),
                        ft.Column(
                            [
                                ft.Image(
                                    src=r'img\cnn_confusion_matrix.png',
                                    width=550,
                                    fit=ft.ImageFit.COVER
                                ),
                                ft.Text(
                                    "classification confusion matrix",
                                    size=14,
                                    text_align=ft.TextAlign.CENTER
                                ),
                            ],
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            spacing=10
                        ),
                        ft.Column(
                            [
                                ft.Image(
                                    src=r"img\gtzan_spectrogram_samples.png",
                                    width=700,
                                    fit=ft.ImageFit.COVER
                                ),
                                ft.Text(
                                    "spectrogram samples",
                                    size=14,
                                    text_align=ft.TextAlign.CENTER
                                ),
                            ],
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            spacing=10
                        ),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,  # 這裡要有逗號
                    spacing=10
                )
            )
        page.update()

    page.title = "Music classification & generation app"
    page.navigation_bar = ft.NavigationBar(
        on_change=handle_nav_change,
        destinations=[
            ft.NavigationBarDestination(icon=ft.Icons.CLASS_, label="Music Classification"),
            ft.NavigationBarDestination(icon=ft.Icons.MUSIC_VIDEO, label="Music Generate"),
            ft.NavigationBarDestination(icon=ft.Icons.IMAGE_SEARCH, label="Visualization"),
        ],
    )
    page.theme_mode = ft.ThemeMode.LIGHT
    page.add(ft.Text("選擇下方功能"))

ft.app(target=main)
