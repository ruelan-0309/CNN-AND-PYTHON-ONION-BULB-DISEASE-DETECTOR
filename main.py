from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.uix.label import Label
import cv2
import base64
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
import numpy as np

class CameraApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera = cv2.VideoCapture(0)
        self.image = Image()
        self.add_widget(self.image)
        self.create_buttons()

    def create_buttons(self):
        button_layout = BoxLayout(orientation='horizontal')
        self.capture_button = Button(text="Take Photo", on_press=self.take_photo)
        button_layout.add_widget(self.capture_button)
        self.add_widget(button_layout)
        self.choose_button = Button(text="Choose Photo", on_press=self.choose_photo)
        self.add_widget(self.choose_button)

    def take_photo(self, instance):
        try:
            ret, frame = self.camera.read()
            ret, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer)
            self.image.source = f"data:image/jpeg;base64,{jpg_as_text.decode()}"
        except Exception as e:
            print(f"Error taking photo: {e}")

    def choose_photo(self, instance):
        file_chooser = FileChooserListView()
        file_chooser.bind(on_submit=self.load_image)
        popup = Popup(title="Select an image", content=file_chooser, size_hint=(0.9, 0.9))
        popup.open()

    def load_image(self, instance, selection):
        if selection:
            path, filename = selection[0]
            try:
                with open(path, "rb") as file:
                    jpg_as_text = base64.b64encode(file.read())
                    self.image.source = f"data:image/jpeg;base64,{jpg_as_text.decode()}"
            except Exception as e:
                print(f"Error loading image: {e}")

class DiseaseAlert(App):
    def build(self):
        return CameraApp()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selected_file = None
        self.image_source = Image()
        self.icon = "icon.png"


if __name__ == '__main__':
    cameraapp = DiseaseAlert()
    cameraapp.run()
