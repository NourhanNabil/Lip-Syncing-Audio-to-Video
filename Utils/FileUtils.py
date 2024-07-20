import os 
import cv2 

class FileUtils:
    
    @classmethod
    def save_audio_video(cls, file_name: str, file_content: bytes):
        with open(file_name,"wb") as file:
            file.write(file_content)
        return file_name
    
    @classmethod
    def get_file_type(cls, video_path: str):
        _, ext = os.path.splitext(video_path)

        if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            return 'image'
        elif ext.lower() in ['.avi', '.mp4', '.mov', '.flv', '.mkv']:
            return 'video'
        else:
            return 'unsupported'
    
    @classmethod
    def get_video_fps(cls, video_path: str):
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        return fps