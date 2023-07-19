from moviepy.editor import *
video = VideoFileClip("C:\\Users\\midori\\Downloads\\GR1-Record-2021-2022\\a80.mp4")
video.audio.write_audiofile("a80.mp3")
