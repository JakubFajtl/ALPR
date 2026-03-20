from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips

### IMPORTANT ###
### If you want to run this file you might have to add `moviepy` to your requirements.txt file
### Or you can just do `pip install moviepy`

input_video_path = "dataset/trainingsvideo.avi"
output_video_path = "dataset/my_output_video.avi"

video_clip = VideoFileClip(input_video_path)

# If you want to make your own shorter vides you can change the start(left) / end(right) times here
# You can also add more, currently it takes 4 sub-clips and merges them
cutting_times = [(0, 3), (94, 98), (132.5, 135), (147.5, 150)]

clips = [video_clip.subclip(start_time_seconds, end_time_seconds)
         for (start_time_seconds, end_time_seconds) in cutting_times]

merged_clip = concatenate_videoclips(clips)

merged_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")


