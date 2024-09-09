import os
import yt_dlp as youtube_dl

def download_audio(youtube_url, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Define options based on platform
    ydl_opts = {
        'outtmpl': os.path.join(target_dir, '%(title)s.mp3'),
        'format': 'bestaudio/best',
        'quiet': False,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '96',
        }],
    }
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        title = info_dict.get('title', 'Unknown Title')
        print(f'{title} audio track downloaded')
        return title

