import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import json
from .shell import run  # Import your custom language interpreter function
from bs4 import BeautifulSoup
from sympy import factorint
import numpy as np
from io import StringIO
from decimal import Decimal, getcontext
import math
import io
import base64
import decimal
import matplotlib.pyplot as plt
import sympy as sp
from mpmath import mp
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime, timedelta
import time
getcontext().prec = decimal.MAX_PREC
mp.dps = decimal.MAX_PREC
from functools import reduce
import numpy as np
from math import gcd
import sys
from .text_morse import morse_translate, reverse_morse_translate
# views.py
from .jvdict import Jvdict
from .transliteratejav import transliterate
from .aksara import dotransliterate
from deep_translator import GoogleTranslator
from langdetect import detect
import roman
import textwrap
import os
from math import factorial
from itertools import permutations
from langdetect.lang_detect_exception import LangDetectException
# Set the path to the tessdata directory
from PIL import Image, ImageDraw, ImageFont
import os
from ast import keyword

import base64
#import torch
#import cv2
from django.views.decorators.csrf import csrf_exempt

import numpy as np
from django.http import JsonResponse
#import torch
from PIL import Image, ImageFilter
import argparse

#from models.module_photo2pixel import Photo2PixelModel
#from utils import img_common_util
from io import BytesIO
from django.http import StreamingHttpResponse
from django.shortcuts import render
#from .camera import VideoCamera
#from torchvision.transforms import ToTensor
#from torchvision.transforms.functional import resize
from numpy.linalg import inv
from alquran_id import AlQuran as Quran


from numpy.linalg import solve
# views.py
import matplotlib.pyplot as plt
from sympy import sympify, simplify
from django.shortcuts import render
import numpy as np
import sympy as sp
import re
import binascii
# views.py
from django.http import HttpResponse
from django.conf import settings
import os
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.http import JsonResponse
import os
from django.conf import settings
from django.http import HttpResponse, HttpResponseNotFound
from django.views.generic import View

import os
from django.conf import settings
from django.http import HttpResponse
from django.views import View
# importing models and libraries
from django.shortcuts import render
from .models import posts
from django.views import generic
from django.views.decorators.http import require_GET
from django.http import HttpResponse
# views.py
from django.core.mail import send_mail
from django.http import HttpResponse
from django.shortcuts import render

from django.shortcuts import render, get_object_or_404
from .models import Quiz, Question, Choice
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.shortcuts import render
from sympy import symbols, solve
import matplotlib.pyplot as plt
import io
import urllib, base64
from django.shortcuts import render
from sympy import symbols, solve, Eq
import matplotlib.pyplot as plt
import io
import urllib, base64
from django.shortcuts import render
from sympy import symbols, solve, Eq
import matplotlib.pyplot as plt
import io
import urllib, base64

from django.shortcuts import render
from sympy import symbols, solve, Eq, sin, log, sqrt
import matplotlib.pyplot as plt
import numpy as np
import io
import urllib, base64
# calculus/views.py
# calculus/views.py
# calculus/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.shortcuts import render
import sympy as sp
from django.shortcuts import render
import yt_dlp

def spotify_to_youtube(request):
    error_message = ""
    title = ""
    downloads = []

    if request.method == 'POST':
        # Retrieve the Spotify links input
        spotify_links_input = request.POST.get('spotify_links')

        if spotify_links_input:
            # Split the input into lines, assuming each line is a separate link
            spotify_links = spotify_links_input.splitlines()

            if spotify_links:
                try:
                    for spotify_link in spotify_links:
                        # Use a placeholder for actual track-to-YouTube URL conversion
                        youtube_search_url = convert_spotify_to_youtube(spotify_link)

                        # Download using yt-dlp
                        ydl_opts = {
                            'format': 'bestaudio/best',
                            'outtmpl': '%(title)s.%(ext)s',
                        }

                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            info_dict = ydl.extract_info(youtube_search_url, download=False)
                            if 'entries' in info_dict:
                                for entry in info_dict['entries']:
                                    downloads.append({
                                        'title': entry.get('title'),
                                        'url': entry.get('webpage_url')
                                    })
                            else:
                                error_message = "No entries found in the search results."
                except Exception as e:
                    error_message = f"Error: {str(e)}"
            else:
                error_message = "No valid Spotify links found in the input."
        else:
            error_message = "Please enter valid Spotify links."

    context = {
        'title': title,
        'downloads': downloads,
        'error_message': error_message,
    }

    return render(request, 'spotify.html', context)

def convert_spotify_to_youtube(spotify_link):
    # Placeholder for actual track-to-YouTube conversion
    # In practice, you would need to search YouTube for each track
    return "https://www.youtube.com/results?search_query=" + spotify_link

def derivative_view(request):
    return render(request, 'diff.html')

def process_function(request):
    if request.method == "POST":
        function_input = request.POST.get('function', '')
        step = request.POST.get('step', '')

        response_data = {
            'error': None,
            'result': '',
        }

        try:
            x = sp.symbols('x')
            func = sp.sympify(function_input)

            if step == 'diff':
                result = sp.diff(func, x)
                response_data['result'] = str(result)
            elif step == 'simplify':
                simplified = sp.simplify(func)
                response_data['result'] = str(simplified)
            else:
                response_data['result'] = str(func)
        except Exception as e:
            response_data['error'] = str(e)

        return JsonResponse(response_data)

@csrf_exempt
def solve_inequality(request):
    x = symbols('x')
    solution = None
    plot_url = None

    if request.method == 'POST':
        expr = request.POST.get('expression')
        solve_type = request.POST.get('solve_type')

        try:
            if expr:
                if solve_type == 'inequality':
                    # Solve as an inequality
                    solution = solve(expr, x)
                elif solve_type == 'equation':
                    # Ensure expression contains '='
                    if '=' in expr:
                        lhs, rhs = expr.split('=')
                        lhs = lhs.strip()
                        rhs = rhs.strip()

                        # Handle special cases for trigonometric functions
                        if 'sin' in lhs or 'cos' in lhs or 'tan' in lhs:
                            lhs = lhs.replace('^', '**')  # Replace ^ with **
                            rhs = rhs.replace('^', '**')  # Replace ^ with **

                        # Construct the equation
                        equation = Eq(eval(lhs), eval(rhs))
                        solution = solve(equation, x)
                    else:
                        solution = "Error: The input does not contain an '=' sign."

                # Generate plot
                fig, ax = plt.subplots()
                ax.axhline(0, color='black', lw=2)
                ax.axvline(0, color='black', lw=2)

                x_vals = np.linspace(-10, 10, 400)  # Use numpy for numerical ranges
                y_vals = []


                try:
                    if solve_type == 'inequality':
                        # Evaluate expression for each x value
                        for val in x_vals:
                            try:
                                y_val = float(expr.subs(x, val))
                            except Exception:
                                y_val = float('nan')  # Handle cases where evaluation fails
                            y_vals.append(y_val)
                        ax.plot(x_vals, y_vals, label=str(expr))
                        ax.fill_between(x_vals, y_vals, where=[val < 0 for val in y_vals], alpha=0.3)

                    elif solve_type == 'equation':
                        # Use lambda to evaluate the numerical function
                        lhs_func = lambda x_val: eval(lhs.replace('x', str(x_val)))
                        rhs_func = lambda x_val: eval(rhs.replace('x', str(x_val)))
                        y_vals = [lhs_func(val) - rhs_func(val) for val in x_vals]
                    ax.plot(x_vals, y_vals, label=expr)
                    ax.fill_between(x_vals, y_vals, where=[val < 0 for val in y_vals], alpha=0.3)
                except Exception as e:
                    solution = f"Error in plotting: {str(e)}"

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                plot_url = urllib.parse.quote(string)
        except Exception as e:
            solution = f"Error: {str(e)}"

    return render(request, 'solve_inequality.html', {
        'solution': solution,
        'plot_url': plot_url,
    })

from django.shortcuts import render, redirect
from django.http import HttpResponse


import os
from django.shortcuts import render, redirect
from django.http import HttpResponse
import numpy as np
from .utils import check_winner, check_draw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from io import StringIO, BytesIO
import base64
def process_csv(data):
    try:
        # Convert the CSV data into a DataFrame with comma separation
        df = pd.read_csv(StringIO(data), header=None)
    except Exception as e:
        raise ValueError(f"Error reading CSV data: {e}")

    if df.empty:
        raise ValueError("CSV data is empty.")
    
    if df.shape[1] == 0:
        raise ValueError("CSV data has no columns.")
    
    # Assuming data is in the first row, and converting all values to float
    data = pd.Series(df.iloc[0].dropna()).astype(float)  # Convert to float for numerical operations
    
    if data.empty:
        raise ValueError("Column data is empty after dropping NaN values.")
    
    # Statistik
    mean = data.mean()
    mode = data.mode().tolist()
    median = data.median()
    
    try:
        quartiles = np.percentile(data, [25, 50, 75])
        deciles = np.percentile(data, [10 * i for i in range(1, 10)])
        percentiles = np.percentile(data, [i for i in range(1, 100)])
    except IndexError as e:
        raise ValueError(f"Error computing percentiles: {e}")

    # Visualizations
    plots = {
        'histogram': plot_histogram(data),
        'pie_chart': plot_pie_chart(data),
        'frequency_table': plot_frequency_table(data)
    }

    return {
        'mean': mean,
        'mode': mode,
        'median': median,
        'quartiles': quartiles,
        'deciles': deciles,
        'percentiles': percentiles,
        'plots': plots
    }

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO





def plot_histogram(data):
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins='auto', alpha=0.7, color='blue', edgecolor='black')
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    return fig_to_base64()

def plot_pie_chart(data):
    counts = data.value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=plt.cm.Paired(range(len(counts))))
    plt.title('Pie Chart')
    return fig_to_base64()

def plot_frequency_table(data):
    counts = data.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    table_data = pd.DataFrame({'Value': counts.index, 'Frequency': counts.values})
    table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    return fig_to_base64()


def fig_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return img_str
    
@csrf_exempt
def statis(request):
    return render(request, 'statistics/home.html')

@csrf_exempt
def stats(request):
    if request.method == 'POST':
        csv_data = request.POST.get('csv_data', '')
        if csv_data:
            try:
                stats = process_csv(csv_data)
                return render(request, 'statistics/stats.html', {'stats': stats})
            except ValueError as e:
                return render(request, 'statistics/stats.html', {'error': str(e)})
        else:
            return render(request, 'statistics/stats.html', {'error': 'No CSV data provided.'})
    return render(request, 'statistics/stats.html')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import base64
import pandas as pd
import numpy as np
from io import StringIO

def process_interval_data(data):
    try:
        # Read and process the CSV data
        df = pd.read_csv(StringIO(data), header=None)
        df.columns = ['Interval', 'Frequency']
        df[['Lower', 'Upper']] = df['Interval'].str.split('-', expand=True).astype(float)
        df['Frequency'] = df['Frequency'].astype(float)
        
        # Flatten the data according to frequency
        processed_data = []
        for _, row in df.iterrows():
            processed_data.extend(np.linspace(row['Lower'], row['Upper'], num=int(row['Frequency'])).tolist())
        processed_data = pd.Series(processed_data)
        
        # Calculate statistics
        mean = processed_data.mean()
        mode = processed_data.mode()
        median = processed_data.median()
        quartiles = np.percentile(processed_data, [25, 50, 75])
        deciles = np.percentile(processed_data, [10 * i for i in range(1, 10)])
        percentiles = np.percentile(processed_data, [i for i in range(1, 100)])

        # Generate plots
        plots = {
            'histogram': plot_histogram_interval(processed_data, df[['Lower', 'Upper', 'Frequency']]),
            'pie_chart': plot_pie_chart_interval(df[['Lower', 'Upper', 'Frequency']]),
            'frequency_table': plot_frequency_table_interval(df[['Lower', 'Upper', 'Frequency']])
        }

        return {
            'mean': mean,
            'mode': mode,
            'median': median,
            'quartiles': quartiles,
            'deciles': deciles,
            'percentiles': percentiles,
            'plots': plots
        }
    except Exception as e:
        raise ValueError(f"Error processing data: {e}")

def plot_histogram_interval(data, intervals_df):
    plt.figure(figsize=(10, 6))
    # Create histogram bins
    bin_edges = [row['Lower'] for _, row in intervals_df.iterrows()] + [intervals_df['Upper'].iloc[-1]]
    plt.hist(data, bins=bin_edges, alpha=0.7, color='blue', edgecolor='black', weights=np.ones_like(data) * (1.0 / len(data)))
    plt.title('Histogram of Interval Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    return fig_to_base64()

def plot_pie_chart_interval(intervals_df):
    plt.figure(figsize=(8, 6))
    # Calculate frequencies
    frequencies = intervals_df['Frequency'].tolist()
    labels = [f'{row["Lower"]}-{row["Upper"]}' for _, row in intervals_df.iterrows()]
    plt.pie(frequencies, labels=labels, autopct='%1.1f%%', colors=plt.cm.Paired(range(len(frequencies))), startangle=140)
    plt.title('Pie Chart of Interval Frequencies')
    return fig_to_base64()

def plot_frequency_table_interval(intervals_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    # Prepare frequency data
    table_data = pd.DataFrame({
        'Interval': [f'{row["Lower"]}-{row["Upper"]}' for _, row in intervals_df.iterrows()],
        'Frequency': intervals_df['Frequency']
    })
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    plt.title('Frequency Table')
    return fig_to_base64()

def fig_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return img_str


@csrf_exempt
def home(request):
    return render(request, 'home.html')

@csrf_exempt
def interval_stats(request):
    if request.method == 'POST':
        csv_data = request.POST.get('csv_data', '')
        if csv_data:
            try:
                stats = process_interval_data(csv_data)
                return render(request, 'statistics/interval_stats.html', {'stats': stats})
            except ValueError as e:
                return render(request, 'statistics/interval_stats.html', {'error': str(e)})
        else:
            return render(request, 'statistics/interval_stats.html', {'error': 'No CSV data provided.'})
    return render(request, 'statistics/interval_stats.html')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from io import StringIO, BytesIO
import base64
def process_interval_data_detailed(data):
    try:
        # Convert the CSV data into a DataFrame with comma separation
        df = pd.read_csv(StringIO(data), header=None)
    except Exception as e:
        raise ValueError(f"Error reading CSV data: {e}")

    if df.empty:
        raise ValueError("CSV data is empty.")
    
    if df.shape[1] == 0:
        raise ValueError("CSV data has no columns.")

    # Flatten and process the data
    processed_data = []
    intervals = []
    for entry in df[0]:
        if '-' in entry:
            # Handle range data
            start, end = map(float, entry.split('-'))
            intervals.append((start, end))
            processed_data.extend(np.linspace(start, end, num=10).tolist())
        else:
            # Handle individual values
            processed_data.append(float(entry))

    processed_data = pd.Series(processed_data)
    
    if processed_data.empty:
        raise ValueError("Data is empty after processing.")
    
    # Calculate statistics
    mean = processed_data.mean()
    mode = processed_data.mode()
    median = processed_data.median()
    
    range_ = processed_data.max() - processed_data.min()
    n = len(processed_data)
    
    # Calculate class intervals
    intervals = [(start, end) for start, end in intervals]
    class_width = [end - start for start, end in intervals]
    class_boundaries = [(start - 0.5 * width, end + 0.5 * width) for (start, end), width in zip(intervals, class_width)]
    midpoints = [(start + end) / 2 for start, end in intervals]

    try:
        quartiles = np.percentile(processed_data, [25, 50, 75])
        deciles = np.percentile(processed_data, [10 * i for i in range(1, 10)])
        percentiles = np.percentile(processed_data, [i for i in range(1, 100)])
    except IndexError as e:
        raise ValueError(f"Error computing percentiles: {e}")

    # Return detailed statistics without plots
    return {
        'mean': mean,
        'mode': mode,
        'median': median,
        'range': range_,
        'n': n,
        'class_intervals': intervals,
        'class_width': class_width,
        'class_boundaries': class_boundaries,
        'midpoints': midpoints,
        'quartiles': quartiles,
        'deciles': deciles,
        'percentiles': percentiles
    }


@csrf_exempt
def interval_detailed_stats(request):
    if request.method == 'POST':
        csv_data = request.POST.get('csv_data', '')
        if csv_data:
            try:
                stats = process_interval_data_detailed(csv_data)
                return render(request, 'statistics/interval_detailed_stats.html', {'stats': stats})
            except ValueError as e:
                return render(request, 'statistics/interval_detailed_stats.html', {'error': str(e)})
        else:
            return render(request, 'statistics/interval_detailed_stats.html', {'error': 'No CSV data provided.'})
    return render(request, 'statistics/interval_detailed_stats.html')

def initialize_game(request):
    if "board" not in request.session:
        request.session["board"] = np.array([["" for _ in range(3)] for _ in range(3)]).tolist()
        request.session["current_player"] = "X"
        request.session.modified = True
    return redirect('game_board')

def game_board(request):
    board = np.array(request.session.get("board"))
    current_player = request.session.get("current_player")
    winner = check_winner(board)

    if winner is not None:
        context = {"message": f"Player {winner} wins!", "board": board, "winner": winner}
        return render(request, 'game/game_board.html', context)
    elif check_draw(board):
        context = {"message": "Draw!", "board": board}
        return render(request, 'game/game_board.html', context)

    if request.method == "POST":
        row = int(request.POST.get("row"))
        col = int(request.POST.get("col"))
        
        if board[row, col] == "":
            board[row, col] = current_player
            request.session["board"] = board.tolist()
            request.session["current_player"] = "O" if current_player == "X" else "X"
            request.session.modified = True
        
        return redirect('game_board')
    
    context = {"board": board}
    return render(request, 'game/game_board.html', context)

def reset_game(request):
    if "board" in request.session:
        del request.session["board"]
    if "current_player" in request.session:
        del request.session["current_player"]
    return redirect('initialize_game')
'''

def quiz_list(request):
    quizzes = Quiz.objects.all()
    return render(request, 'quiz/quiz_list.html', {'quizzes': quizzes})

def quiz_detail(request, quiz_id):
    quiz = get_object_or_404(Quiz, pk=quiz_id)
    questions = quiz.question_set.all()
    return render(request, 'quiz/quiz_detail.html', {'quiz': quiz, 'questions': questions})

def submit_quiz(request, quiz_id):
    quiz = get_object_or_404(Quiz, pk=quiz_id)
    questions = quiz.question_set.all()
    score = 0
    for question in questions:
        selected_choice = request.POST.get(f'question_{question.id}')
        if selected_choice:
            choice = Choice.objects.get(pk=selected_choice)
            if choice.is_correct:
                score += 1
    return render(request, 'quiz/quiz_result.html', {'quiz': quiz, 'score': score})
'''
from django.shortcuts import render, redirect, get_object_or_404
from .models import Quiz, Question, Choice, QuizResult
from .forms import UserInfoForm

def quiz_list(request):
    quizzes = Quiz.objects.all()
    return render(request, 'quiz/quiz_list.html', {'quizzes': quizzes})
from django.http import HttpResponseBadRequest
def quiz_detail(request, quiz_id):
    quiz = get_object_or_404(Quiz, pk=quiz_id)
    questions = quiz.question_set.all()
    form = UserInfoForm()
    return render(request, 'quiz/quiz_detail.html', {'quiz': quiz, 'questions': questions, 'form': form})
def submit_quiz(request, quiz_id):
    quiz = get_object_or_404(Quiz, pk=quiz_id)
    questions = quiz.question_set.all()
    score = 0

    # Cek apakah semua pertanyaan telah diisi
    unanswered_questions = []
    for question in questions:
        selected_choice = request.POST.get(f'question_{question.id}')
        if not selected_choice:
            unanswered_questions.append(question)

    if unanswered_questions:
        return HttpResponseBadRequest("Semua pertanyaan harus dijawab.")

    # Hitung skor
    for question in questions:
        selected_choice = request.POST.get(f'question_{question.id}')
        if selected_choice:
            choice = Choice.objects.get(pk=selected_choice)
            if choice.is_correct:
                score += 1

    form = UserInfoForm(request.POST)
    if form.is_valid():
        user_info = form.save(commit=False)
        user_info.quiz = quiz
        user_info.score = score
        user_info.save()
    
    return render(request, 'quiz/quiz_result.html', {'quiz': quiz, 'score': score})

# class based views for posts
class postslist(generic.ListView):
	queryset = posts.objects.filter(status=1).order_by('-created_on')
	template_name = 'home.html'
	paginate_by = 4

# class based view for each post
class postdetail(generic.DetailView):
	model = posts
	template_name = "page.html"

class StaticFilesView(View):
    def get(self, request, filename):
        static_dir = settings.BASE_DIR / 'bima/static/'
        file_path = os.path.join(static_dir, filename)
        
        # Check if the requested file exists
        if os.path.exists(file_path) and os.path.isfile(file_path):
            # Open the file in binary mode
            with open(file_path, 'rb') as f:
                # Read the file content
                file_content = f.read()
            
            # Determine the content type based on the file extension
            content_type = 'application/octet-stream'
            if filename.endswith('.png'):
                content_type = 'image/png'
            elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
                content_type = 'image/jpeg'
            elif filename.endswith('.svg'):
                content_type = 'image/svg'
            elif filename.endswith('.css'):
                content_type = 'text/css'
            elif filename.endswith('.js'):
                content_type = 'text/javascript'
            # Return the file content with appropriate content type
            return HttpResponse(file_content, content_type=content_type)
        
        # If the file is not found, return a 404 response
        return HttpResponse(status=404)

            
def spinner(request):
    return render(request, 'spinner.html')
@csrf_exempt
def spin_wheel(request):
    labels = []
    pie_colors = ["#8b35bc", "#b163da", "#d88a40", "#c66e16", "#a33520", "#e53935"]
    
    numbers_param = request.POST.get("numbers", '')
    if numbers_param:
        labels = [label.strip() for label in numbers_param.split(',')]
    
    response_data = {"labels": labels, "pie_colors": pie_colors}
    
# Render the template with the data
    return render(request, 'spinner.html', {'labels': labels, 'pie_colors': pie_colors})

def images(request, img_type):
    image_path = os.path.join(settings.BASE_DIR, 'bima/images/', f'{img_type}.png')
    
    # Check if the image file exists
    if os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            return HttpResponse(f.read(), content_type='image/png')
    else:
        return HttpResponse(status=404)
def chess_piece_image(request, piece_type):
    # Construct the path to the chess piece image
    image_path = os.path.join(settings.BASE_DIR, 'bima/img/chesspieces/wikipedia', f'{piece_type}.png')
    
    # Check if the image file exists
    if os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            return HttpResponse(f.read(), content_type='image/png')
    else:
        return HttpResponse(status=404)

def chess(request):
    return render(request, 'chess.html')

# Fungsi untuk mengenkripsi teks menggunakan Uuencoding dengan hasil dalam huruf kecil
def encrypt_uuencode(text):
    encoded_bytes = binascii.b2a_uu(text.encode('utf-8'))
    encoded_text = encoded_bytes.decode('utf-8')
    return encoded_text.lower()  # Mengubah hasil enkripsi menjadi huruf kecil

# Fungsi untuk mendekripsi teks yang telah dienkripsi menggunakan Uuencoding
def decrypt_uuencode(encoded_text):
    try:
        decoded_bytes = binascii.a2b_uu(encoded_text.encode('utf-8'))
        decoded_text = decoded_bytes.decode('utf-8')
        return decoded_text
    except binascii.Error:
        return None

hruf_latin = {
    "a" : "._",
    "b" : "_...",
    "c" : "_._.",
    "d" : "_..",
    "e" : ".",
    "f" : ".._.",
    "=" : "....____",
    "." : "._____.",
    "," : ".___.___.",
    "+" : ".___..___.",
    "-" : "________",
    "*" : "........",
    "/" : "_____.....",
    "\\" : "_.__.__._.____.",
    "&" : "._._._._",
    "#" : "_._._._._.",
    "~" : ".._.___.",
    "%": "______...", "$": ".____", "(": "____.__._._._._", ")": "_....__._..",
    "g" : "__.",
    "h" : "....",
    "i" : "..",
    "j" : ".___",
    "k" : "_._",
    "l" : "._..",
    "m" : "__",
    "n" : "_.",
    "o" : "___",
    "p" : ".__.",
    "q" : "__._",
    "r" : "._.",
    "s" : "...",
    "t" : "_",
    "u" : "..._",
    "w" : ".__",
    "x" : "_.._",
    "y" : "_.__",
    "z" : "__..",
    "1" : ".._",
    "v" : ".____",
    "2" : "..___",
    "3" : "...__",
    "4" : "...._",
    "5" : ".....",
    "6" : "_....",
    "7" : "__...",
    "8" : "___..",
    "9" : "____.",
    "0" : "_____"
}
def encrypt_base64(text):
    encrypted_bytes = base64.b64encode(text.encode('utf-8'))
    encrypted_text = encrypted_bytes.decode('utf-8')
    return encrypted_text
def morse_translate(text):
    morse_code = []
    for char in text.lower():
        if char in hruf_latin:
            morse_code.append(hruf_latin[char])
        else:
            morse_code.append(" ")  # Menggunakan spasi jika karakter tidak ada dalam dictionary
    return " ".join(morse_code)


def reverse_morse_translate(morse_code):
    reverse_dict = {value: key for key, value in hruf_latin.items()}
    morse_chars = morse_code.strip().split(" ")
    text = "".join([reverse_dict[char] if char in reverse_dict else " " for char in morse_chars])
    return text

def decrypt_base64(encrypted_text):
    try:
        decrypted_bytes = base64.b64decode(encrypted_text.encode('utf-8'))
        decrypted_text = decrypted_bytes.decode('utf-8')
        return decrypted_text
    except UnicodeDecodeError:
        return None
        
        
def morse64(request):
    expression = request.POST.get('expression', '')
    result = ''
    if request.method == 'POST':
        if 'morse' in request.POST:
            reversed_str = expression[::-1].lower()

            char = {'a': 'z', 'b': 'y', 'c': 'x', 'd': 'w', 'e': 'v', 'f': 'u', 'g': 't', 'h': 's', 'i': 'r', 'j': 'q', 'k': 'p', 'l': 'o', 'm': 'n', 'n': 'm', 'o': 'l', 'p': 'k', 'q': 'j', 'r': 'i', 's': 'h', 't': 'g', 'u': 'f', 'v': 'e', 'w': 'd', 'x': 'c', 'y': 'b', 'z': 'a'}
            reversed_with_replacement = ''.join(char[c] if c in char else c for c in reversed_str)

            enc_str = morse_translate(reversed_with_replacement)

            result = encrypt_base64(enc_str)
        elif 'latin' in request.POST:
        
            a = expression.strip()
            reversed_str = reverse_morse_translate(decrypt_base64(a))
            latin_str = reversed_str[::-1].lower()

            char = {'z': 'a', 'y': 'b', 'x': 'c', 'w': 'd', 'v': 'e', 'u': 'f', 't': 'g', 's': 'h', 'r': 'i', 'q': 'j', 'p': 'k', 'o': 'l', 'n': 'm', 'm': 'n', 'l': 'o', 'k': 'p', 'j': 'q', 'i': 'r', 'h': 's', 'g': 't', 'f': 'u', 'e': 'v', 'd': 'w', 'c': 'x', 'b': 'y', 'a': 'z'}

            dec_str = ''.join(char[c] if c in char else c for c in latin_str)
            
            result = dec_str

    return render(request, 'morse64.html', {'result' : result})
    

        
def add_implicit_multiplication(expression):
    # Tambahkan operator '*' eksplisit sebelum variabel tanpa operator di antara mereka
    # Misalnya, ubah '3x+5y' menjadi '3*x+5*y'
    operators = {'x', 'y'}
    new_expr = ""
    for i, char in enumerate(expression):
        new_expr += char
        if char.isalpha() and i < len(expression) - 1 and expression[i+1] not in operators:
            new_expr += '*'
    return new_expr

def solve_linear_system(request):
    result_expr = None
    try:
        x = 'x'
        if request.method == 'POST':
            f_equation_str = request.POST.get('f_equation')
            g_equation_str = request.POST.get('g_equation')
            operator = request.POST.get('operator')
            f_x = sp.Function('f')(x)
            g_x = sp.Function('g')(x)

            x = sp.Symbol('x')
            f_x_expr = sp.sympify(f_equation_str.replace(' ', ''))
            g_x_expr = sp.sympify(g_equation_str.replace(' ', ''))

            result_expr = None
            if operator == '+':
                result_expr = f_x_expr + g_x_expr
            elif operator == '-':
                result_expr = f_x_expr - g_x_expr
            elif operator == '*':
                result_expr = f_x_expr * g_x_expr
            elif operator == '/':
                result_expr, _ = sp.div(f_x_expr, g_x_expr)
            elif operator == '%':
                quotient, remainder = sp.div(f_x_expr, g_x_expr)
                result_expr = remainder
            else:
                return render(request, 'solve_linear_system.html', {'error_message': 'Invalid operator'})

            result_expr = sp.expand(result_expr)
            
        return render(request, 'solve_linear_system.html', {'result': result_expr})

    except Exception as e:
        pass

    return render(request, 'solve_linear_system.html', {'result': result_expr})



def polino(request):
    try:
        result = None
        if request.method == 'POST':
            polynomial1 = request.POST.get('polynomial1')
            polynomial2 = request.POST.get('polynomial2')
            operation = request.POST.get('operation')

            x = sp.symbols('x')
            poly1 = sp.Poly(polynomial1, x).as_expr()  # Convert to SymPy expression
            poly2 = sp.Poly(polynomial2, x).as_expr()  # Convert to SymPy expression

            if operation == 'addition':
                result = sp.expand(poly1 + poly2)
            elif operation == 'multiplication':
                result = sp.expand(poly1 * poly2)
            elif operation == 'division':
                quotient, remainder = sp.div(poly1, poly2)
                result = {'quotient': quotient, 'remainder': remainder}
            elif operation == 'roots':
                result = sp.solve(poly1, x)

        # Convert result to LaTeX format for displaying in template
            if result:
                result = sp.latex(result)

        # Plot the polynomial if operation is 'roots'
            if operation == 'roots':
                x_vals = np.linspace(-10, 10, 400)
                y_vals = np.array([poly1.subs(x, val) for val in x_vals], dtype=float)
                plt.plot(x_vals, y_vals)
                plt.xlabel('x')
                plt.ylabel('f(x)')
                plt.title('Plot of the polynomial')
                plt.grid(True)
                plt.savefig('/bima/static/polynomial_plot.png')
                plt.close()
    except Exception as e:
        pass
    return render(request, 'polino.html', {'result': result})

def algebra(request):
    try:
        if request.method == 'POST':
            matrix_a = request.POST.get('matrix_a')
            matrix_b = request.POST.get('matrix_b')
            operation = request.POST.get('operation')

        # Convert string matrices to numpy arrays
            array_a = np.array([list(map(float, row.split(','))) for row in matrix_a.split('\n')])
            array_b = np.array([list(map(float, row.split(','))) for row in matrix_b.split('\n')])

            result = None
            if operation == 'addition':
                result = np.add(array_a, array_b)
            elif operation == 'subtraction':
                result = np.subtract(array_a, array_b)
            elif operation == 'scalar':
                result = np.multiply(array_a, array_b)
            elif operation == 'multi':
                result = np.dot(array_a, array_b)
            elif operation == 'division':
                result = np.divide(array_a, array_b)
            elif operation == 'modulo':
                result = np.mod(array_a, array_b)
            elif operation == 'floor':
                result = np.floor_divide(array_a, array_b)
            elif operation == 'inverse':
                try:
                    result = inv(array_a)
                except np.linalg.LinAlgError:
                    result = "Matrix A is singular, inverse does not exist."

            return render(request, 'algebra.html', {
                'matrix_a': matrix_a,
                'matrix_b': matrix_b,
                'scalar': result if operation == 'scalar' else None,
                'operation': operation,
                'result': result,
                'show_results': True
            })
    except Exception as e:
        pass
    return render(request, 'algebra.html')


"""
def cartoonize (image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurImage = cv2.medianBlur(image, 1)

  edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

  color = cv2.bilateralFilter(image, 9, 200, 200)

  cartoon = cv2.bitwise_and(color, color, mask = edges)

  return cartoon
  
def monitorbelakang(request):
    return render(request, 'monitorbelakang.html')
"""
def quran(request):
    ayah = ''
    translate = ''
    if request.method == "POST":
        try:
            quran = Quran()
            idsurah = int(request.POST.get('idsurah', ''))
            jml_ayat = quran.JumlahAyat(idsurah)
            nama_surat = quran.Surat(idsurah)
            ayah_list = []
            translate_list = []
            for ayat_id in range(1, jml_ayat + 1):
                ayah_temp = quran.Ayat(idsurah, ayat_id)
                translate_temp = quran.Terjemahan(idsurah, ayat_id)
                # Menggunakan numerals Arab untuk nomor ayat
                ayah_list.append(f"{str(ayat_id).replace('1', '١').replace('2', '٢').replace('3', '٣').replace('4', '٤').replace('5', '٥').replace('6', '٦').replace('7', '٧').replace('8', '٨').replace('9', '٩').replace('0', '٠')}. {ayah_temp}")
                translate_list.append(translate_temp)

            # Menggabungkan setiap ayat dengan terjemahan menjadi satu string dengan karakter baris baru
            ayah = '\n'.join(ayah_list)
            translate = '\n'.join(translate_list)

        except Exception as e:
            ayah = ''
            translate = ''
    return render(request, 'quran.html', {'ayah': ayah, 'translate': translate})
'''
def apply_filter(image):
    out_celeba = image.filter(ImageFilter.CONTOUR)
    return out_celeba

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def monitor(request):
    return render(request, 'monitor.html')

def webcam_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()), content_type='multipart/x-mixed-replace; boundary=frame')

# Fungsi untuk menerapkan efek pointilisme
# Fungsi untuk menerapkan efek pointilisme
def apply_pointillism(image):
    # Ubah gambar ke dalam format grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Kurangi resolusi gambar untuk membuat titik-titik yang kasar
    small_image = cv2.resize(gray, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)
    
    # Kembalikan ukuran gambar ke ukuran aslinya
    small_image = cv2.resize(small_image, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    
    return small_image

# Fungsi untuk menerapkan gaya abstrak
def apply_abstract_style(image):
    # Lakukan modifikasi gambar dengan teknik tertentu untuk menciptakan efek abstrak
    # Misalnya, ubah ukuran gambar, aplikasikan filter, atau modifikasi warna
    
    # Contoh sederhana: ubah gambar menjadi citra negatif
    abstract_image = cv2.bitwise_not(image)
    
    return abstract_image

@csrf_exempt
def process_image(request):
    if request.method == 'POST':
        # Receive image data from POST request
        image_data_base64 = request.POST.get('image_data', '')

        # Decode base64 data into bytes
        image_data = BytesIO(base64.b64decode(image_data_base64.split(',')[1]))

        # Open image using PIL
        image = Image.open(image_data)

        # Get the selected menu from the request
        selected_menu = request.POST.get('filter', '')

        # Apply the selected filter/menu
        if selected_menu == 'cartoonize':
            img_output = cartoonize(np.array(image))
            output = BytesIO()
            img_output_pil = Image.fromarray(img_output)  # Convert NumPy array to PIL image
            img_output_pil.save(output, format='JPEG')    # Save the PIL image to BytesIO buffer

            # Get the value from the buffer
            filtered_image_data = output.getvalue()

            # Convert the image to base64 format
            filtered_image_base64 = base64.b64encode(filtered_image_data).decode('utf-8')

            # Respond with the filtered image data

        elif selected_menu == 'apply_filter':
            img_output = apply_filter(image)
            output = BytesIO()  # Convert NumPy array to PIL image
            img_output.save(output, format='JPEG')    # Save the PIL image to BytesIO buffer

            # Get the value from the buffer
            filtered_image_data = output.getvalue()

            # Convert the image to base64 format
            filtered_image_base64 = base64.b64encode(filtered_image_data).decode('utf-8')

        elif selected_menu == 'abstrak':
            img_output = apply_abstract_style(np.array(image))
            output = BytesIO()
            img_output_pil = Image.fromarray(img_output)  # Convert NumPy array to PIL image
            img_output_pil.save(output, format='JPEG')    # Save the PIL image to BytesIO buffer

            # Get the value from the buffer
            filtered_image_data = output.getvalue()

            # Convert the image to base64 format
            filtered_image_base64 = base64.b64encode(filtered_image_data).decode('utf-8')

        elif selected_menu == 'point':
            img_output = apply_pointillism(np.array(image))
            output = BytesIO()
            img_output_pil = Image.fromarray(img_output)  # Convert NumPy array to PIL image
            img_output_pil.save(output, format='JPEG')    # Save the PIL image to BytesIO buffer

            # Get the value from the buffer
            filtered_image_data = output.getvalue()

            # Convert the image to base64 format
            filtered_image_base64 = base64.b64encode(filtered_image_data).decode('utf-8')

        elif selected_menu == 'blur':
            img_output = image.filter(ImageFilter.BLUR)
            output = BytesIO()  # Convert NumPy array to PIL image
            img_output.save(output, format='JPEG')    # Save the PIL image to BytesIO buffer

            # Get the value from the buffer
            filtered_image_data = output.getvalue()

            # Convert the image to base64 format
            filtered_image_base64 = base64.b64encode(filtered_image_data).decode('utf-8')
        elif selected_menu == 'emboss':
            img_output = image.filter(ImageFilter.EMBOSS)
            output = BytesIO()  # Convert NumPy array to PIL image
            img_output.save(output, format='JPEG')    # Save the PIL image to BytesIO buffer

            # Get the value from the buffer
            filtered_image_data = output.getvalue()

            # Convert the image to base64 format
            filtered_image_base64 = base64.b64encode(filtered_image_data).decode('utf-8')
        elif selected_menu == 'edge':
            img_output = image.filter(ImageFilter.EDGE_ENHANCE)
            output = BytesIO()  # Convert NumPy array to PIL image
            img_output.save(output, format='JPEG')    # Save the PIL image to BytesIO buffer

            # Get the value from the buffer
            filtered_image_data = output.getvalue()

            # Convert the image to base64 format
            filtered_image_base64 = base64.b64encode(filtered_image_data).decode('utf-8')

        elif selected_menu == 'model':
            # Konversi gambar menjadi tensor
            img_pt_input = img_common_util.convert_image_to_tensor(image)

            # Inisialisasi model Photo2PixelModel
            model = Photo2PixelModel()
            model.eval()
            kernel_size = 10
            pixel_size = 6
            edge_thresh = 10

            # Lakukan pemrosesan gambar dengan model
            with torch.no_grad():
                img_pt_output = model(
                    img_pt_input,
                    param_kernel_size=kernel_size,
                    param_pixel_size=pixel_size,
                    param_edge_thresh=edge_thresh
                )

            # Konversi tensor gambar menjadi gambar PIL
            img_output = img_common_util.convert_tensor_to_image(img_pt_output)

        # Simpan gambar yang telah difilter ke dalam buffer BytesIO
            output = BytesIO()
            
            img_output.save(output, format='JPEG')
            # Ambil nilai dari buffer BytesIO
            filtered_image_data = output.getvalue()

            # Konversi gambar ke dalam format base64
            filtered_image_base64 = base64.b64encode(filtered_image_data).decode('utf-8')



        else:
            # Handle invalid menu selection
            return JsonResponse({'error': 'Invalid menu selection'}, status=400)

        # Simpan gambar yang telah difilter ke dalam buffer BytesIO
        return JsonResponse({'image_data': filtered_image_base64})
    else:
        # Return method not allowed if the request method is not POST
        return JsonResponse({'error': 'Method not allowed'}, status=405)
'''
def satuan(request):
    output_deret = ''
    if request.method == 'POST':
        number = float(request.POST.get('number'))
        deret = request.POST.get('deret', '')
        
        if deret == 'k-h':
            output_deret = number * (10**1)
        elif deret == 'k-da':
            output_deret = number * (10**2)
        elif deret == 'k-m':
            output_deret = number * (10**3)   
        elif deret == 'k-d':
            output_deret = number * (10**4) 
        elif deret == 'k-c':
            output_deret = number * (10**5) 
        elif deret == 'k-mm':
            output_deret = number * (10**6)
        elif deret == 'h-k':
            output_deret = number * (10**-1)
        elif deret == 'h-da':
            output_deret = number * (10**1)
        elif deret == 'h-m':
            output_deret = number * (10**2)   
        elif deret == 'h-d':
            output_deret = number * (10**3) 
        elif deret == 'h-c':
            output_deret = number * (10**4) 
        elif deret == 'h-mm':
            output_deret = number * (10**5)
        elif deret == 'da-k':
            output_deret = number * (10**-2)
        elif deret == 'da-h':
            output_deret = number * (10**-1)
        elif deret == 'da-m':
            output_deret = number * (10**1)   
        elif deret == 'da-d':
            output_deret = number * (10**2) 
        elif deret == 'da-c':
            output_deret = number * (10**3) 
        elif deret == 'da-mm':
            output_deret = number * (10**4)
        elif deret == 'm-k':
            output_deret = number * (10**-3)
        elif deret == 'm-h':
            output_deret = number * (10**-2)
        elif deret == 'm-da':
            output_deret = number * (10**-1)   
        elif deret == 'm-d':
            output_deret = number * (10**1) 
        elif deret == 'm-c':
            output_deret = number * (10**2) 
        elif deret == 'm-mm':
            output_deret = number * (10**3)
        elif deret == 'd-k':
            output_deret = number * (10**-4)
        elif deret == 'd-h':
            output_deret = number * (10**-3)
        elif deret == 'd-da':
            output_deret = number * (10**-2)   
        elif deret == 'd-m':
            output_deret = number * (10**-1) 
        elif deret == 'd-c':
            output_deret = number * (10**1) 
        elif deret == 'd-mm':
            output_deret = number * (10**2)
        
        elif deret == 'c-k':
            output_deret = number * (10**-5)
        elif deret == 'c-h':
            output_deret = number * (10**-4)
        elif deret == 'c-da':
            output_deret = number * (10**-3)   
        elif deret == 'c-m':
            output_deret = number * (10**-2) 
        elif deret == 'c-d':
            output_deret = number * (10**-1) 
        elif deret == 'c-mm':
            output_deret = number * (10**1)
        elif deret == 'mm-k':
            output_deret = number * (10**-6)
        elif deret == 'mm-h':
            output_deret = number * (10**-5)
        elif deret == 'mm-da':
            output_deret = number * (10**-4)   
        elif deret == 'mm-m':
            output_deret = number * (10**-3) 
        elif deret == 'mm-d':
            output_deret = number * (10**-2) 
        elif deret == 'mm-c':
            output_deret = number * (10**-1)
    derets = ['k-h', 'k-da', 'k-m', 'k-d', 'k-c', 'k-mm','h-k', 'h-da', 'h-m', 'h-d', 'h-c', 'h-mm', 'da-k', 'da-h', 'da-m', 'da-d', 'da-c', 'da-mm','m-k', 'm-h', 'm-da', 'm-d', 'm-c', 'm-mm','d-k', 'd-h', 'd-da', 'd-m', 'd-c', 'd-mm', 'c-k', 'c-h', 'c-da', 'c-m', 'c-d', 'c-mm', 'mm-k', 'mm-h', 'mm-da', 'mm-m', 'mm-d', 'mm-c']
    
    return render(request, 'satuan.html', {'derets': derets, 'output_deret': output_deret})

def personal(request):

    return render(request, 'personal.html')
def project(request):
    images_dir = os.path.join(settings.STATIC_URL, 'image')  # Sesuaikan dengan nama direktori yang benar
    dirs = os.path.join(settings.BASE_DIR, 'bima', 'static', 'image')  # Path lengkap ke direktori gambar
    image_files = os.listdir(dirs)

    image_paths = []
    for file in image_files:
        # Buat path lengkap ke file gambar
        full_path = os.path.join(dirs, file)
        
        # Buka gambar dengan Pillow
        image = Image.open(full_path)
        
        # Lakukan scaling gambar ke ukuran yang diinginkan
        scaled_image = image.resize((300, 200))  # Ubah ukuran gambar sesuai kebutuhan
        
        # Simpan gambar yang telah di-scaling ke direktori tempat Anda menyimpan gambar yang di-scaling
        scaled_image.save(os.path.join(dirs, f'{file}'))

        # Tambahkan path gambar yang telah di-scaling ke list image_paths
        image_paths.append(os.path.join(images_dir, f'{file}'))

    context = {
        'image_paths': image_paths
    }
    
    return render(request, 'project.html', context)

def profile(request):

    return render(request, 'profile.html')

def spinner(request):
    return render(request, 'spinner.html')
def generate_image(text, image_width=1000, image_height=1000):
    # Create a blank image with white background
    image = Image.new("RGB", (image_width, image_height), "white")
    draw = ImageDraw.Draw(image)

    # Load a font
    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NotoSansJavanese.ttf")
    font = ImageFont.truetype(font_path, size=20)

    # Wrap the text to fit the image width
    wrapped_text = textwrap.fill(text, width=40)  # Adjust the width as needed

    # Draw text on a temporary image to get its bounding box
    temp_image = Image.new("RGB", (1, 1), "white")
    temp_draw = ImageDraw.Draw(temp_image)
    text_bbox = temp_draw.textbbox((0, 0), wrapped_text, font=font)

    # Calculate text position
    text_x = (image_width - (text_bbox[2] - text_bbox[0])) // 2
    text_y = (image_height - (text_bbox[3] - text_bbox[1])) // 2

    # Draw text on the main image
    draw.text((text_x, text_y), wrapped_text, fill="black", font=font)

    return image
    
def is_prima(x):
    for i in range(2, x):
        if x % i == 0:
            return False
    return True

def prima(request):
    result = "1 100"
    if request.method == "GET":
        try:
            if 'prima' in request.GET:
                
                input_str = request.GET.get("input")
                index1, index2 = map(int, input_str.split())

                result = []
                for x in range(index1, index2 + 1):
                    if is_prima(x):
                        result.append(x)
            elif 'kpk' in request.GET:
                input_str = request.GET.get("input")
                index1, index2 = map(int, input_str.split())
                a = [index1, index2]
                lcm = 1
                for i in a:
                    lcm = lcm*i//gcd(lcm, i)
                result = (factorint(lcm), lcm)
            elif 'fpb' in request.GET:
                try:
                    input_str = request.GET.get("input")
                    numbers = list(map(int, input_str.split()))

                    # Find the GCD using the reduce and gcd functions
                    result = (factorint(reduce(gcd, numbers)), reduce(gcd, numbers))
                except Exception as e:
                    result = str(e)
        except Exception as e:
            result = e
    return render(request, 'prima.html', {'result': result})

def generate_permutations(iterable, r):
    n = len(iterable)
    if r > n:
        raise ValueError("r cannot be greater than n")
    return permutations(iterable, r)

def generate_combinations(iterable, r):
    n = len(iterable)
    if r > n:
        raise ValueError("r cannot be greater than n")
    return combinations(iterable, r)

def permutation_count(n, r):
    if r == 0:
        return 1
    return factorial(n) // factorial(n - r)

def combination_count(n, r):
    return factorial(n) // (factorial(r) * factorial(n - r))

def probli(request):
    result = "n r"
    if request.method == "POST":
        try:
            input_str = request.POST.get("input")
            n, r = map(int, input_str.split())
            if 'permu' in request.POST:
                result = permutation_count(n, r)
            elif 'combi' in request.POST:
                result = combination_count(n, r)
        except Exception as e:
            result = str(e)
    return render(request, 'probli.html', {'result': result})
def translator(request):
    to_translate = request.POST.get('expression', '')
    target_lang = 'id'

    # Check if the input text is not empty
    if not to_translate:
        return render(request, 'translator.html', {'result': 'Input text is empty. Please enter text to translate.'})

    try:
        # Detect the language of the input text
        input_lang = detect(to_translate)
        lang = input_lang
        if input_lang in ['so', 'tl']:
            result = to_translate
        # Set the target language based on the detected language
        if input_lang == 'id':
            target_lang = 'en'
        elif input_lang == 'en':
            target_lang = 'id'

        # Translate the input text to the target language
        result = GoogleTranslator(source=input_lang, target=target_lang).translate(to_translate)

        return render(request, 'translator.html', {'lang': lang, 'result': result})

    except LangDetectException as e:
        return render(request, 'translator.html', {'result': f'Error detecting language: {str(e)}'})
def morse(request):
    result = ""
    expression = request.POST.get('expression', '')
    if 'morse' in request.POST:
        result = morse_translate(expression)
    elif 'latin' in request.POST:
        result = reverse_morse_translate(expression)

    return render(request, 'morse.html', {'result' : result})
def convert_image(request):

    if request.method == 'POST':

        try:
            
            dicts = Jvdict()
            # Ambil gambar dari formulir

            aksara_text = request.POST.get('aksara_text', '')
            translated_text = ' '.join(transliterate(aksara_text, dicts.return_javtolatin()))
            # Path untuk menyimpan gambar sementara
            image = generate_image(translated_text)
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()


            # Menampilkan hasil pada template
            return render(request, 'aksara/convert_image.html', {'image': image_base64, 'translated_text': translated_text})

        except Exception as e:
            # Tangani kesalahan
            return render(request, 'aksara/error.html', {'error_message': str(e)})
            
    return render(request, 'aksara/convert_image.html')

            # Open the uploaded image using PIL
            #image = Image.open(uploaded_image)

            # Perform OCR on the image to extract Javanese text
            #extracted_text = pytesseract.image_to_string(image, lang='jav')
def aksara_converter(request):
    if request.method == 'POST':
        text_to_convert = request.POST.get('text_to_convert', '')
        converted_text = dotransliterate(text_to_convert)
        image = generate_image(converted_text)

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()


        return render(request, 'aksara/convert.html', {'image': image_base64, 'converted_text': converted_text})

    return render(request, 'aksara/convert.html')
    
@csrf_exempt
def translated_search(request):
    if request.method == 'POST':
        try:
            # Get user input for original coordinates and translation values
            x_origin = float(request.POST.get('x_origin', 0.0))
            y_origin = float(request.POST.get('y_origin', 0.0))
            translation_x = float(request.POST.get('translation_x', 0.0))
            translation_y = float(request.POST.get('translation_y', 0.0))

            # Calculate the translated coordinates
            x_translated = x_origin + translation_x
            y_translated = y_origin + translation_y

            # Pass the values to the template
            context = {
                'x_origin': x_origin,
                'y_origin': y_origin,
                'translation_x': translation_x,
                'translation_y': translation_y,
                'x_translated': x_translated,
                'y_translated': y_translated,
            }

        except ValueError:
            # Handle the case where user input is not valid (not a float)
            context = {
                'error_message': 'Invalid input. Please enter numeric values.'
            }

        # Render the HTML template with the context
        return render(request, 'translate_end.html', context)

    return render(request, 'translate_end.html', {})
    
@csrf_exempt
def search_original_coordinates(request):
    if request.method == 'POST':
        try:
            # Get translated point coordinates from the form
            x_translated = float(request.POST.get('tx', 0.0))
            y_translated = float(request.POST.get('ty', 0.0))

            # Get translation values from the form
            tx = float(request.POST.get('tx_value', 0.0))
            ty = float(request.POST.get('ty_value', 0.0))

            # Compute the original coordinates
            x_original = x_translated - tx
            y_original = y_translated - ty

            # Pass the values to the template
            context = {
                'x_translated': x_translated,
                'y_translated': y_translated,
                'tx': tx,
                'ty': ty,
                'x_original': x_original,
                'y_original': y_original,
            }

        except ValueError:
            # Handle the case where user input is not valid (not a float)
            context = {
                'error_message': 'Invalid input. Please enter numeric values.'
            }

        # Render the HTML template with the context
        return render(request, 'translate_ori.html', context)

    return render(request, 'translate_ori.html', {})
@csrf_exempt
def translate(request):
    if request.method == 'POST':
        # Get original point coordinates from the form
        x = float(request.POST.get('x', 0.0))
        y = float(request.POST.get('y', 0.0))

        # Get translated point coordinates from the form
        x_translated = float(request.POST.get('tx', 0.0))
        y_translated = float(request.POST.get('ty', 0.0))

        # Compute the translation values
        translation_vector = np.array([x_translated - x, y_translated - y])

        # Access the individual translation values
        tx = translation_vector[0]
        ty = translation_vector[1]

        # Define a point in 2D space (x, y)
        point = np.array([[x, y, 1.0]])

        # Define the translation matrix
        MTranslation = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])

        # Apply the translation to the point
        translated_point = np.dot(MTranslation, point.T).T

        # Pass the values to the template
        context = {
            'x': x,
            'y': y,
            'x_translated': x_translated,
            'y_translated': y_translated,
            'tx': tx,
            'ty': ty,
            'translated_point': translated_point[0, :2],
        }

        # Render the HTML template with the context
        return render(request, 'translate.html', context)

    return render(request, 'translate.html', {})
    
def base(request):
    return render(request, "base.html")
import pytube
from pytube.innertube import _default_clients


_default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID_CREATOR"]

def youtube(request):
    resolutions = ['360p', '720p', '1080p', '1440p', '2160p', 'mp3']
    title = ""
    streams = []
    streams3 = []
    error_message = ""
    resolution = []

    if request.method == 'POST':
        youtube_link = request.POST.get('youtube_link')
        resolution = request.POST.get('resolution')

        if youtube_link and resolution:
            try:
                yt = pytube.YouTube(youtube_link)
                title = yt.title

                if resolution == 'mp3':
                    streams3 = yt.streams.filter(only_audio=True)
                else:
                    streams = yt.streams.filter(res=resolution)
            except pytube.exceptions.VideoUnavailable as e:
                error_message = f"Error: Video is unavailable ({str(e)})"
            except Exception as e:
                error_message = f"Error: {str(e)}"
        else:
            error_message = "Please enter a YouTube link and select a resolution."

    context = {
        'title': title,
        'streams': streams,
        'streams3': streams3,
        'resolutions': resolutions,
        'selected_resolution': resolution,
        'error_message': error_message,
    }

    return render(request, 'youtube.html', context)

def server_time(request):
    now = datetime.now()
    response_data = {'server_time': now.strftime('%Y-%m-%d %H:%M:%S')}
    return render(request, "time.html", response_data)
def server_time(request):
    now = datetime.now()
    response_data = {'server_time': now.strftime('%Y-%m-%d %H:%M:%S')}
    return render(request, "time.html", response_data)
    
def calculate_result(expression):
    
    try:
        
        expression = expression
        result = Decimal(eval(expression))
        return str(result)
    except Exception as e:
        return "Error"
@csrf_exempt
def kalkulator(request):
    result = ""

    if request.method == 'POST':
        expression = request.POST.get('expression', '')
        result = calculate_result(expression)
        if 'log' in request.POST :
            try:
                angka = expression.split()
                a = int(angka[0])
                b = int(angka[1])
                if b == '':
                    
                    result = math.log(a)
                elif b == '10':
                    result = math.log10(a)
                else :
                    result = math.log(a, b)
            except Exception as e:
                result = "Gunakan 2 angka dipisah spasi"
        if 'to_roman' in request.POST:
            try:
                result = roman.toRoman(int(expression))
            except ValueError or IndexError:
                result = "Invalid number must be 0 - 4999"
            except Exception:
                result = "Invalid number must be 0 - 4999"
        elif 'from_roman' in request.POST:
            roman_number = expression
            try:
                result = roman.fromRoman(roman_number)
            except roman.InvalidRomanNumeralError:
                result = "Invalid Roman numeral"
        elif 'sin' in request.POST:
            sin = expression
            try:
                result = math.sin(float(sin))
            except ValueError:
                result = "error"
        elif 'cos' in request.POST:
            cos = expression
            try:
                result = math.cos(float(cos))
            except ValueError:
                result = "error"
        elif 'tan' in request.POST:
            tan = expression
            try:
                result = math.tan(float(tan))
            except ValueError:
                result = "error"
        elif 'asin' in request.POST:
            asin = expression
            try:
                result = math.asin(float(asin))
            except ValueError:
                result = "error"
        elif 'acos' in request.POST:
            acos = expression
            try:
                result = math.acos(float(acos))
            except ValueError:
                result = "error"
        elif 'atan' in request.POST:
            atan = expression
            try:
                result = math.atan(float(atan))
            except ValueError:
                result = "error"
        elif 'factorial' in request.POST:
            fact = expression
            try:
                result = math.factorial(int(fact))
            except ValueError:
                result = "Dont use comma"
        elif 'c_to_f' in request.POST:
            celcius = float(expression)
            try:
                result = ((celcius * 9/5) + 32)
            except ValueError:
                result = "error"
        elif 'f_to_c' in request.POST:
            fahrenheit = float(expression)
            try:
                result = ((fahrenheit - 32) * 5/9)
            except ValueError:
                result = "error"
        elif 'c_to_k' in request.POST:
            celcius = expression
            try:
                result = (float(celcius) + 273.15)
            except ValueError:
                result = "error"
        elif 'k_to_c' in request.POST:
            kelvin = float(expression)
            try:
                result = (kelvin - 273.15)
            except ValueError:
                result = "error"
        elif 'c_to_r' in request.POST:
            celcius = float(expression)
            try:
                result = (celcius * 4/5)
            except ValueError:
                result = "error"
        elif 'r_to_c' in request.POST:
            reamur = float(expression)
            try:
                result = (reamur * 5/4)
            except ValueError:
                result = "error"
        elif 'r_to_c' in request.POST:
            reamur = float(expression)
            try:
                result = (float(reamur) * 5/4)
            except ValueError:
                result = "error"
        elif 'f_to_r' in request.POST:
            fahrenheit = float(expression)
            try:
                result = ((fahrenheit - 32) * 4/9)
            except ValueError:
                result = "error"
        elif 'f_to_k' in request.POST:
            fahrenheit = float(expression)
            try:
                result = ((fahrenheit + 459.67)* 5/9)
            except ValueError:
                result = "error"
        elif 'r_to_f' in request.POST:
            reamur = float(expression)
            try:
                result = ((reamur * 9/4) + 32)
            except ValueError:
                result = "error"
        elif 'r_to_k' in request.POST:
            reamur = float(expression)
            try:
                result = ((reamur * 5/4) + 273.15)
            except ValueError:
                result = "error"
        elif 'k_to_r' in request.POST:
            kelvin = float(expression)
            try:
                result = ((kelvin - 273.15) * 4/5)
            except ValueError:
                result = "error"
        elif 'k_to_f' in request.POST:
            kelvin = float(expression)
            try:
                result = ((kelvin * 9/5) - 459.67)
            except ValueError:
                result = "error"
        elif 'binary' in request.POST:
            n = int(expression)
            try:
                result = format(n ,"b")
            except ValueError:
                result = "Don't use comma"
        elif 'num' in request.POST:
            n = expression
            try:
                result = int(n, 2)
            except ValueError:
                result = "Not Binary"
                
        elif 'sqrt' in request.POST:
            n = int(expression)
            try:
                result = math.sqrt(n)
            except ValueError:
                result = "Invalid Number"

            
    return render(request, 'kalkulator.html', {'result': result})
    
def index(request):
    try:
        output_result = ""
        selisih = 0.0
        output = ""
        if request.method == 'POST':
            input_code = request.POST.get('input_code', '')
            html_string = input_code
            soup = BeautifulSoup(html_string, 'html.parser')
            text = soup.get_text()

            if text.strip() != "":
                old_stdout = sys.stdout
                new_stdout = StringIO()
                sys.stdout = new_stdout

                # Execute the custom language interpreter function
                result, error = run('<stdin>', text)

                # Restore stdout to its original value
                sys.stdout = old_stdout

                # Get the output from StringIO
                output = new_stdout.getvalue()
                
                # Check if result is a list=
                if error:
                    output_result = repr(error.as_string())  # Ubah error menjadi string
                elif result:
                    if hasattr(result, 'elements') and len(result.elements) == 1:
                        output_result = repr(result.elements[0])
                        output = output_result
                    else:
                        # Check if result is a list
                        if isinstance(result, list):
                            output_result = json.dumps(result)  # Convert list to JSON string

                        else:
                            output_result = repr(result)

    except Exception as e:
        # Handle specific exceptions if needed
        print(f"An error occurred: {e}")
        output_result = f"An error occurred: {e}"

    return render(request, 'index.html', {'output_result': output_result, 'output': output})
