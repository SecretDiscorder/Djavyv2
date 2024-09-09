"""
URL configuration for project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from . import views
from .views import StaticFilesView
urlpatterns = [
    path('home/', views.postslist.as_view(), name='home'),
    # route for posts
    path('post/<slug:slug>/', views.postdetail.as_view(), name='post_detail'),
    path('spldtv/', views.solve_linear_system, name='solve_linear_system'),
    path('morse64/', views.morse64, name='morse64'),
    path('chess/', views.chess, name="chess"),
    path('polino/', views.polino, name='polinomial'),
    path('algebra/', views.algebra, name='algebra'),
    #path('monitorb/', views.monitorbelakang, name='monitorbelakang'),
    #path('monitor/', views.monitor, name='monitor'),
    path('quran/', views.quran, name='quran'),
    #path('feed/', views.webcam_feed, name='webcam_feed'),
    path('tictac/', views.initialize_game, name='initialize_game'),
    path('board/', views.game_board, name='game_board'),
    path('reset/', views.reset_game, name='reset_game'),
    #path('process_image/', views.process_image, name='process_image'),
    path('diff/', views.derivative_view, name='derivative_view'),
    path('process_diff/', views.process_function, name='process_function'),

    path('quiz/', views.quiz_list, name='quiz_list'),
    path('quiz/<int:quiz_id>/', views.quiz_detail, name='quiz_detail'),
    path('quiz/<int:quiz_id>/submit/', views.submit_quiz, name='submit_quiz'),
    path('', views.personal, name=""),
    path('project/', views.project, name="project"),
    path('profile/', views.profile, name="profile"),
    path('base/', views.base, name="base"),
    path('satuan/', views.satuan, name="satuan"),
    path('blang/', views.index, name="blang"),
    path('kalkulator/', views.kalkulator, name="kalkulator"),
    path('clock/',views.server_time,name="clock"),
    path('youtube/',views.youtube,name="youtube"),
    path('morse/',views.morse,name="morse"),
    path('prima/',views.prima,name="prima"),
    path('translator/',views.translator,name="translator"),
    path('probli/', views.probli, name='probli'),
    path('spinwheel/', views.spin_wheel, name ='spinner'),
    path('chess/img/chesspieces/wikipedia/<str:piece_type>.png', views.chess_piece_image, name='chess_piece_image'),
    path('spinwheel/images/<str:img_type>.png', views.images, name='images'),
    path('statistics/', views.statis, name='home'),
    path('statistics/interval_detailed_stats/', views.interval_detailed_stats, name="interval_detailed_stats"),
    path('statistics/interval_stats/', views.interval_stats, name='interval_stats'),
    
    path('statistics/stats/', views.stats, name="stats"),
    path('inequality/', views.solve_inequality, name='solve_inequality'),

    path('aksara_converter/', views.aksara_converter, name='aksara_converter'),
    path('aksara_converter/aksara_converter_image/', views.convert_image, name='convert_image'),
    path('static/<path:filename>', StaticFilesView.as_view(), name='static-files'),
    
]
