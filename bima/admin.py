
from django.contrib import admin
from .models import (
    posts,
    Quiz,
    Question,
    Choice,
    QuizResult,
    Gallery,
)


@admin.register(posts)
class PostsAdmin(admin.ModelAdmin):
    list_display = ('title', 'author', 'status', 'created_on')
    search_fields = ('title', 'content')


@admin.register(Quiz)
class QuizAdmin(admin.ModelAdmin):
    list_display = ('id', 'title')


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ('id', 'text', 'quiz')


@admin.register(Choice)
class ChoiceAdmin(admin.ModelAdmin):
    list_display = ('id', 'text', 'question', 'is_correct')


@admin.register(QuizResult)
class QuizResultAdmin(admin.ModelAdmin):
    list_display = ('name', 'number', 'quiz', 'score', 'timestamp')


@admin.register(Gallery)
class GalleryAdmin(admin.ModelAdmin):
    list_display = ('title', 'category', 'created_at')
    list_filter = ('category',)
    search_fields = ('title', 'description')

