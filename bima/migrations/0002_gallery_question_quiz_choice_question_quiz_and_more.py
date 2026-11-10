
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('bima', '0001_initial'),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(

            # Hanya membuat tabel Gallery di SQLite.
            database_operations=[
                migrations.CreateModel(
                    name='Gallery',
                    fields=[
                        ('id', models.BigAutoField(
                            auto_created=True,
                            primary_key=True,
                            serialize=False,
                            verbose_name='ID'
                        )),
                        ('title', models.CharField(max_length=150)),
                        ('description', models.TextField(blank=True)),
                        ('image', models.ImageField(
                            upload_to='gallery/'
                        )),
                        ('category', models.CharField(
                            choices=[
                                ('personal', 'Personal'),
                                ('achievement', 'Achievement'),
                                ('project', 'Project'),
                                ('documentation', 'Documentation'),
                            ],
                            default='personal',
                            max_length=30
                        )),
                        ('created_at', models.DateTimeField(
                            auto_now_add=True
                        )),
                    ],
                    options={
                        'ordering': ['-created_at'],
                    },
                ),
            ],

            # Memberitahu Django bahwa model lama sudah tersedia.
            state_operations=[
                migrations.CreateModel(
                    name='Quiz',
                    fields=[
                        ('id', models.BigAutoField(
                            auto_created=True,
                            primary_key=True,
                            serialize=False,
                            verbose_name='ID'
                        )),
                        ('title', models.CharField(max_length=200)),
                    ],
                ),
                migrations.CreateModel(
                    name='Question',
                    fields=[
                        ('id', models.BigAutoField(
                            auto_created=True,
                            primary_key=True,
                            serialize=False,
                            verbose_name='ID'
                        )),
                        ('text', models.CharField(max_length=255)),
                        ('quiz', models.ForeignKey(
                            on_delete=django.db.models.deletion.CASCADE,
                            to='bima.quiz'
                        )),
                    ],
                ),
                migrations.CreateModel(
                    name='Choice',
                    fields=[
                        ('id', models.BigAutoField(
                            auto_created=True,
                            primary_key=True,
                            serialize=False,
                            verbose_name='ID'
                        )),
                        ('text', models.CharField(max_length=255)),
                        ('is_correct', models.BooleanField(default=False)),
                        ('question', models.ForeignKey(
                            on_delete=django.db.models.deletion.CASCADE,
                            to='bima.question'
                        )),
                    ],
                ),
                migrations.CreateModel(
                    name='QuizResult',
                    fields=[
                        ('id', models.BigAutoField(
                            auto_created=True,
                            primary_key=True,
                            serialize=False,
                            verbose_name='ID'
                        )),
                        ('name', models.CharField(max_length=100)),
                        ('number', models.CharField(max_length=20)),
                        ('score', models.IntegerField()),
                        ('timestamp', models.DateTimeField(
                            auto_now_add=True
                        )),
                        ('quiz', models.ForeignKey(
                            on_delete=django.db.models.deletion.CASCADE,
                            to='bima.quiz'
                        )),
                    ],
                ),
                migrations.CreateModel(
                    name='Gallery',
                    fields=[
                        ('id', models.BigAutoField(
                            auto_created=True,
                            primary_key=True,
                            serialize=False,
                            verbose_name='ID'
                        )),
                        ('title', models.CharField(max_length=150)),
                        ('description', models.TextField(blank=True)),
                        ('image', models.ImageField(
                            upload_to='gallery/'
                        )),
                        ('category', models.CharField(
                            choices=[
                                ('personal', 'Personal'),
                                ('achievement', 'Achievement'),
                                ('project', 'Project'),
                                ('documentation', 'Documentation'),
                            ],
                            default='personal',
                            max_length=30
                        )),
                        ('created_at', models.DateTimeField(
                            auto_now_add=True
                        )),
                    ],
                    options={
                        'ordering': ['-created_at'],
                    },
                ),
            ],
        ),
    ]

