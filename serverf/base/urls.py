from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [ 
    path('', views.home, name="home"),
    # path('train/', views.train, name="train"),
    path('api/get-student-tlu/', views.getMark, name='get_student_tlu'),
    path('api/post-student-data/', views.post_student_data, name='save_student_data'),
]