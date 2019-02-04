from django.urls import path

from . import views

urlpatterns = [
    path('', views.main_page, name='index'),
    path('test/', views.data_return),
    path('', views.index, name='index'),
]