from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.ListCreatePost.as_view(), name='post_list'),
]
