from django.urls import re_path

from . import views

urlpatterns = [
    re_path(r"^check_text$", views.check_text_offensiveness, name='check_text_offensiveness'),
]
