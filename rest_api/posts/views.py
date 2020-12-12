from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from .apps import PostsConfig


@api_view(['POST'])
@permission_classes((AllowAny,))
def check_text_offensiveness(request):
    data = request.data
    text = data['text']
    offensive_rate = PostsConfig.predictor.predict(text)[0][1]
    return Response({'offensive_rate': offensive_rate})