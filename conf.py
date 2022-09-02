import json


def get_intrinsics(path='camera.json'):
    try:
        with open(path, 'r') as fr:
            K = json.load(fr)
        return K
    except Exception as e:
        return None

K = get_intrinsics()