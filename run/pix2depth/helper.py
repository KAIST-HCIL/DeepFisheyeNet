def arrange_data(data):
    _data = {'pix': data['fish']}
    if 'fish_depth' in data:
        _data['depth'] = data['fish_depth']

    if 'joint' in data:
        _data['joint'] = data['joint']

    return _data
