import datetime


def float_to_datetime(f):
    return datetime.fromtimestamp(f).strftime('%Y-%m-%d %H:%M:%S')