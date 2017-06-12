from datetime import datetime

def getCurrentTime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
