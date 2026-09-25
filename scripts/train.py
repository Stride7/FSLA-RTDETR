import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('')
    # model.load('') # loading pretrain weights
    model.train(data=r'',
                cache=False,
                imgsz=512,
                epochs=300,
                batch=4,
                workers=4,
                device='1',
                # resume='', # last.pt path
                patience=0,
                project='runs/train',
                name='_r18_3407seed',
                )