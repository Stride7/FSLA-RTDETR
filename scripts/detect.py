import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train/_FSLA_RTDETR/weights/best.pt') # select your model.pt path
    model.predict(source='datasets/images/val',
                  conf=0.5,
                  project='runs/detect',
                  name='_target-free',
                  save=True,
                  # visualize=True # visualize model features maps
                  line_width=1, # line width of the bounding boxes
                  show_conf=False, # do not show prediction confidence
                  show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                  )