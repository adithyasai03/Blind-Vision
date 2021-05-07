import os
import ocr_text as oc
import time


def main():
    text_len=os.system('py ocr_text1.py')
    print(text_len)
    if text_len < 5:
        os.system('py detect.py --source 0 --weights yolov5x.pt ')
    else:
        os.system('py ocr_text1.py')


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
