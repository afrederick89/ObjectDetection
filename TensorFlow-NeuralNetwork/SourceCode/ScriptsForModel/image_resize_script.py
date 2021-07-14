from PIL import Image
import os, sys

path = "C:\\Users\\Ryan\\Desktop\\Fullerton Fall 2020\\Tuesday.Thursday - CPSC 481 - Artificial Intelligence\\Final\\TestingImages\\"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((200,200), Image.ANTIALIAS)
            imResize.save(f + '.png', 'PNG', quality=90)

resize()