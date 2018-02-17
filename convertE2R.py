#This python files inputs as ellipstical coordinates and output rectangular coordinates

import cv2
import os

def convert(a,b,cx,cy):
    p1x = cx-b
    p1y = cy+a
    p2x = cx+b
    p2y = cy+a
    p3x = cx-b
    p3y = cy-a
    p4x = cx+b
    p4y = cy-a

    return p1x,p1y,p2x,p2y,p3x,p3y,p4x,p4y

def main():
    a = 1
    b = 2
    cx = 3
    cy = 4
    return convert(a,b,cx,cy)

if "__name__" == "main":
    main()
