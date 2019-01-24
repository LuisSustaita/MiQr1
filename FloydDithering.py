import cv2
imagen="carrito.png"
img = cv2.imread(imagen)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.jpg', img_gray)


def find_closest_palette_color(pixel):
    return round(pixel/255)


for x in range(img_gray.shape[0]-1):
    for y in range(img_gray.shape[1]-1):
        oldpixel = img_gray[x, y]
        newpixel = find_closest_palette_color(oldpixel)
        img_gray[x, y] = newpixel
        quant_error = oldpixel-newpixel

        img_gray[x+1, y  ] = round(img_gray[x+1, y  ]+quant_error * 7/16)
        img_gray[x-1, y+1] = round(img_gray[x-1, y+1]+quant_error * 3/16)
        img_gray[x  , y+1] = round(img_gray[x  , y+1]+quant_error * 5/16)
        img_gray[x+1, y+1] = round(img_gray[x+1, y+1]+quant_error * 1/16)

for x in range(img_gray.shape[0]):
    for y in range(img_gray.shape[1]):
        if img_gray[x, y] > 23:
            img_gray[x, y] = 0
        else:
            img_gray[x, y] = 255

cv2.imwrite('floydDit.jpg', img_gray)







