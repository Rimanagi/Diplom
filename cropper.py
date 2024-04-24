import cv2
import glob

images_path = r'/Users/rimanagi/PycharmProjects/Diplom/lense_None/'
cropped_path = r'/Users/rimanagi/PycharmProjects/Diplom/lense_None/'


def crop(sources_path, result_path):
    for image_path in glob.glob(sources_path + '*.png'):
        img = cv2.imread(image_path)
        print(img.shape)
        img = img[133:-147, 112:-112]
        # print(image_path.title().lower().rsplit('\\', 1)[1])
        fname = result_path + image_path.title().lower().rsplit('/', 1)[1].replace('.png', '.jpg')
        print(fname)
        print(img.shape)
        cv2.imwrite(fname, img)
        cv2.imwrite(fname, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


crop(images_path, cropped_path)