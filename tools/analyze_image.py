import cv2
    

if __name__ == "__main__":
    # Command line argument: image path
    image_path = "/home/yinh4/DENSPOUT/data/dsWLC/trainA/01053.png"
    read = cv2.imread(image_path)
    print(read.shape)
