print("start to load all models. The first time loading would be slow")
import sys
import os
import dlib
import glob
import ntpath
import numpy as np
from skimage import io
from skimage.draw import circle
import cv2
import copy

# the following lines contain some models used in this code. Please modify them according to your installation folder
predictor_path = "alignment_model/shape_predictor_68_face_landmarks.dat" #dlib shape model
detector = dlib.get_frontal_face_detector()  #dlib face detector
shape_predictor = dlib.shape_predictor(predictor_path) 
opencv_face_cascade = cv2.CascadeClassifier("face_detection_opencv_models/haarcascade_frontalface_alt2.xml")
network_structure_path_name = "C:/pkg/caffe-windows-master/models/vgg_face_caffe/VGG_FACE_16_deploy.prototxt"
vgg_model_path_name = "C:/pkg/caffe-windows-master/models/vgg_face_caffe/VGG_FACE.caffemodel";
caffe_root = "C:/pkg/caffe-windows-master/"
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_gpu()
net = caffe.Net(network_structure_path_name, vgg_model_path_name,  caffe.TEST) #caffe net model
max_cc = np.loadtxt("max_cc.txt")
threshold = 0.396
print("loading ends")

def main() :
    if len(sys.argv) != 3:
        print(
            "Argument should be 3. Need to input two images name with their path. \n"
            "For example, if you are in the folder containing this file, execute"
            "this program by running: face_recognition.py [image1_path]/image1.jpg"
            "[image2_path]/image2.jpg . Now use two default images 0000000022698720.jpg"
            " and 0000000023306859.jpg \n")
        image_path_name1 = "examples/0000000021336196.jpg"
        image_path_name2 = "examples/0000000024508983.jpg"
    else :
        image_path_name1 = sys.argv[1]
        image_path_name2 = sys.argv[2]

    feature01 = copy.deepcopy(np.squeeze(_extract_feature(image_path_name1)))
    feature02 = copy.deepcopy(np.squeeze(_extract_feature(image_path_name2)))

    #feature normalization
    #Step 1: load the max value of 4096 dim. Normalized by this vector
    n_feat01 = np.divide(feature01,max_cc)
    n_feat02 = np.divide(feature02,max_cc)
    
    #step 2: normalize by the L2 norm
    normed_feature01 = np.divide(n_feat01,np.sqrt(np.sum(n_feat01**2)))
    normed_feature02 = np.divide(n_feat02,np.sqrt(np.sum(n_feat02**2)))

    #now compute the inner product
    similarity = np.sum(normed_feature01*normed_feature02)
    print "The computed score is %f." % similarity
    if similarity >= threshold :
        print("Recognized as the same person \n")
    else :
        print("Not Recognized as the same person \n")

def _shape_to_np(shape):
    xy = []
    for i in range(68):
        xy.append((shape.part(i).x, shape.part(i).y,))
    xy = np.asarray(xy, dtype='float32')
    return xy


def _extract_feature(image_path_name) :
    img = io.imread(image_path_name)
    if len(img.shape) == 2: #if gray image, transfer to color
        img_temp = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
        img_temp[:,:,0] = img
        img_temp[:,:,1] = img
        img_temp[:,:,2] = img
        img = img_temp
    imgd = copy.deepcopy(img)
    dets = detector(img, 3) #face detection by dlib detector
    max_area_open = 0
    max_x = 0
    max_y = 0
    max_w = 0
    max_h = 0
    if len(dets) == 0 : # if dlib detector failed, use opencv detection instead
        img_opencv = cv2.imread(image_path_name)
        gray = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('img_open',img_open)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(0, 0),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )
        if len(faces) == 0 :
            print(
                "Face in the image cannot be detected by both dlib and opencv detector. \n"
                "The image quality is low. Exit.\n")
            exit()
        else : # find the largest rectangle
            for (x,y,w,h) in faces :
                if w*h > max_area_open :
                    max_area_open = w*h
                    max_x = x
                    max_y = y
                    max_w = w
                    max_h = h
            rec = dlib.rectangle(max_x, max_y, max_x + max_w, max_y + max_h)
            dets.append(rec)
            
    max_area = 0
    max_d = dets[0]
    max_k = 0
    for k, d in enumerate(dets): # find the largest rectangle
        current_area = (d.right()-d.left())*(d.bottom()-d.top())
        if current_area > max_area :
            max_area = current_area
            max_d = d
            max_k = k
    xydet = []        
    xydet.append((max_d.left(), max_d.top(), max_d.right(), max_d.bottom(),))
    xydet = np.asarray(xydet, dtype='float32')
    np.savetxt(os.path.splitext(image_path_name)[0] + "_det.txt",xydet,'%3d') # save the detection result
    # extract shape information (68 points)
    shape = shape_predictor(img, max_d)
    xy = _shape_to_np(shape)
    strtxt = "_alig.txt";
    strjpg = "_alig.jpg";
    np.savetxt(os.path.splitext(image_path_name)[0] + strtxt, xy, '%3d')
    for ind_part in range(68):
        if (shape.part(ind_part).y + 1) < imgd.shape[0] and (shape.part(ind_part).x + 1) < imgd.shape[1] :
            rr, cc = circle(shape.part(ind_part).y, shape.part(ind_part).x, 2)
            imgd[rr, cc] = [255, 0, 0]
    io.imsave(os.path.splitext(image_path_name)[0] + strjpg, imgd) # save the extracted shape to disk
    cv2.imshow('image',imgd[:,:,::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # align input image to the optimal format of VGG face net
    lefteye = np.mean(xy[36:42,:], axis=0)
    rigteye = np.mean(xy[42:48,:], axis=0)
    ori_degree = np.arctan((rigteye[1] - lefteye[1])/(rigteye[0] - lefteye[0]))
    degree = ori_degree*180/np.pi
    scale = 55/np.sqrt(((rigteye-lefteye)**2).sum())
    rows = img.shape[0]
    cols = img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
    dst = cv2.warpAffine(img[:,:,::-1],M,(cols,rows))
    res = cv2.resize(dst,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    #cv2.imshow('rotated image',res)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #io.imsave(os.path.splitext(image_path_name)[0] + "_test.jpg", res[:,:,::-1])
    # decide new eye position
    cos_v = np.cos(-ori_degree) # cos value. in np the default direction reversed
    sin_v = np.sin(-ori_degree) # sin value
    rotation_matrix = np.array([[cos_v,-sin_v], [sin_v,cos_v]])
    ori_center = np.array([cols/2, rows/2]) # the rotation center
    ori_center.shape = (2,1) # force to be column vector
    lefteye.shape = (2,1)
    rigteye.shape = (2,1)
    lefteye_r = scale*(np.dot(rotation_matrix,lefteye - ori_center) + ori_center)
    rigteye_r = scale*(np.dot(rotation_matrix,rigteye - ori_center) + ori_center)
    # padding and cropping to from a 256x256 image
    res_padded = np.lib.pad(res, ((128,128),(128,128),(0,0)), 'edge')
    lefteye_r = lefteye_r + 128
    rigteye_r = rigteye_r + 128
    start_x = (np.uint(np.ceil((lefteye_r[0] + rigteye_r[0])/2 - 127)))[0]
    start_y = (np.uint(np.ceil(lefteye_r[1] - 112)))[0]
    end_x = start_x + 256
    end_y = start_y + 256
    saveim = res_padded[start_y:end_y,start_x:end_x,...]
    #io.imsave(os.path.splitext(image_path_name)[0] + "_cropped256.jpg", saveim[:,:,::-1])
    
    #begin caffe processing. Note that saveim is already BGR format
    avg = np.array([93.5940,104.7624,129.1863]) # used for subtracting mean (numpy takes care of dimensions :)
    img_cc = saveim[16:16+224,16:16+224,:] - avg # center image, then -avg
    #img_lt = saveim[0:0+224,0:0+224,:] - avg # left top then -avg
    #img_rt = saveim[0:0+224,255-223:256,:] - avg # right top then -avg
    #img_ld = saveim[255-223:256,0:224,:] - avg # left down then -avg
    #img_rd = saveim[255-223:256,255-223:256,:] - avg # right down then -avg

    #to save time, in  the following we only use the center crop. 
    img_cc = img_cc.transpose((2,0,1))
    img_cc = img_cc[None,:] # add singleton dimension
    out_cc = net.forward_all( data = img_cc )
    featfc7_cc = net.blobs['fc7'].data
    #np.savetxt(os.path.splitext(image_path_name)[0] + "_f.txt",featfc7_cc,'%f')
    return featfc7_cc
    
if __name__ == "__main__":
    main()
