from Tkinter import *
from PIL import Image
from PIL import ImageTk
from PIL import ExifTags
import tkFileDialog
import tkSimpleDialog
import sys
import copy
import os
import glob
import ntpath
import math
import numpy as np
from skimage import io
from skimage.draw import circle
import tkMessageBox

predictor_path = "alignment_model/shape_predictor_68_face_landmarks.dat" #dlib shape model
network_structure_path_name = "C:/pkg/caffe-windows-master/models/vgg_face_caffe/VGG_FACE_16_deploy.prototxt"
vgg_model_path_name = "C:/pkg/caffe-windows-master/models/vgg_face_caffe/VGG_FACE.caffemodel";
caffe_root = "C:/pkg/caffe-windows-master/"
sys.path.insert(0, caffe_root + 'python')

def _shape_to_np(shape):
    xy = []
    for i in range(68):
        xy.append((shape.part(i).x, shape.part(i).y,))
    xy = np.asarray(xy, dtype='float32')
    return xy


def load_models():
    global label
    label.config(text = "Importing all models...")
    label.update_idletasks()
    global dlib
    import dlib
    global cv2
    import cv2
    global detector
    detector = dlib.get_frontal_face_detector()  #dlib face detector
    global shape_predictor
    shape_predictor = dlib.shape_predictor(predictor_path)
    global opencv_face_cascade
    opencv_face_cascade = cv2.CascadeClassifier("face_detection_opencv_models/haarcascade_frontalface_alt2.xml")
    global caffe
    import caffe
    caffe.set_mode_gpu()
    global net
    net = caffe.Net(network_structure_path_name, vgg_model_path_name,  caffe.TEST) #caffe net model
    global max_cc
    max_cc = np.loadtxt("max_cc.txt")
    global threshold
    threshold = 0.396
    label.config(text = "Models loaded.")
    #print vgg_model_path_name

def _dlib_detect(img) :
    dets = detector(img, 1) #face detection by dlib detector
    if len(dets) == 0 :
        tkMessageBox.showinfo("Warning","Not face image: try again")
        dets = detector(img, 2)
        
    
    max_area = 0
    max_d = dets[0]
    max_k = 0
    for k, d in enumerate(dets): # find the largest rectangle
        current_area = (d.right()-d.left())*(d.bottom()-d.top())
        if current_area > max_area :
            max_area = current_area
            max_d = d
            max_k = k
    return max_d
    
def _load_image(image_path_name):
    global label2
    try:
        image=Image.open(image_path_name)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break

        exif=dict(image._getexif().items())
        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)

        image.save(image_path_name)
        image.close()
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass
        
    #img = io.imread(image_path_name)
    try :
        img_opencv = cv2.imread(image_path_name)
        img = img_opencv[...,::-1].copy()
        print("loaded images")
        print img.shape
        print img.dtype
        if img.shape[2] == 4 :
            print("in case there are four channels")
            img_rgb = img[:,:,0:3].copy()
            print img_rgb.shape
            img = img_rgb

    except:
        tkMessageBox.showinfo("Warning","Not valid image format")
    
    if len(img.shape) == 2: #if gray image, transfer to color
        img_temp = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
        img_temp[:,:,0] = img
        img_temp[:,:,1] = img
        img_temp[:,:,2] = img
        img = img_temp

    face_width = 0 # just for initialization
    
    # check image size
    if img.shape[0] > 600 or img.shape[1] > 700 : # crop or resize
        print img.shape
        label2.configure(text = "Images too big; original images have been cropped")
        d = _dlib_detect(img)
        face_width = d.right()-d.left()
        
        if face_width < 200 : # face not very big, only crop
            left_col = max(d.left()-150,0);
            rigt_col = min(d.right() + 150,img.shape[1]);
            up_row = max(d.top() - 150,0);
            dn_row = min(d.bottom() + 150, img.shape[0]);
            img = img[up_row:dn_row,left_col:rigt_col,:].copy()
            img.astype(np.uint8)

        else :# face too big. resize the image and crop
            ratio = 150.0/face_width
            #print ratio
            small = cv2.resize(img[:,:,::-1], None, fx=ratio, fy=ratio, interpolation = cv2.INTER_CUBIC)
            d = _dlib_detect(small[:,:,::-1].copy())
            face_width = d.right()-d.left()

            left_col = max(d.left()-150,0);
            rigt_col = min(d.right() + 150,small.shape[1]);
            up_row = max(d.top() - 150,0);
            dn_row = min(d.bottom() + 150, small.shape[0]);
            img = small[up_row:dn_row,left_col:rigt_col,::-1].copy()
            img.astype(np.uint8)
            print img.shape
    # check face size
    
    return img
    

def _extract_feature(img, image_path_name, zero_or_one) :
    global panelA, panelB
    global label2
    #label2.config(text = "Detect face features ...")
    label.update_idletasks()
    imgd = copy.deepcopy(img)
    print img.shape
    dets = detector(img, 2) #face detection by dlib detector
    max_area_open = 0
    max_x = 0
    max_y = 0
    max_w = 0
    max_h = 0
    if len(dets) == 0 : # if dlib detector failed, use opencv detection instead
        img_opencv = img[...,::-1]
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
    
    pathname, file_name = os.path.split(image_path_name)
    file_name_noext = os.path.splitext(file_name)[0]
    inter_path = pathname + "/inter_files/";
    if (not(os.path.isdir(inter_path))) :
        os.makedirs(inter_path)
    np.savetxt(inter_path + file_name_noext + "_det.txt",xydet,'%3d') # save the detection result
    # extract shape information (68 points)
    shape = shape_predictor(img, max_d)
    xy = _shape_to_np(shape)
    strtxt = "_alig.txt";
    strjpg = "_alig.jpg";
    np.savetxt(inter_path + file_name_noext +  strtxt, xy, '%3d')
    for ind_part in range(68):
        if (shape.part(ind_part).y + 1) < imgd.shape[0] and (shape.part(ind_part).x + 1) < imgd.shape[1] :
            rr, cc = circle(shape.part(ind_part).y, shape.part(ind_part).x, 2)
            imgd[rr, cc] = [255, 0, 0]
    io.imsave(inter_path + file_name_noext +  strjpg, imgd) # save the extracted shape to disk
    imagedd = Image.fromarray(imgd)
    imagedd = ImageTk.PhotoImage(imagedd)
    if zero_or_one == 0:
        panelA.configure(image=imagedd)
        panelA.image = imagedd
    elif zero_or_one == 1:
        panelB.configure(image=imagedd)
        panelB.image = imagedd
    #label2.config(text = "Alignment completed")

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

def select_image_single0():
    # grab a reference to the image panel1
    global panelA, panelB, panelC
    global image0
    global image00, image11
    global image_name_yes, image_name_no, image_name_blank
    global im_path_name0
    global label2

    image_name_yes = "yes.jpg";
    image_name_no = "no.jpg";
    image_name_blank = "blank.jpg";

    label2.configure(text = "Will show a score here")
 
    # open a file chooser dialog and allow the user to select an input
    # image
    open_filename = tkFileDialog.askopenfilename()
    print open_filename
    if len(open_filename) > 0 :
        image0 = _load_image(open_filename)
        im_path_name0 = open_filename
        image00 = Image.fromarray(image0)
        image00 = ImageTk.PhotoImage(image00)


        if panelA is None :
            panelA = Label(image=image00)
            panelA.image = image00
            panelA.pack(side="left", padx=10, pady=10)

        else :
            panelA.configure(image=image00)
	    panelA.image = image00

	    image_sign = _load_image(image_name_blank)
            image_signs = Image.fromarray(image_sign)
            image_signs = ImageTk.PhotoImage(image_signs)

	    panelC.configure(image=image_signs)
	    panelC.image = image_signs
	    
	    panelB.configure(image=image11)
	    panelB.image = image11


def select_image_single1():
    # grab a reference to the image panel1
    global panelA, panelB, panelC
    global image00, image11
    global image1, image_sign
    global im_path_name1
    global image_name_yes, image_name_no, image_name_blank
    global label2
    image_name_yes = "yes.jpg";
    image_name_no = "no.jpg";
    image_name_blank = "blank.jpg";
 
    # open a file chooser dialog and allow the user to select an input
    # image
    open_filename = tkFileDialog.askopenfilename()
    
    if len(open_filename) > 0 :

        image_sign = _load_image(image_name_blank)
        image_signs = Image.fromarray(image_sign)
        image_signs = ImageTk.PhotoImage(image_signs)

        image1 = _load_image(open_filename);
        im_path_name1 = open_filename
        image11 = Image.fromarray(image1)
        image11 = ImageTk.PhotoImage(image11)

        if panelB is None:
            panelC = Label(image=image_signs)
            panelC.image = image_signs
            panelC.pack(side="left", padx=10, pady=10)

            panelB = Label(image=image11)
            panelB.image = image11
            panelB.pack(side="left", padx=10, pady=10)

        else :
            image_sign = _load_image(image_name_blank)
            image_signs = Image.fromarray(image_sign)
            image_signs = ImageTk.PhotoImage(image_signs)
            panelA.configure(image=image00)
            panelC.configure(image=image_signs)
	    panelB.configure(image=image11)
	    panelA.image = image00
	    panelC.image = image_signs
	    panelB.image = image11


def select_image():
    # grab a reference to the image panels
    global panelA, panelB, panelC
    global image0, image1, image_sign
    global im_path_name0, im_path_name1
    global image_name_yes, image_name_no, image_name_blank
    global label2
    image_name_yes = "yes.jpg";
    image_name_no = "no.jpg";
    image_name_blank = "blank.jpg";
 
    # open a file chooser dialog and allow the user to select an input
    # image
    filez = tkFileDialog.askopenfilenames()
    #print root.tk.splitlist(filez)
    lst = list(filez)
    #print lst[0]
    #print lst[1]
    if len(lst[0]) > 0 and len(lst[1]) > 0:
        label2.configure(text = "Will show a score here")
        image0 = _load_image(lst[0])
        im_path_name0 = lst[0]
        image00 = Image.fromarray(image0)
        image00 = ImageTk.PhotoImage(image00)

        image_sign = _load_image(image_name_blank)
        image_signs = Image.fromarray(image_sign)
        image_signs = ImageTk.PhotoImage(image_signs)

        image1 = _load_image(lst[1]);
        im_path_name1 = lst[1]
        image11 = Image.fromarray(image1)
        image11 = ImageTk.PhotoImage(image11)

        if panelA is None and panelB is None:
            panelA = Label(image=image00)
            panelA.image = image00
            panelA.pack(side="left", padx=10, pady=10)

            panelC = Label(image=image_signs)
            panelC.image = image_signs
            panelC.pack(side="left", padx=10, pady=10)

            panelB = Label(image=image11)
            panelB.image = image11
            panelB.pack(side="left", padx=10, pady=10)

        else :
            panelA.configure(image=image00)
            panelC.configure(image=image_signs)
	    panelB.configure(image=image11)
	    panelA.image = image00
	    panelC.image = image_signs
	    panelB.image = image11


def recognize():
	# grab a reference to the image panels
	#global image0, image1
	#cv2.imshow('img_open',image0)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    
    feature01 = copy.deepcopy(np.squeeze(_extract_feature(image0, im_path_name0, 0)))
    feature02 = copy.deepcopy(np.squeeze(_extract_feature(image1, im_path_name1, 1)))

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
        image_sign = _load_image(image_name_yes)
        image_signs = Image.fromarray(image_sign)
        image_signs = ImageTk.PhotoImage(image_signs)

        panelC.configure(image=image_signs)
        panelC.image = image_signs

        #print("Recognized as the same person \n")
        showText = "Same person; Score = %f" % (similarity);
        label2.config(text = showText)
        #tkMessageBox.showinfo("Result",showText)
    else :
        image_sign = _load_image(image_name_no)
        image_signs = Image.fromarray(image_sign)
        image_signs = ImageTk.PhotoImage(image_signs)

        panelC.configure(image=image_signs)
        panelC.image = image_signs
        
        #print("Not Recognized as the same person \n")
        showText = "Not same person; Score = %f" % (similarity);
        label2.config(text = showText)
        #tkMessageBox.showinfo("Result",showText)

def adjustMarker():
    newval = tkSimpleDialog.askfloat('Set a new threshold', 'Enter a value between 0 and 1', minvalue=0, maxvalue=1.0)
    global threshold
    threshold = newval
    
    #cenCross.Set(offset=newval)
    #self.vf.GUI.VIEWER.Redraw()
    #self.adjustCross.set(0)
                

root = Tk()
panelA = None
panelB = None
panelC = None
image0 = None
image1 = None
image_sign = None

root.configure(background='white')
#root.resizable(width=False, height=False)
#root.geometry("1200x800")

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
label = Label(root, text = "Please load models first", fg="red", bg='white')
label.pack()
btn0 = Button(root, text="Load models", command=load_models)
btn0.pack(side="top", fill="both", expand="yes", padx="10", pady="10")

btn01 = Button(root, text="Set a threshold - default to be 0.396", command=adjustMarker)
btn01.pack(side="top", fill="both", expand="yes", padx="10", pady="10")

#btn1 = Button(root, text="Select two images", command=select_image)
#btn1.pack(side="top", fill="both", expand="yes", padx="10", pady="10")

top = Frame(root,bg='white')
bottom = Frame(root)
top.pack(side=TOP)
bottom.pack(side=BOTTOM, fill=BOTH, expand=True)


b = Button(root, text="Load Image 1", width=15, height=1, command=select_image_single0)
c = Button(root, text="Load Image 2", width=15, height=1, command=select_image_single1)
b.pack(in_=top, side=LEFT,fill="both",expand="yes", padx="10", pady="10")
c.pack(in_=top, side=LEFT,fill="both",expand="yes", padx="10", pady="10")



btn2 = Button(root, text="Verify Identity", command=recognize)
btn2.pack(side="top", fill="both", expand="yes", padx="10", pady="10")

label2 = Label(root, text = "Will show a score here", fg="red",bg='white')
label2.pack()

 
# kick off the GUI
root.mainloop()

