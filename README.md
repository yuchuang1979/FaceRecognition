# Face Recognition
This is a face recognition demo using VGG_FACE.caffemodel. The sample code can be run on Windows.

Guideline to install packages used for face recognition pipeline

Should be clear enough; please let me know if you find anything wrong. Please follow these steps one by one - do not skip.

1. Install Python 2.7.11 64bit version: Windows x86-64 MSI installer download from https://www.python.org/downloads/windows/
Must be a 64bit version. If not loading large models would be very difficult.
You may need to restart the computer to use python in Windows Command Prompt.

2. Install Numpy+MKL( IntelÆ Math Kernel Library) 64bit package
   a) download it from http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy [file name is: numpy-1.11.0+mkl-cp27-cp27m-win_amd64.whl]
   Or I have downloaded and put it in a folder named: "softwareAndPackages"
   b) install the package in windows command prompt: 
   pip install [download path]numpy-1.11.0+mkl-cp27-cp27m-win_amd64.whl

3. Install 64bit OpenCV python wrapper (I used it for frontalizing face images w.r.t. alignment results and cropping)
   a) download opencv_python-2.4.12-cp27-none-win_amd64.whl from http://www.lfd.uci.edu/~gohlke/pythonlibs/
   b) install the wrapper in windows command prompt: 
   pip install [download path]opencv_python-2.4.12-cp27-none-win_amd64.whl

4. Install dlib-18.18 for face alignment
   a) make sure you have C++ compiler installed. In the following I am using MS Visual Studio 2013 (MSVC 12.0). If you do not have it, you can install one which can be download for free.
   b) download dlib-18.18 from http://dlib.net/ and decompress it to a folder
   c) download and install cmake software from https://cmake.org/download/. I use cmake-3.4.3-win32-x86.exe (a previous version is always more safe)
   d) download and install boost binary boost_1_57_0-msvc-12.0-64.exe from https://sourceforge.net/projects/boost/files/boost-binaries/1.57.0/
   This is a must if you want to compile dlib package.
   e) You need to set up two environment variables related to boost in the windows system, 
   e.g. BOOST_ROOT = C:\local\boost_1_57_0 and BOOST_LIBRARYDIR = C:\local\boost_1_57_0\lib64-msvc-12.0
   f) Cd to the dlib-18.18 root folder in Windows Command Prompt and compile the python-dlib wrapper (in my case 64bit version):
      python setup.py install

5. Install python package skimage (for some imageIO, processing etc.)
   a) Download scikit_image-0.12.3-cp27-cp27m-win_amd64.whl from http://www.lfd.uci.edu/~gohlke/pythonlibs/
   b) install the wrapper in windows command prompt: pip install scikit_image-0.12.3-cp27-cp27m-win_amd64.whl

6. Install scipy: pip install scipy-0.17.0-cp27-none-win_amd64.whl this file could be downloaded as above files

7. Install matplotlib-1.5.1-cp27-none-win_amd64.whl (optional, if you want to draw something)

The following steps are used for installing Caffe-Windows with Python support:

8. Decompress caffe-windows-master.zip to some folder, then start to build the software. Before building you need to setup according to following steps.
   a) Before build the solutions, pip install the python package protobuf-3.0.0b2-py2.py3-none-any.whl. It could be found and downloaded online:
   https://pypi.python.org/pypi/protobuf/3.0.0b2
   b) Before build the solutions, make sure you have installed Cuda_7.5_windows.exe (search and download it). You may need to restart the computer to continue.
   c) In subfolder 'buildVS2013' of caffe-windows-master, open MainBuilder.sln by MSVC 2013. Build the solution. You may got an error related to Matlab if you want to build the project of matcaffe but do not have Matlab installed and/or do not set Matlab path in this project properly. But this error will not affect the pycaffe we will use. Ignore it.
   d) Add [root path]\caffe-windows-master\bin and [root path]\caffe-windows-master\3rdparty\bin into the system environment variable 'PATH'. You may need to restart the computer to use it.

9. Now you can run face_recognition.py as a demo. Note: you should modify the path to model files in face_recognition.py. 
   max_cc.txt is used to record the vector used for normalization.
   alignment_model folder contains the alignment model.
   face_detection_opencv_models folder contains the detection model.
   examples folder contians the sample images.
   softwareAndPackages folder contains some of above package files and exe installation files.
   


