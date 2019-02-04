from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse
from pylab import *
import numpy as np
from os import listdir
from sklearn.svm import LinearSVC
import cv2
from PIL import Image
from sklearn import svm
import imagehash
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn import linear_model
import csv


# from PIL import Image
# from scipy.misc import imread, imresize, imsave
import base64
from io import BytesIO

# Create your views here.

#Creating function to Preprocess the image
def preprocess_image(path, display=False):
    raw_image = cv2.imread(path)
    bw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    bw_image = 255 - bw_image

    if display:
        cv2.imshow("RGB to Gray", bw_image)
        cv2.waitKey()

    _, threshold_image = cv2.threshold(bw_image, 30, 255, 0)

    if display:
        cv2.imshow("Threshold", threshold_image)
        cv2.waitKey()

    return threshold_image

#Importing Images from Folders Genuine : Genuine , Forge for Uploading the testing images
genuine_image_filenames = listdir("C:\\Work\\Axis Projects\\SignatureVerification\\Dataset\\dataset1\\real")
forged_image_filenames = listdir("C:\\Work\\Axis Projects\\SignatureVerification\\Dataset\\dataset1\\test")

genuine_image_paths = "C:\\Work\\Axis Projects\\SignatureVerification\\Dataset\\dataset1\\real"
#forged_image_paths = "C:\\Work\\Axis Projects\\SignatureVerification\\Dataset\\dataset1\\test"


def signature_verification (path,user_id,display=False):
    genuine_image_features = [[] for x in range(200)]
    forged_image_features = [[] for x in range(200)]

    # signature_id = 1
    signature_id1 = user_id

    for name in genuine_image_filenames:
        signature_id = int(name[:3])
        genuine_image_features[signature_id - 1].append({"name": name})

    a1 = 0
    for a1 in range(len(path)):
        name1 = path[a1]
        forged_image_features[a1].append({"name": name1})

    # Processing Starts here
    cor = 0
    wrong = 0

    im_contour_features = []
    ch_contour_features = []
    tot_train_genuine = np.empty((0, 54), float32)
    tot_test_genuine = np.empty((0, 54), float32)
    tot_train_forge = np.empty((0, 54), float32)
    tot_test_forge = np.empty((0, 54), float32)

    test_features = forged_image_features[0][0]
    test_name = test_features.get("name")
    # test_user_id = int(test_name[5:8])
    #   test_in_des1=test_features.get("in_des1")
    #  test_cont1=test_features.get("temp_cont1")
    test_user_id = int(user_id)
    check_features = genuine_image_features[test_user_id - 1]
    y = 0
    sift_corr = 0
    sift_wrong = 0
    cont_right = 0
    cont_wrong = 0
    for y in range(len(check_features)):
        check_features1 = check_features[y]
        check_features_name = check_features1.get("name")
        #     check_in_des1=check_features1.get("in_des1")
        #    check_cont1=check_features1.get("temp_cont1")

        gray1 = preprocess_image(genuine_image_paths + "/" + check_features_name)
        gray2 = preprocess_image(path)

        sift1 = cv2.xfeatures2d.SIFT_create()
        check_kp1, check_in_des1 = sift1.detectAndCompute(gray1, None)
        sift2 = cv2.xfeatures2d.SIFT_create()
        tes_kp2, test_in_des1 = sift2.detectAndCompute(gray2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(test_in_des1, check_in_des1, k=2)
        # Apply ratio test

        good = []
        for m, n in matches:
            if m.distance < 0.98 * n.distance:
                good.append([m])
                a = len(good)
                percent = (a * 100) / max(len(test_in_des1), len(check_in_des1))
        if percent >= 50.00:
            sift_corr = sift_corr + 1
        if percent < 50.00:
            sift_wrong = sift_wrong + 1

        og = cv2.imread(genuine_image_paths + "/" + check_features_name)
        og_gray = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)
        ret, temp_thre = cv2.threshold(og_gray, 200, 255, 10)
        _, contours, hierarchy = cv2.findContours(temp_thre, 1, 2)

        temp_cont1 = contours[0]

        dup = cv2.imread(path)
        dup_gray = cv2.cvtColor(dup, cv2.COLOR_BGR2GRAY)
        ret, tar_thr = cv2.threshold(dup_gray, 200, 255, 10)

        _, contours, hierarchy = cv2.findContours(tar_thr, 1, 2)

        temp_cont2 = contours[0]

        for c in contours:
            match = cv2.matchShapes(temp_cont1, temp_cont2, 1, 0)

            if match <= 0.2:
                cont_right = cont_right + 1
            else:
                cont_wrong = cont_wrong + 1

    sift_final = sift_corr / (sift_wrong + sift_corr)
    if sift_final > 0.49:
        sift_status = "Im_Match "
    if sift_final <= 0.49:
        sift_status = "Im_Not_Match "

    cont_final = cont_right / (cont_wrong + cont_right)
    if cont_final > 0.49:
        cont_status = "Genuine"
    if cont_final <= 0.49:
        cont_status = "Forged"
    final_status = sift_status + cont_status
    return cont_status

def main_page(request):
	return render(request, 'sig_verify/index1.html')


def data_return(request):
    if request.method == 'POST':
        accountNumber1 = request.POST.get('accountNumber')

        # Saving received image into Images as FormImage.png
        image1 = request.POST.get('image1')
        image1 = Image.open(BytesIO(base64.b64decode(image1[22:])))
        # imsave('Images/FormImage.png', image1)
        imsave('C:\Work\Axis Projects\SignatureVerification\Frontend\Hackathon\webapp\Images\FormImage.png', image1)
        image_path='C:\Work\Axis Projects\SignatureVerification\Frontend\Hackathon\webapp\Images\FormImage.png'
        # Sending FormImage.png to accountSIgnatureExtraction to get account number and sign image
        # X = accountSignatureExtraction.acc_numo('Images/FormImage.png')
        # [full_name,acc_num,start_date,end_date,sign]
        # signImage = X[4]

        # print("Full_Name " + X[0])
        # print("Account_Number " + str(X[1]))
        # print("Start_Date  " + X[2])
        # print("End_Date  " + X[3])

        # Database
        # Database.Add_Entry(12345678911, "sign.jpg")
        # filename = Database.Extract(X[1])
        xx = 'static/genuine/'+'00'+accountNumber1+'0000'+accountNumber1+'.png'
        print(xx)
        print(accountNumber1)




        # Saving signature image to images as SignImage.png
        # imsave('Images/SignImage.png', signImage)

            # Sending SignImage to signet to get probability
        probability = signature_verification(image_path,accountNumber1)
        print(probability)
        # x = "{% static '" + filename[0] + "' %}"
        # returning probability
        data = {'account': str(123), 'prob': str(probability), 'sig1': xx}
        print("save success!")
        # os.remove( r"C:\Users\HARSH\Desktop\FinalApp\mysite\Images\FormImage.png")
        return JsonResponse(data)
    else:
        data = {'is_taken': "NotAPostRequest"}
        return JsonResponse(data)


def index(request):
    return HttpResponse("<H2>YOO BRO !!   YOU DID IT   </H2>")