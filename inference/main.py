import sys
import os
#sys.path.append('.')
#sys.path.append('/home/cmcc/caffe-master/python')
import tools_matrix as tools
import caffe
import cv2
import numpy as np
deploy = '12net.prototxt'
caffemodel = '12net.caffemodel'
net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = '24net.prototxt'
caffemodel = '24net.caffemodel'
net_24 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = '48net.prototxt'
caffemodel = '48net.caffemodel'
net_48 = caffe.Net(deploy,caffemodel,caffe.TEST)
test_image_path="../DMS_data/yawn"######################need to write

def get_all_files(dir):
    files_ = []
    list = os.listdir(dir)
    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
        if os.path.isdir(path):
            files_.extend(get_all_files(path))
        if os.path.isfile(path):
            files_.append(path)
    return files_


def detectFace(img_path,threshold):
    img = cv2.imread(img_path)
    caffe_img = (img.copy()-127.5)/128
    origin_h,origin_w,ch = caffe_img.shape
    scales = tools.calculateScales(img)
    out = []
    for scale in scales:
        hs = int(origin_h*scale)
        ws = int(origin_w*scale)
        scale_img = cv2.resize(caffe_img,(ws,hs))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_12.blobs['data'].reshape(1,3,ws,hs)
        net_12.blobs['data'].data[...]=scale_img
        caffe.set_device(0)
        caffe.set_mode_gpu()
        out_ = net_12.forward()
        out.append(out_)
    image_num = len(scales)
    rectangles = []
    for i in range(image_num):    
        cls_prob = out[i]['prob1'][0][1]    #s（见我手动画的图）网格点上有人脸的概率
        roi      = out[i]['conv4-2'][0]     #pnet卷积后得到特征图上bbox的坐标
        out_h,out_w = cls_prob.shape    # 经过pnet 卷积后s网格的长和宽
        out_side = max(out_h,out_w)
        rectangle = tools.detect_face_12net(cls_prob,roi,out_side,1/scales[i],origin_w,origin_h,threshold[0])
        rectangles.extend(rectangle)
    rectangles = tools.NMS(rectangles,0.7,'iou')




    if len(rectangles)==0:
        return rectangles
    net_24.blobs['data'].reshape(len(rectangles),3,24,24)
    crop_number = 0
    for rectangle in rectangles:
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img,(24,24))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_24.blobs['data'].data[crop_number] =scale_img 
        crop_number += 1
    out = net_24.forward()
    cls_prob = out['prob1']
    roi_prob = out['conv5-2']
    rectangles = tools.filter_face_24net(cls_prob,roi_prob,rectangles,origin_w,origin_h,threshold[1])
    
    
    
    
    if len(rectangles)==0:
        return rectangles
    net_48.blobs['data'].reshape(len(rectangles),3,48,48)
    crop_number = 0
    for rectangle in rectangles:
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img,(48,48))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_48.blobs['data'].data[crop_number] =scale_img 
        crop_number += 1
    out = net_48.forward()
    cls_prob = out['prob1']
    roi_prob = out['conv6-2']
    pts_prob = out['conv6-3']
    rectangles = tools.filter_face_48net(cls_prob,roi_prob,pts_prob,rectangles,origin_w,origin_h,threshold[2])

    return rectangles

threshold = [0.4,0.4,0.5]
k=0
len_test_files = len(get_all_files(test_image_path))
all_test_path = get_all_files(test_image_path)
for i in range(0,len_test_files):
    print("loading ", i, " picture")
    imgpath = all_test_path[i]
    image_name = imgpath[len(imgpath)-imgpath[::-1].find('/'):]
    print(image_name)
    try:
        rectangles = detectFace(imgpath,threshold)
        img = cv2.imread(imgpath)
        h_image = img.shape[0]
        w_image = img.shape[1]

        #draw = img
        if len(rectangles) == 0:
            k=k+1
            cv2.imwrite("../yawn_crop_Data/error_data/"+image_name,img)########################need to write
        else :
            #img = cv2.imread(imgpath)
            #draw = img.copy()
            j=0
            for rectangle in rectangles:
                j=j+1
                w=int(rectangle[13])-int(rectangle[11])
                h=2*(int(rectangle[12])-int(rectangle[10]))
                new_x1=int(rectangle[11])-int(float(0.2*w))
                new_x2=int(rectangle[13])+int(float(0.2*w))
                new_y1=int(min(rectangle[14],rectangle[12]))-int(float(0.35*h))
                new_y2=int(max(rectangle[14],rectangle[12]))+int(float(0.6*h))
                
                if new_x1<0 :
                    new_x1=0
                if new_y1<0:
                    new_y1=0
                if new_x2>w_image:
                    new_x2=w_image
                if new_y2>h_image:
                    new_y2=h_image
                #cv2.rectangle(img,(new_x1,new_y1),(new_x2,new_y2),(0,255,0),2)
                #for i in range(5,15,2):
                    #cv2.circle(img,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
                #cv2.rectangle(img,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
                img_crop=img[new_y1:new_y2,new_x1:new_x2]
            
                print(int(rectangle[0]),int(rectangle[1]),int(rectangle[2]),int(rectangle[3]))
                print(w,h)
                print(new_x1,new_y1,new_x2,new_y2)
                #cv2.imwrite("../testimg/crop_yawn/"+str(j)+'_'+image_name,img_crop)###########################need to write
            cv2.imwrite("../yawn_crop_Data/yawn/"+str(j)+'_'+image_name,img_crop)
    except:
        continue
        #print(rectangle)
        #cv2.putText(draw,str(rectangle[4]),(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        #cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
        #for i in range(5,15,2):
            #cv2.circle(draw,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
#cv2.imshow("test",draw)
#cv2.waitKey():
#cv2.imwrite('test.jpg',draw)
print("all picture number :",len_test_files)

print("loss picture number :",k)
