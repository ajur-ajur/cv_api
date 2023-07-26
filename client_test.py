#############################################
#											#
#	Ceritanya ini berarti pc yang client	#
#	folder face_detector server				#
#############################################

# import the necessary packages
import requests
import cv2

# define the URL to our face detection API
url = 'http://localhost:8000/face_detection/detect/' #ini url local host, kalau ganti port berarti ganti
concat = 'aldira' #ganti sama nama file tanpa .jpg nya
img_dir = 'images/'+concat+'.jpg'
font = cv2.FONT_HERSHEY_DUPLEX

img = cv2.imread(img_dir)
payload = {'image': open(img_dir, 'rb')}

r = requests.post(url, files=payload).json()
print ("Prediction: {}".format(r))

for (startX, startY, endX, endY) in r['faces']:
	cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
	cv2.putText(img, "Nama: {}".format(r['name']), (startX + 20, startY + - 15), font, 1, (0, 255, 0), 2)    #masih ada bug dalam deteksi nama, mungkin harus ganti model
	cv2.putText(img, "Gender: {}".format(r['gender']), (startX + 20, startY + 30), font, 1, (0, 255, 0), 1)		
	cv2.putText(img, "Usia: {}".format(r['age']), (startX + 20, startY + 60), font, 1, (0, 255, 0), 1)
	cv2.putText(img, "Emosi: {}".format(r['emotion']), (startX + 20, startY + 90), font, 1, (0, 255, 0), 1)
	
cv2.namedWindow("prediksi", cv2.WINDOW_NORMAL)
cv2.imshow("prediksi", img)
cv2.waitKey(0)