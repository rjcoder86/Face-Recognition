import os
import face_recognition
import cv2

known_image = face_recognition.load_image_file(r"test_pics/jd.jpg")
known_img_loc = face_recognition.face_locations(known_image)
print('Faces Found to search :', len(known_img_loc))

for i in known_img_loc:
    t, r, b, l = i
    cv2.rectangle(known_image, (l, t), (r, b), (255, 255, 1), 4)
cv2.imshow('Knoen Image', known_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# select a index of face to search
search_face_index = 0
print('Face index selected to search :', search_face_index)
known_image_encoding = face_recognition.face_encodings(known_image)[search_face_index]

path = r"test_pics"
all_img = list(os.path.join(path, str(i)) for i in os.listdir(path))
for i in all_img:
    unknown_image = face_recognition.load_image_file(i)
    # print(f'{len(face_recognition.face_locations(unknown_image))} Faces Found')
    loc = face_recognition.face_locations(unknown_image)
    for j in range(len(loc)):
        unknown_image_encoding = face_recognition.face_encodings(unknown_image)[j]
        results = face_recognition.compare_faces([known_image_encoding], unknown_image_encoding)
        if results == [True]:
            print('Face found in image', i)
            t, r, b, l = loc[j]
            cv2.rectangle(unknown_image, (l, t), (r, b), (255, 255, 1), 4)
            cv2.imshow('Image', unknown_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
