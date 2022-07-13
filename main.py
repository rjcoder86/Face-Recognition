import face_recognition
import cv2

# Enter a path of first image
path1=r"test_pics\jd.jpg"

known_image = face_recognition.load_image_file(path1)

# locations of faces found in first image
known_img_loc = face_recognition.face_locations(known_image)
for i in known_img_loc:
    t, r, b, l = i
    cv2.rectangle(known_image, (l, t), (r, b), (255, 255, 1), 4)
cv2.imshow('locations', known_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('Faces Found :', len(known_img_loc))

# select a index of face to search
search_face_index = 0   #default first
print('Face index selected to search :', search_face_index)
known_image_encoding = face_recognition.face_encodings(known_image)[search_face_index]


# Enter a path of second image
path2=r"test_pics\modi_obama.jpg"
unknown_image = face_recognition.load_image_file(path2)

#locations of faces found to check
unknown_img_loc = face_recognition.face_locations(unknown_image)


# iterating though all found faces to compare
for i in range(len(unknown_img_loc)):
    unknown_image_encoding = face_recognition.face_encodings(unknown_image)[i]
    results = face_recognition.compare_faces([known_image_encoding], unknown_image_encoding)
    if results == [True]:
        t, r, b, l = unknown_img_loc[i]
        cv2.rectangle(unknown_image, (l, t), (r, b), (255, 255, 1), 4)
        cv2.imshow('locations', unknown_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('Face found in image')
        break
