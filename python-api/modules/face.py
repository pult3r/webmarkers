import face_recognition
def detect_faces(path):
    img = face_recognition.load_image_file(path)
    return {'faces': face_recognition.face_locations(img)}
def recognize_face(path):
    img = face_recognition.load_image_file(path)
    enc = face_recognition.face_encodings(img)
    return {'encodings': [e.tolist() for e in enc]}