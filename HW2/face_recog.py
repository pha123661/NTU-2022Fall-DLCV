import face_recognition
import os
import tqdm
import argparse

def face_recog(image_dir):
    image_ids = os.listdir(image_dir)
    total_faces = len(image_ids)
    num_faces = 0
    print("Start face recognition...")
    for image_id in tqdm.tqdm(image_ids):
        image_path = os.path.join(image_dir, image_id)
        try: # Prevent unexpected file
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image, model="HOG")
            if len(face_locations) == 1:
                num_faces += 1
        except:
            total_faces -= 1
    acc = (num_faces / total_faces) * 100
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help="path to the folder for output images", type=str)
    args = parser.parse_args()

    acc = face_recog(args.image_dir)
    print("Face recognition Accuracy: {:.3f}%".format(acc))