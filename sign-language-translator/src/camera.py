import cv2
import numpy as np
import onnxruntime as ort
f = open('output.txt', 'w')
f.truncate(0)
f.close()
saved = None

def center_crop(frame):
    h, w, _ = frame.shape
    start = abs(h - w) // 2
    if h > w:
        return frame[start: start + w]
    return frame[:, start: start + h]

def checkClick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global saved
        with open("output.txt", 'a') as f:
            f.write(saved)

def main():
    # constants
    index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')
    mean = 0.485 * 255.
    std = 0.229 * 255.
    # create runnable session with exported model
    ort_session = ort.InferenceSession("signlanguage.onnx")

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Click to add letter")
    cv2.setMouseCallback("Click to add letter", checkClick)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # preprocess data
        frame = center_crop(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        x = cv2.resize(frame, (28, 28))
        x = (x - mean) / std

        x = x.reshape(1, 1, 28, 28).astype(np.float32)
        y = ort_session.run(None, {'input': x})[0]

        index = np.argmax(y, axis=1)
        letter = index_to_letter[int(index)]
        global saved
        saved = letter
        cv2.putText(frame, letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        cv2.imshow("Sign Language Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            with open("output.txt", 'r') as f:
                print("Your input was: " + f.read())
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
