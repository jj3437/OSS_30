import cv2

def person_only(input_path, output_path, format='mp4'):
    # Load Haar cascade for full-body detection
    person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("Cannot open input video file.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Select codec based on the format
    if format == 'mp4':
        output_file = f"{output_path}.mp4"
        codec = cv2.VideoWriter_fourcc(*'mp4v')
    elif format == 'avi':
        output_file = f"{output_path}.avi"
        codec = cv2.VideoWriter_fourcc(*'XVID')
    else:
        raise ValueError("Unsupported format. Use 'mp4' or 'avi'.")

    # Initialize video writer
    out = cv2.VideoWriter(output_file, codec, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        persons = person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Create a black mask
        mask = frame.copy() * 0
        for (x, y, w, h) in persons:
            mask[y:y + h, x:x + w] = frame[y:y + h, x:x + w]

        out.write(mask)

    cap.release()
    out.release()
    print(f"Processing completed. Saved to {output_file}")
