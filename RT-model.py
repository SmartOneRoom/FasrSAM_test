from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import cv2
import numpy as np
import logging
import traceback
import time
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_delivery_box(annotations, confidence_threshold=0.5, size_threshold=0.1):
    for ann in annotations:
        # 신뢰도 확인
        if ann['confidence'] < confidence_threshold:
            continue
        
        # 박스 크기 확인 (이미지 크기의 10% 이상)
        box = ann['bbox']
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        image_area = ann['segmentation'].shape[0] * ann['segmentation'].shape[1]
        if box_area / image_area < size_threshold:
            continue
        
        # 여기에 추가적인 조건을 넣을 수 있습니다 (예: 형태, 색상 등)
        
        return True
    return False

def process_frame(frame, model, prompt_process, box_detected):
    try:
        # Run inference on the frame
        results = model(frame, device="cpu", retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        
        # Update the prompt process with new results
        prompt_process.results = results
        prompt_process.ori_img = frame
        
        # Apply the pre-defined prompt (e.g., text prompt)
        ann = prompt_process.text_prompt(text="delivery box")
        
        # Check if a delivery box is detected
        if not box_detected and detect_delivery_box(ann):
            logging.info("Delivery box detected!")
            # Save the annotated frame
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"delivery_box_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(output_path, prompt_process.plot(annotations=ann))
            logging.info(f"Annotated image saved to: {output_path}")
            return prompt_process.plot(annotations=ann), True
        
        # Plot the results on the frame
        annotated_frame = prompt_process.plot(annotations=ann)
        
        return annotated_frame, box_detected
    
    except Exception as e:
        logging.error(f"An error occurred while processing frame: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return frame, box_detected

try:
    # Create a FastSAM model
    logging.info("Creating FastSAM model...")
    model = FastSAM("FastSAM-s.pt")  # or FastSAM-x.pt
    logging.info("FastSAM model created successfully")

    # Open video capture
    cap = cv2.VideoCapture(0)  # 0 for default camera, or use a video file path
    
    # Read first frame to initialize FastSAMPrompt
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Failed to read from video source")
    
    # Initialize FastSAMPrompt with the first frame
    prompt_process = FastSAMPrompt(frame, None, device="cpu")

    box_detected = False

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        start_time = time.time()
        annotated_frame, box_detected = process_frame(frame, model, prompt_process, box_detected)
        end_time = time.time()

        # Calculate and display FPS
        fps = 1 / (end_time - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the result
        cv2.imshow('FastSAM Real-time', annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
    logging.error(f"Traceback: {traceback.format_exc()}")

finally:
    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()