from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import os
import logging
import traceback

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Define an inference source
    source = "data/2.jpg"
    logging.info(f"Processing image: {source}")

    # Create a FastSAM model
    logging.info("Creating FastSAM model...")
    model = FastSAM("FastSAM-s.pt")  # or FastSAM-x.pt
    logging.info("FastSAM model created successfully")

    # Run inference on an image
    logging.info("Running inference on the image...")
    everything_results = model(source, device="cpu", retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
    logging.info("Inference completed")

    # Prepare a Prompt Process object
    logging.info("Preparing Prompt Process object...")
    prompt_process = FastSAMPrompt(source, everything_results, device="cpu")
    logging.info("Prompt Process object prepared")

    # Everything prompt
    logging.info("Applying Everything prompt...")
    ann = prompt_process.everything_prompt()
    logging.info("Everything prompt applied")

    # Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
    logging.info("Applying Box prompt...")
    ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])
    logging.info("Box prompt applied")

    # Text prompt
    logging.info("Applying Text prompt...")
    ann = prompt_process.text_prompt(text="delivery things")
    logging.info("Text prompt applied")

    # Point prompt
    logging.info("Applying Point prompt...")
    ann = prompt_process.point_prompt(points=[[200, 200]], pointlabel=[1])
    logging.info("Point prompt applied")

    # Create output directory if it doesn't exist
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # Save the annotated image
    output_path = os.path.join(output_dir, "annotated_image.jpg")
    logging.info(f"Saving annotated image to {output_path}...")
    prompt_process.plot(annotations=ann, output=output_path)
    logging.info("Annotated image saved successfully")

    print(f"Annotated image saved to: {output_path}")

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
    logging.error(f"Traceback: {traceback.format_exc()}")