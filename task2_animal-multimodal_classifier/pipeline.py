import argparse
from models.ner.inference_ner import predict_text_animal
from models.image_classification.inference_img import predict_image_animal

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline that checks if the animal in the text matches the animal in the image"
    )
    parser.add_argument("--text", type=str, required=True, help="Text containing an animal")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--ner_model_path", type=str, default="trained_models/ner_model")
    parser.add_argument("--label_map_path", type=str, default="trained_models/ner_model/label2id.json")
    parser.add_argument("--img_model_path", type=str, default="trained_models/img_model.pth")
    parser.add_argument("--classes_path", type=str, default="data/classes.json")
    
    args = parser.parse_args()
    text_animal = predict_text_animal(
        args.text,
        args.ner_model_path,
        args.label_map_path
    )
    image_animal = predict_image_animal(
        args.image,
        args.img_model_path,
        args.classes_path
    )
    print(f"Animal in text: {text_animal}")
    print(f"Animal in image: {image_animal}")

    match = (text_animal == image_animal) if text_animal is not None else False
    print(match) 

if __name__ == "__main__":
    main()


