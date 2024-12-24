from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import uvicorn
import torch
from torchvision import transforms
from PIL import Image
import os
from torchvision import models

# Define the model architecture (same as during training)
model = models.resnet18(pretrained=False, num_classes=120)  # Adjust num_classes as needed

# Load the saved state dictionary
MODEL_PATH = "./model.pth"
state_dict = torch.load(MODEL_PATH, map_location="cpu")  # Use "cuda" if on GPU
model.load_state_dict(state_dict)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class labels
class_labels = ['affenpinscher', 'Afghan hound', 'African hunting dog', 'Airedale', 'American Staffordshire terrier', 'Appenzeller', 'Australian terrier', 'basenji', 'basset', 'beagle', 'Bedlington terrier', 'Bernese mountain dog', 'black and tan coonhound', 'Blenheim spaniel', 'bloodhound', 'bluetick', 'Border collie', 'Border terrier', 'borzoi', 'Boston bull', 'Bouvier des Flandres', 'boxer', 'Brabancon griffon', 'briard', 'Brittany spaniel', 'bull mastiff', 'cairn', 'Cardigan', 'Chesapeake Bay retriever', 'Chihuahua', 'chow', 'clumber', 'cocker spaniel', 'collie', 'curly-coated retriever', 'Dandie Dinmont', 'dhole', 'dingo', 'Doberman', 'English foxhound', 'English setter', 'English springer', 'EntleBucher', 'Eskimo dog', 'flat-coated retriever', 'French bulldog', 'German shepherd', 'German short-haired pointer', 'giant schnauzer', 'golden retriever', 'Gordon setter', 'Great Dane', 'Great Pyrenees', 'Greater Swiss Mountain dog', 'groenendael', 'Ibizan hound', 'Irish setter', 'Irish terrier', 'Irish water spaniel', 'Irish wolfhound', 'Italian greyhound', 'Japanese spaniel', 'keeshond', 'kelpie', 'Kerry blue terrier', 'komondor', 'kuvasz', 'Labrador retriever', 'Lakeland terrier', 'Leonberg', 'Lhasa', 'malamute', 'malinois', 'Maltese dog', 'Mexican hairless', 'miniature pinscher', 'miniature poodle', 'miniature schnauzer', 'Newfoundland', 'Norfolk terrier', 'Norwegian elkhound', 'Norwich terrier', 'Old English sheepdog', 'otterhound', 'papillon', 'Pekinese', 'Pembroke', 'Pomeranian', 'pug', 'redbone', 'Rhodesian ridgeback', 'Rottweiler', 'Saint Bernard', 'Saluki', 'Samoyed', 'schipperke', 'Scotch terrier', 'Scottish deerhound', 'Sealyham terrier', 'Shetland sheepdog', 'Shih-Tzu', 'Siberian husky', 'silky terrier', 'soft-coated wheaten terrier', 'Staffordshire bullterrier', 'standard poodle', 'standard schnauzer', 'Sussex spaniel', 'Tibetan mastiff', 'Tibetan terrier', 'toy poodle', 'toy terrier', 'vizsla', 'Walker hound', 'Weimaraner', 'Welsh springer spaniel', 'West Highland white terrier', 'whippet', 'wire haired fox terrier', 'Yorkshire terrier']

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image = Image.open(file.file).convert("RGB")

        # Apply transformations
        input_tensor = transform(image).unsqueeze(0).to(torch.device('cpu'))

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            label = class_labels[predicted.item()]

        return JSONResponse(content={"label": label})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Custom OpenAPI schema
@app.get("/openapi.json", include_in_schema=False)
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Image Classification API",
        version="1.0.0",
        description="API for classifying images using a PyTorch model",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
