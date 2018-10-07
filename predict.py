import torch
import numpy as np
from torchvision import models
from helper import process_image

def load_model(model_checkpoint):
    checkpoint = torch.load(model_checkpoint)
    
    model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["state_dict"])
    class_idx_mapping = checkpoint["class_idx_mapping"]
    idx_class_mapping = {v: k for k, v in class_idx_mapping.items()}
    
    return model, idx_class_mapping

def predict(image_path, model_checkpoint, topk=5):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    
    Arguments:
        image_path: Path to the image
        model: Trained model
    '''
    
    # Build the model from the checkpoint
    model, idx_class_mapping = load_model(model_checkpoint)
    
    # No need for GPU
    model.to("cpu")
    
    model.eval()
     
    img = process_image(image_path)
    img = np.expand_dims(img, axis=0)
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor).to('cpu')
    
    with torch.no_grad():
        log_probabilities = model.forward(img_tensor)
    
    probabilities = torch.exp(log_probabilities)
    probs, indices = probabilities.topk(5)
    
    probs = probs.numpy().squeeze()
    indices = indices.numpy().squeeze()
    classes = [idx_class_mapping[index] for index in indices]
    
    return probs, classes