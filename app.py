import  gradio as  gr
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

#Load the trained weights
model = torch.load("data_model_28.pt")


#print(model.eval())
target_layers = [model.layer4[-1]]

class_labels = [ 'NORMAL', 'PNEUMONIA']


norm_mean = (0.485, 0.456, 0.406)
norm_std = (0.229, 0.224, 0.225)

transform = transforms.Compose([ # resize image to the network input size
                  transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                  transforms.RandomRotation(degrees=15),
                  transforms.RandomHorizontalFlip(),
                  transforms.CenterCrop(size=224),
                  transforms.ToTensor(),
                  transforms.Normalize(norm_mean, norm_std)
                  ])


# convert tensor to numpy array
#def tensor2npimg(tensor, mean, std): 
    # inverse of normalization
   # tensor.device = 'cpu'
   # tensor = tensor.clone()
   # mean_tensor = torch.as_tensor(list(mean), dtype=tensor.dtype, device='cpu').view(-1,1,1)
  #  std_tensor = torch.as_tensor(list(std), dtype=tensor.dtype, device='cpu').view(-1,1,1)
   # tensor.cpu().mul_(std_tensor).add_(mean_tensor)
  # convert tensor to numpy format for plt presentation
    #npimg =  tensor.cpu().detach().numpy()
   # npimg = tensor.detach().cpu().numpy()
   # print(npimg.shape)
    #npimg = np.transpose(npimg,(3, 2, 0,1)) # C*H*W => H*W*C
    
 #   print(npimg.shape)
    
  #  return npimg




def predict_lungs(img):
    
    #Classification of Lungs Disease
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    test_image_tensor = transform(img)
    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
    
    with torch.no_grad():
         # probabilities of all classes
        prediction = torch.nn.functional.softmax(model(test_image_tensor)[0], dim=0)
        
  # class with hightest probability
    pred = torch.argmax(model(test_image_tensor)[0], dim=0) # (outputs, dim=1).cpu().numpy()
  # diagnostic suggestions
    if pred == 1:
        suggestion = "Consult your medical doctor for treatment!"
    else:
        suggestion = "Nothing to be worried about."
    
        
    return {class_labels[i]: float(prediction[i]) for i in range(2)}, suggestion #, output_img

inputs = gr.inputs.Image()
outputs = [gr.outputs.Label(num_top_classes=2, label="Predict Result"), gr.outputs.Textbox(label="Medical Recommendation") ] #, gr.outputs.Image(type='numpy', label="GRADCAM")]
gr.Interface(fn=predict_lungs, 
             inputs=inputs, 
             outputs=outputs, 
             title="Chest X-Ray Diagnostic Application Tool",
             description="A diagnostic medical tool that predicts the existence of pneumonia in an X-ray image",
             interpretation="default"
).launch()
