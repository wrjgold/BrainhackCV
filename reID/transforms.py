from torchvision import transforms

class Transforms:
    def __init__(self):
        #series of transformations to apply
        self.transform = transforms.Compose([transforms.Resize((105,105)),
                                     transforms.ToTensor()])
                                    
    def __call__(self, image):
        return self.transform(image)
