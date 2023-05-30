from torchvision import transforms

class Transforms:
    def __init__(self):
        #series of transformations to apply
        self.transform = transforms.Compose([transforms.Resize((128,128)),
                                     transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ]
                                    )
    def __call__(self, image):
        return self.transform(image)