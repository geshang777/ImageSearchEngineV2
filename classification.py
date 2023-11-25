from ultralytics import YOLO
def yoloclas(img):
    model = YOLO('yolov8l-cls.pt')  # load an official model
    ind2word = {}
    with open('id2class.txt', 'r') as f:
        for index,line in enumerate(f):
            idx, class_name = line.strip().split(' ')
            ind2word[index] = class_name
    # ind2word =  {0: 'sushi', 1: 'tacos', 2: 'takoyaki', 3: 'tiramisu', 4: 'tuna_tartare', 5: 'waffles'}
    image=[]
    image.append(img)
    # model = YOLO('best.pt')  # load a custom model
    # model.export(format='onnx')

    # Predict with the model
    results = model(image)  # predict on an image
    cla= None
    for result in results:
        print(result.keypoints)
        cla = ind2word[result.probs.top1] # Probs object for classification outputs

    return cla
if __name__=="__main__":
    img = ["/Users/geshang/Downloads/food_101/images/tacos/3804283.jpg"]
    ind2word =  {0: 'sushi', 1: 'tacos', 2: 'takoyaki', 3: 'tiramisu', 4: 'tuna_tartare', 5: 'waffles'}
    results=yoloclas(img)
    # model = YOLO('best.pt')  # load a custom model
    # model.export(format='onnx')
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = ind2word[result.probs.top1] # Probs object for classification outputs
        print(probs)




