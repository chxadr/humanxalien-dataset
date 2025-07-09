<div align="center">

# 👽 Alien vs Human Dataset 👓

<img src="assets/images/john.jpg" width="400">

</div>

## 🎯 Content

This repository contains a dataset designed for **demonstration purposes**, taking inspiration from the John Carpenter’s movie *They Live* (1988), available for free on the YouTube platform, to explore object detection and classification pipelines using two well-known machine-learning models:

- **[Keras MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)[^1] (classification)**
- **[Ultralytics YOLOv8n](https://docs.ultralytics.com/models/yolov8/)[^2] (object detection)**

[^1]: Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen (2019). *MobileNetV2: Inverted Residuals and Linear Bottlenecks* (Version 4.0.0) [Computer software]. https://arxiv.org/abs/1801.04381

[^2]: Glenn Jocher, Ayush Chaurasia, Jing Qiu (2023). *YOLO by Ultralytics* (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics

The **same dataset is provided, rearranged to fit each model’s requirements**, allowing you to experiment with both **workflows side-by-side**.

### Dataset Constitution

- **Images:** Size 224x224 with consistent scenes and objects under controlled conditions.
- **Capture:** USB 2Mpx Camera, Variable focal length, 2.8-12mm lens, Manual focus, 320x240 MJPEG @ 120FPS.
- **Lighting:** LED Spot lighting to reduce shadows and reflections.
- **Background:** A wooden plank.

### Purposes

- Describe the common points and differences between the YOLOv8n model and the MobileNetV2 model.
- Understand basic model training workflows and practical considerations.
- Learn how to create a dataset for training AI models.
- Explore how to automate the labeling process using Python, in a practical case.

### ✨ Credits

I would like to thank [**Selva Systems**](https://selvasystems.net/), where I completed my internship, for providing the resources that allowed me to capture this dataset, then train and test a [YOLOv8n](https://docs.ultralytics.com/models/yolov8/) model on an [Orange Pi Zero 3](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-Zero-3.html) SBC, with the [Coral USB Accelerator](https://coral.ai/products/accelerator), as part of a project described in my [Edge AI with Orange Pi](https://github.com/chxadr/edgeai-orangepi) repository.

## 📝 Common Tasks and CNNs

### Keras MobileNetV2

### Ultralytics YOLOv8n

## 🏃 Requirements to Train a Model

### Build a Dataset

### Automate the Labeling Process

## 🔎 Practical Example

### Image Capture and Labeling

### Rearranging the Dataset
