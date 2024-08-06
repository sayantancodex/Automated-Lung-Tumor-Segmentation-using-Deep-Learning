
# Automated Lung Tumor Segmentation Using Deep Learning

**Author:** Sayantan Patra  
**Degree:** Bachelor of Technology, Computer Science and Engineering  
**Institution:** Siksha 'O' Anusandhan University, Bhubaneshwar, Odisha, India  
**Email:** sayantanpatra68@gmail.com  

## Abstract

Lung cancer remains a leading cause of cancer-related mortality worldwide, highlighting the necessity for early and accurate detection to improve patient outcomes. Manual segmentation of lung tumors from computed tomography (CT) images is labor-intensive and subjective, resulting in variability in results. This project develops a deep learning-based approach for lung tumor segmentation using the UNET model, known for its effectiveness in biomedical image segmentation. By automating the segmentation process, this study aims to enhance the precision and efficiency of diagnosis and treatment planning for lung cancer patients. The proposed method demonstrates significant improvements in segmentation accuracy and consistency, thereby reducing the workload of radiologists and improving patient care.

## 1. Introduction

### 1.1 Background

Lung cancer accounts for a substantial number of cancer-related deaths globally, underscoring the importance of early detection and accurate diagnosis. Traditional manual segmentation methods are time-consuming and require extensive expertise, leading to inconsistent results. The advent of deep learning offers a promising solution to these challenges, enabling automated, accurate, and efficient lung tumor segmentation from CT images.

### 1.2 Problem Statement

Manual segmentation of lung tumors is not only time-consuming but also subjective, leading to variability in results. Automating this process using deep learning can significantly reduce the workload of radiologists and improve the consistency and accuracy of tumor identification.

### 1.3 Objectives

- Develop a deep learning model for automated lung tumor segmentation.
- Evaluate the model's performance using various metrics.
- Compare the automated approach with traditional manual segmentation methods.

## 2. Literature Review

### 2.1 Traditional Segmentation Methods

Traditional methods for lung tumor segmentation involve manual delineation by radiologists, which is labor-intensive and prone to inter-observer variability. Semi-automated methods using image processing techniques such as thresholding, region growing, and edge detection have been proposed but still require significant manual intervention.

### 2.2 Deep Learning in Medical Imaging

Deep learning, particularly convolutional neural networks (CNNs), has revolutionized medical imaging by enabling automated feature extraction and classification. Models like UNET, V-Net, and their variants have shown promising results in various medical image segmentation tasks, including brain tumor segmentation, cardiac segmentation, and lung nodule detection.

### 2.3 UNET Architecture

The UNET model, introduced by Ronneberger et al., is specifically designed for biomedical image segmentation. Its U-shaped architecture, consisting of a contracting path (encoder) and an expansive path (decoder) with skip connections, allows it to capture both local and global features, making it highly effective for segmentation tasks.

## 3. Methodology

### 3.1 UNET Model Overview

The UNET model's architecture consists of a contracting path that captures context and an expansive path that enables precise localization. The model's key components include:

#### 3.1.1 Contracting Path (Encoder)

The contracting path consists of repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a ReLU activation and a 2x2 max pooling operation for down-sampling. At each down-sampling step, the number of feature channels is doubled.

Mathematically, each convolution operation can be expressed as:

\[ x^{(l+1)} = \sigma(W^{(l)} * x^{(l)} + b^{(l)}) \]

where \( x^{(l)} \) is the input feature map at layer \( l \), \( W^{(l)} \) and \( b^{(l)} \) are the weight and bias parameters of the convolutional layer, \( * \) denotes the convolution operation, and \( \sigma \) is the ReLU activation function.

#### 3.1.2 Bottleneck

The bottleneck acts as a bridge between the encoder and decoder, containing the deepest layers with the highest level of feature extraction. It consists of two 3x3 convolutions followed by ReLU activations.

#### 3.1.3 Expansive Path (Decoder)

The expansive path consists of an up-sampling of the feature map followed by a 2x2 convolution ("up-convolution") that halves the number of feature channels. This is concatenated with the correspondingly cropped feature map from the contracting path, followed by two 3x3 convolutions and ReLU activations.

#### 3.1.4 Skip Connections

Skip connections link corresponding layers in the encoder and decoder, preserving spatial information lost during down-sampling and improving gradient flow during training.

#### 3.1.5 Output Layer

The final layer uses a 1x1 convolution to map the feature maps to the desired number of classes, producing a segmentation map with the same spatial dimensions as the input image.

### 3.2 Data Collection and Preprocessing

#### 3.2.1 Data Collection

The dataset for this project was collected from various sources, ensuring comprehensive coverage of lung tumor images under different conditions.

#### 3.2.2 Data Extraction

The data was extracted and converted to a suitable format for easier manipulation and visualization.

#### 3.2.3 Data Augmentation

To enhance the diversity of the training dataset and improve model robustness, several data augmentation techniques were applied:
- **Rotation:** Introduced variability by rotating images at different angles.
- **Scaling:** Adjusted the size of the images to create a variety of perspectives.
- **Flipping:** Horizontal and vertical flips to increase data diversity.
- **Cropping:** Random cropping to simulate different zoom levels.
- **Intensity Variation:** Altered brightness and contrast to mimic different imaging conditions.

### 3.3 Mathematical Formulation

#### 3.3.1 Loss Function

The loss function combines Dice Loss and Binary Cross-Entropy (BCE) Loss to address class imbalance and improve segmentation accuracy.

**Binary Cross-Entropy Loss:**

\[ \text{BCE Loss} = -\frac{1}{N} \sum_{i=1}^N [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)] \]

where \( y_i \) is the ground truth label, \( p_i \) is the predicted probability, and \( N \) is the number of pixels.

**Dice Loss:**

\[ \text{Dice Loss} = 1 - \frac{2 \sum_{i=1}^N y_i p_i}{\sum_{i=1}^N y_i + \sum_{i=1}^N p_i} \]

**Combined Loss:**

\[ \text{Dice BCE Loss} = \alpha \cdot \text{Dice Loss} + \beta \cdot \text{BCE Loss} \]

where \( \alpha \) and \( \beta \) are weighting factors.

#### 3.3.2 Optimization

The Adam optimizer was used for its adaptive learning rate capabilities. The optimization process can be described by the following update rules:

\[ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \]  
\[ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \]  
\[ \hat{m}_t = \frac{m_t}{1 - \beta_1^t} \]  
\[ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \]  
\[ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t \]

where \( g_t \) is the gradient at time step \( t \), \( m_t \) and \( v_t \) are the first and second moment estimates, \( \beta_1 \) and \( \beta_2 \) are decay rates, \( \eta \) is the learning rate, \( \epsilon \) is a small constant, and \( \theta \) represents the model parameters.

### 3.4 Training and Evaluation

#### 3.4.1 Training Process

The training process involved preparing the data, initializing the model, defining the loss function and optimizer, and using a learning rate scheduler.

#### 3.4.2 Training Loop

The training loop iteratively trained the model, validated its performance, and adjusted the learning rate.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
loss_fn = DiceBCELoss()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')
    
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

 * images.size(0)
    
    val_loss /= len(val_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')
    
    scheduler.step(val_loss)
```

#### 3.4.3 Evaluation Metrics

- **Dice Coefficient:** Measures the overlap between the predicted and ground truth masks.
  
\[ \text{Dice Coefficient} = \frac{2 |A \cap B|}{|A| + |B|} \]

- **Precision and Recall:** Evaluate the accuracy and completeness of the segmentation.

## 4. Results and Discussion

### 4.1 Quantitative Results

The model's performance was evaluated using metrics such as Dice Coefficient, Precision, and Recall. The results demonstrate a significant improvement over traditional methods, with higher accuracy and consistency in tumor segmentation.

### 4.2 Qualitative Results

Visual inspection of the segmented images further validates the model's effectiveness. The automated approach produces clear, accurate segmentation boundaries, closely matching the ground truth annotations.

### 4.3 Comparison with Traditional Methods

The automated deep learning-based segmentation method outperforms traditional manual and semi-automated methods in terms of accuracy, efficiency, and consistency.

## 5. Conclusion

### 5.1 Summary

This project successfully developed a deep learning-based approach for automated lung tumor segmentation using the UNET model. The proposed method demonstrated significant improvements in segmentation accuracy and consistency, reducing the workload of radiologists and enhancing patient care.

### 5.2 Future Work

Future work may focus on further refining the model architecture, incorporating additional imaging modalities, and exploring real-time implementation for clinical use.
