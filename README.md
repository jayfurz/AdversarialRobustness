# Adversarial Robustness in CNNs via TRADES, PAT, and RoBal

This repository contains a Jupyter Notebook that explores adversarial robustness across different Convolutional Neural Network (CNN) architectures. Specifically, it examines the implementation of three adversarial training methods: TRADES, Perceptual Adversarial Training (PAT), and RoBal.

## How to Run the Notebook

To run the experiments contained in the Jupyter Notebook:

1. You can run the notebook ```AdversarialNetworks.ipynb``` in Google Colab without worrying about dependencies and skip to step 5, otherwise go to step 2.
2. Ensure that you have Jupyter installed and the dependencies. If not, install it using pip:
   ```sh
   pip install notebook torch torchvision
3. Start the Jupyter Notebook server:
    ```
    jupyter notebook
4. Navigate to the notebook ```AdversarialNetworks.ipynb``` within the Jupyter Notebook GUI in your web browser.
5. Execute the cells in the notebook in order to reproduce the experiments. You do not need to run the CIFAR-100 or ImageNet for this experiment, whih is noted in the notebook.

## Notebook Content Description

- __Code Cells__: The notebook's code cells feature the construction and training of the CNN models, along with the adversarial attacks employed to test their robustness.
- __Comments and Markdown Text__: In-depth commentary and descriptions are provided throughout to explain the nuances of the methods and the expected outcomes of each cell.

## Datasets

- __CIFAR-10__: This dataset was used for the experiments and is directly accessible through PyTorch's torchvision module. The dataset contains 60,000 32x32 color images distributed across 10 classes.
## Models and Implementation Details

- __SimpleCNN__: A baseline CNN model for comparative purposes.
- __TRADESCNN__: An extension of SimpleCNN incorporating the TRADES adversarial training method to create a balance between accuracy and robustness.
- __PATCNN__: A variant of SimpleCNN integrating perceptual adversarial training using VGG16 features to enhance robustness based on perceptual similarity.
- __RoBalCNN__: Not fully implemented due to constraints, but intended to consider class imbalances for robustness.
## Original Features and Modifications

- The notebook is an original compilation that utilizes common deep learning practices, drawing inspiration from prevalent adversarial robustness techniques.
- Adaptations include the integration of pre-trained VGG16 features into the PATCNN model, along with custom loss functions and training procedures specific to TRADES and RoBal methodologies.
## External References and Acknowledgments

- The TRADES algorithm and its implementation are based on the work found in Zhang et al., 2019.
- The PAT method was adapted from methods demonstrated in Laidlaw and Feizi, 2020.
- Ideas behind the RoBal framework are based on works by Wang et al., 2021.

These references ensure transparency regarding the origins of methods and code.

## Contact

If you encounter any issues or have questions about running the notebook, please reach out to Justin Fursov (jfursov@purdue.edu).
