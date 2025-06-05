import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from classifier_model.cnn_2d_model.CNN_LSTM_dataset import HelplessnessVideoDataset as Dataset_2D
from classifier_model.cnn_2d_model.CNN_LSTM_model import HelplessnessClassifier as CNN_2D_Model
from classifier_model.dataset import HelplessnessVideoDataset as Dataset_3D
from classifier_model.cnn_3d_model.model import HelplessnessClassifier as CNN_3D_Model

def validate(model, val_loader, device):
    """Run provided classifier model on validation dataset"""
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            _, preds = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    return true_labels, pred_labels

def load_model(model_class, weights_path, device):
    """Load 2D CNN + LSTM or 3D CNN Model"""
    model = model_class()
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model

def load_transformer_model(weights_path, device):
    """Load SwinTransformer Model"""
    model = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
    model.head = nn.Linear(model.head.in_features, 3)
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model

def generate_cm(true_labels, pred_labels):
    cm = sk_confusion_matrix(true_labels, pred_labels)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['None', 'Little', 'Extreme'])
    cm_display.plot(cmap='Blues')
    plt.title("Confusion Matrix for Helplessness Dataset")
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.show()

    print(f"Precision score: {precision_score(true_labels, pred_labels, average='weighted'):.2f}")
    print(f"Recall score: {recall_score(true_labels, pred_labels, average='weighted'):.2f}")
    print(f"F1 score: {f1_score(true_labels, pred_labels, average='weighted'):.2f}")
    print(f"Accuracy score: {accuracy_score(true_labels, pred_labels):.2f}")

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Define transformations and data loaders
    val_transform_2d = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform_3d = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.41500069, 0.36530493, 0.33830512],
                             [0.29042152, 0.27499218, 0.27738131])
    ])

    val_loader_3d = DataLoader(
        Dataset_3D("data/val", transform=val_transform_3d),
        batch_size=1, shuffle=True, num_workers=4
    )
    val_loader_2d = DataLoader(
        Dataset_2D("data/val", transform=val_transform_2d),
        batch_size=1, shuffle=False, num_workers=0
    )

    # Load models
    model_2d = load_model(
        CNN_2D_Model,
        'classifier_model/cnn_2d_model/grayscale_cnn_lstm.pth',
        device
    )
    model_3d = load_model(
        CNN_3D_Model,
        'classifier_model/cnn_3d_model/model_weights.pth',
        device
    )
    model_transformer = load_transformer_model(
        'classifier_model/pre_trained_transformer_model/model_weights.pth',
        device
    )

    # Create confusion matrix for each model
    true_labels_2d, preds_2d = validate(model_2d, val_loader_2d, device)
    true_labels_3d, preds_3d = validate(model_3d, val_loader_3d, device)
    true_labels_transformer, preds_transformer = validate(model_transformer, val_loader_3d, device)

    generate_cm(true_labels_2d, preds_2d)
    generate_cm(true_labels_3d, preds_3d)
    generate_cm(true_labels_transformer, preds_transformer)


if __name__ == "__main__":
    main()
