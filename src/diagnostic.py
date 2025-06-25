"""
Entry point for running the U-Net model on the pre-processed DICOM files and generate plots used in the report. If a loaded model is found, the training step is skipped.
"""
import numpy as np
import pickle

import torch

from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

from custom_loss import CombinedLoss
from data import DICOMSliceDataset
from research_processing import get_research_information
from trainer import ModelTrainer
from unet import DeeperUNet

from io import BytesIO
from PIL import Image

MODEL_PATH = "./models/unet_CombinedLoss_bs4_lr0.001.pth"
BATCH_SIZE = 3
WORKERS = 4

def diagnostic(research_dir):

    patient = os.path.basename(research_dir)
    data_dict = dict()
    segmentation_dict = dict()
    for date in os.listdir(research_dir):
        data_pickle_path = f"{research_dir}\\{date}\\data\\data_test.pkl"
        segmentation_pickle_path = f"{research_dir}\\{date}\\data\\segmentation_test.pkl"
        if os.path.exists(data_pickle_path) and os.path.exists(segmentation_pickle_path):
            with open(data_pickle_path, "rb") as file:
                data = pickle.load(file)
            with open(segmentation_pickle_path, "rb") as file:
                segmentation = pickle.load(file)
            data_dict.update(data)
            segmentation_dict.update(segmentation)
        else:
            data_list, segmentation_list = get_research_information(os.path.join(research_dir, date))
            data_dict[f"{patient}-{date}"] = data_list
            segmentation_dict[f"{patient}-{date}"] = segmentation_list
            data = dict()
            segmentation = dict()
            data[f"{patient}-{date}"] = data_list
            segmentation[f"{patient}-{date}"] = segmentation_list
            with open(data_pickle_path, 'wb') as file:
                pickle.dump(data, file)
            with open(segmentation_pickle_path, 'wb') as file:
                pickle.dump(segmentation, file)

    test_dataset = DICOMSliceDataset(data_dict, segmentation_dict)

    test_dataloader = DataLoader(
        test_dataset, batch_size=3, shuffle=False, num_workers=4
    )

    model = DeeperUNet(in_channels=1, out_channels=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Уменьшен learning rate

    loss_fn = CombinedLoss(alpha=0.2, gamma=2.0, epsilon=1e-6)  # Явно задать параме
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainer = ModelTrainer(
        test_dataset,
        model,
        loss_fn,
        optimizer,
        device,
        batch_size=4,
        num_epochs=11,
    )

    if trainer.model_exists():
        print("Trained model and corresponding losses found. Loading model...")
        trainer.load_model()

    else:
        print("Model does not exist")
        exit()

    predictions = get_predictions(
        model,
        test_dataloader,
        device
    )
    return predictions

def get_predictions(
        model,
        dataloader,
        device
):
    model.eval()
    prediction_images = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            print(f"-----{batch_idx}-----")
            images = images.to(device)
            predictions = torch.sigmoid(model(images))
            predictions = (predictions > 0.5).float()

            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()
            preds_np = predictions.cpu().numpy()

            for i in range(images.size(0)):
                print(i)
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                print(f"{i}.{0}")
                axes[0].imshow(images_np[i][0], cmap="gray")
                axes[0].set_title("Input Image")
                axes[0].axis("off")
                print(f"{i}.{1}")
                axes[1].imshow(masks_np[i][0], cmap="gray")
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")
                print(f"{i}.{2}")
                axes[2].imshow(preds_np[i][0], cmap="gray")
                axes[2].set_title("Prediction")
                axes[2].axis("off")
                print(f"{i}.{3}")
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                buf.seek(0)
                img = Image.open(buf)
                img_array = np.array(img)
                print(f"{i}.{4}")
                prediction_images.append(img_array)

    return prediction_images

