import torch
import torchvision
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from torchmetrics.classification import BinaryJaccardIndex

from hw.task3.dataset import SelfieDataset


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_single_loader(img_dir,
                      mask_dir,
                      batch_size,
                      transform,
                      num_workers=8,
                      pin_memory=True,):
    ds = SelfieDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        transform=transform,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    return loader


def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        test_dir,
        test_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=8,
        pin_memory=True,
):
    train_ds = SelfieDataset(
        img_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = SelfieDataset(
        img_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    test_ds = SelfieDataset(
        img_dir=test_dir,
        mask_dir=test_maskdir,
        transform=val_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    jaccard_index = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
            )
            jaccard = BinaryJaccardIndex().to(device)
            jaccard_index += jaccard(preds, y)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
    )
    print(f"Dice score: {dice_score / len(loader)}")
    print(f"Jaccard Index score: {jaccard_index / len(loader)}")
    model.train()

    return float(num_correct / num_pixels * 100), float(dice_score / len(loader)), float(jaccard_index / len(loader))


def save_predictions_as_imgs(
        loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
