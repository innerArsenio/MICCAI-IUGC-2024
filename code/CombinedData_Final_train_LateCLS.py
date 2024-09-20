import logging
import os
import sys
from tqdm import tqdm
import itertools
import torch
import torch.nn.functional as F
import monai
from monai.data import DataLoader
from monai.metrics import DiceMetric, ROCAUCMetric
import numpy as np
from model import U_Net
# !!! combined augmentations with helper
from utils.helper_functions import *
from torch.utils.data import WeightedRandomSampler
import schedulefree
import argparse
import warnings


# Mute specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="monai.transforms.intensity.array")
warnings.filterwarnings("ignore", category=FutureWarning, module="monai.losses.dice")
warnings.filterwarnings("ignore", category=FutureWarning)


# Command-line argument parsing
parser = argparse.ArgumentParser(description="Experiment Configuration")
parser.add_argument("--torchamp", type=str, default="True", choices=["True", "False"],
                    help="Enable or disable torch AMP (Automatic Mixed Precision)")
parser.add_argument("--learning_rate", type=float, default=0.005, help="Set the learning rate for the optimizer")

args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set the seed
set_seed(42)

exp_name ='CombinedData_FinalTrain_LateCLS'
torchamp = args.torchamp == "True"

# ========================== set hyper parameters =========================#
BATCH_SIZE = 20
NUM_EPOCHS = 100
BASE_LEARNING_RATE = 0.005
LEARNING_RATE = args.learning_rate
WEAK_SUPERVISION_WEIGHT = 0.9
NUM_WORKERS = 2
# !!! should I remove alpha?
ALPHA = 0.6
# !!! removed weight decay
START_BEST = 999

# =========================================================================#

exp = f'oneoptim'
checkpoint_dir = f"./checkpoints_{exp_name}"
os.makedirs(checkpoint_dir, exist_ok=True)

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"training_{exp_name}.log")),
        logging.StreamHandler(sys.stdout)
    ]
)


hyperparameters = dict(
batch_size=BATCH_SIZE,
num_epochs=NUM_EPOCHS,
base_lr=BASE_LEARNING_RATE,
lr=LEARNING_RATE,
weak_supervision_weight=WEAK_SUPERVISION_WEIGHT,
alpha = ALPHA
)


def xavier_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

torch.cuda.empty_cache()
random_empty_mask_dir = '/home/arsen.abzhanov/salem/FrameDataset_pt/Neg_Masks/20190726T095643_0_1.pt'
train_data = get_data_dict_2task_pt(data_dir='/home/arsen.abzhanov/salem/FrameDataset_pt',random_empty_mask_dir=random_empty_mask_dir)


#Testing Code : Uncomment to get small batch #TODO:
#train_data = train_data[0:432]

print(len(train_data))
sample_weights = get_sample_weights(train_data)
# # Initialize the WeightedRandomSampler with the calculated weights
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

val_dir = '/home/arsen.abzhanov/salem/FrameDataset_pt_val'
val_data = get_val_dict_2task_pt(data_dir=val_dir, random_empty_mask_dir=random_empty_mask_dir)

train_transforms, val_transforms = transformations()
train_transforms.set_random_state(seed=42)

train_ds = monai.data.Dataset(data=train_data, transform=train_transforms) # bottle neck
train_loader = DataLoader(
    train_ds,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    batch_size = BATCH_SIZE,
    
)

val_ds = monai.data.Dataset(data=val_data, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size= BATCH_SIZE, num_workers=NUM_WORKERS, shuffle = False, pin_memory=torch.cuda.is_available())

dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False, num_classes=3)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = U_Net(mode='inference').to(device)
model.apply(xavier_init)

weights = torch.tensor([0.017, 0.924, 0.059]).to(device) # SET IT 

seg_loss_function = monai.losses.DiceFocalLoss(
    include_background=True,
    to_onehot_y=True,
    softmax=True,
    smooth_nr=1e-05,
    smooth_dr=1e-05,
)

optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=LEARNING_RATE, warmup_steps=5000 , weight_decay=0.1)

cls_loss_fn = torch.nn.BCEWithLogitsLoss()

val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
epoch_cls_loss_values=[]
epoch_seg_loss_values=[]
cls_metric_values = []
seg_metric_values = []
CLS_TRAIN=False
scaler = torch.amp.GradScaler("cuda")
start_epoch, best_metric, best_metric_epoch, CLS_TRAIN, START_BEST= load_checkpoint(os.path.join(checkpoint_dir, "last_checkpoint.pth"), model, optimizer,scaler)

roc_auc_metric = ROCAUCMetric()
train_dice_score = 0
for epoch in tqdm(range(start_epoch, NUM_EPOCHS)):
    set_seed(42)
    logging.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    model.train()
    optimizer.train()
    cls_prob_list = []
    cls_label_list = []    
    epoch_loss = 0
    epoch_cls_loss=0
    epoch_seg_loss=0
    step = 0
    cls_correct = 0
    cls_total = 0
    dice_metric.reset()
    
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False, num_classes=3)
    for batch_data in train_loader:

        optimizer.zero_grad()
        set_seed(42)
        cls_inputs= batch_data['image'].to(device)
        cls_labels = batch_data["class"].to(device).unsqueeze(1).float()
        labels =batch_data['label'].to(device)
        mask_type = batch_data['type']
        # Initialize cls_loss to zero
        cls_loss = torch.tensor(0.0, device=device)
        with torch.amp.autocast("cuda", enabled=torchamp):
            outputs_cls, outputs_seg = model(cls_inputs, mask_type=mask_type, mode="inference")
            if epoch + 1 >= 10 and train_dice_score >= 0.96 and not CLS_TRAIN:
                    CLS_TRAIN = True
                    START_BEST = epoch + 2 
            if CLS_TRAIN:
                cls_loss = cls_loss_fn(outputs_cls, cls_labels)
                # Check if classification loss is NaN
                if torch.isnan(cls_loss):
                    logging.warning(f"NaN encountered in classification loss. Skipping this batch.")
                    optimizer.zero_grad()  # Zero out gradients
                    scaler.update()  # Still update the scaler to maintain correct state
                    torch.cuda.empty_cache()  # Clear GPU memory
                    continue
                
                # Classification accuracy calculation 
                cls_probs = torch.sigmoid(outputs_cls)
                cls_preds = (cls_probs >= 0.5).float()
                cls_correct += (cls_preds == cls_labels).sum().item()
                cls_total += cls_labels.size(0)

                cls_prob_list.append(cls_probs)
                cls_label_list.append(cls_labels.detach().cpu())

            #Define losses for segmentation (2 losses one for originally labeled masks and a weakly supervised loss for synthetic masks)
            loss_labeled = torch.tensor(0.0, requires_grad=True).to(device)
            loss_unlabeled = torch.tensor(0.0, requires_grad=True).to(device)

            labeled_mask = [t == 'labeled' for t in mask_type]
            unlabeled_mask = [t == 'unlabeled' for t in mask_type]
            seg_output_1 = outputs_seg[labeled_mask]
            seg_output_2= outputs_seg[unlabeled_mask]

            if seg_output_1.size(0) > 0:
                loss_labeled = seg_loss_function(seg_output_1,  labels[labeled_mask])
                # Check if segmentation loss for labeled is NaN
                if torch.isnan(loss_labeled):
                    logging.warning(f"NaN encountered in labeled segmentation loss. Skipping this batch.")
                    optimizer.zero_grad()  # Zero out gradients
                    scaler.update()  # Still update the scaler to maintain correct state
                    torch.cuda.empty_cache()  # Clear GPU memory
                    continue
                pos_labeled_labels_one_hot = F.one_hot(labels[labeled_mask].squeeze(1).long(), num_classes=3).permute(0, 3, 1, 2).float()
                seg_preds_labeled = F.softmax(seg_output_1, dim=1).argmax(dim=1, keepdim=True)
                seg_preds_labeled_one_hot = F.one_hot(seg_preds_labeled.squeeze(1), num_classes=3).permute(0, 3, 1, 2).float()
                dice_metric(y_pred=seg_preds_labeled_one_hot, y=pos_labeled_labels_one_hot)
            if seg_output_2.size(0) > 0:
                loss_unlabeled = WEAK_SUPERVISION_WEIGHT *seg_loss_function(seg_output_2, labels[unlabeled_mask])
                # Check if segmentation loss for unlabeled is NaN
                if torch.isnan(loss_unlabeled):
                    logging.warning(f"NaN encountered in unlabeled segmentation loss. Skipping this batch.")
                    optimizer.zero_grad()  # Zero out gradients
                    scaler.update()  # Still update the scaler to maintain correct state
                    torch.cuda.empty_cache()  # Clear GPU memory
                    continue
                pos_unlabeled_labels_one_hot = F.one_hot(labels[unlabeled_mask].squeeze(1).long(), num_classes=3).permute(0, 3, 1, 2).float()
                seg_preds_unlabeled = F.softmax(seg_output_2, dim=1).argmax(dim=1, keepdim=True)
                seg_preds_unlabeled_one_hot = F.one_hot(seg_preds_unlabeled.squeeze(1), num_classes=3).permute(0, 3, 1, 2).float()
                dice_metric(y_pred=seg_preds_unlabeled_one_hot, y=pos_unlabeled_labels_one_hot)

        # Calculate the relative magnitudes of the losses
        # cls_loss = (config.alpha * cls_loss)
        # seg_loss = (1- config.alpha) * (loss_labeled+ loss_unlabeled)
        

        if CLS_TRAIN:    
            seg_loss = loss_labeled + loss_unlabeled
            total_loss_magnitude = cls_loss + seg_loss
            cls_weight = cls_loss / total_loss_magnitude
            seg_weight = seg_loss / total_loss_magnitude
            total_loss = (cls_weight.detach() * cls_loss) + (seg_weight.detach() * seg_loss)
        else: 
            seg_loss =loss_labeled + loss_unlabeled
            total_loss = seg_loss
            
        scaler.scale(total_loss).backward()

        # Log gradient norms
        total_norm = 0
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.empty_cache()
        step += 1
        epoch_loss += total_loss.item()
        epoch_cls_loss += cls_loss.item()
        epoch_seg_loss += seg_loss.item()
        
        if (step % 500) == 0:
            logging.info(f"{step}/{len(train_loader)}, train_loss: {total_loss.item():.4f}")


    epoch_loss /= step
    epoch_cls_loss /= step
    epoch_seg_loss /= step

    epoch_loss_values.append(epoch_loss)
    epoch_cls_loss_values.append(epoch_cls_loss)
    epoch_seg_loss_values.append(epoch_seg_loss)
    logging.info(f"Epoch {epoch + 1} average total loss: {epoch_loss:.4f} avg. cls loss:{epoch_cls_loss:.4f} avg. seg loss:{epoch_seg_loss:.4f}")

    if cls_total > 0:
        train_cls_accuracy = cls_correct / cls_total
    else:
        train_cls_accuracy = 0.0  # Or handle as you prefer
    # Calculate training Dice score
    train_dice_score = dice_metric.aggregate().item()
    if cls_prob_list and cls_label_list:
    # AUC calculation for training
        cls_prob_list = torch.cat(cls_prob_list).cpu()
        cls_label_list = torch.cat(cls_label_list).cpu()
        auc_train = roc_auc_metric(y_pred=cls_prob_list, y=cls_label_list)
        auc_train = roc_auc_metric.aggregate()
    else:
        auc_train = 0.0
    # log training accuract and dice score
    logging.info(f"Epoch {epoch + 1} train classification accuracy: {train_cls_accuracy:.4f} | train Dice score: {train_dice_score:.4f},train AUC: {auc_train:.4f}")

    optimizer.eval()
    model = model.to(device)  # Move model to GPU if available

    with torch.no_grad():
        for batch in itertools.islice(train_loader, 50):
            input_tensor = batch['image'].to(device)  # Move input tensor to the same device
            model(input_tensor, mode='forward')
        
    if (epoch + 1) % val_interval == 0:
        dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=False, num_classes=3)
        model.eval()
        dice_metric.reset()
        roc_auc_metric.reset()
        cls_prob_list = []
        cls_label_list = []
        cls_correct = 0
        cls_total = 0
        tp = 0
        fp = 0
        fn = 0
        class_1_dice_scores = []
        class_2_dice_scores = []
        with torch.no_grad():
            for val_data in val_loader:
                
                val_images, val_labels = val_data["image"].to(device), val_data["label"]
                val_cls_labels = val_data["class"].to(device).unsqueeze(1)

                cls_outputs = model(val_images, mode="cls")
                cls_probs = torch.sigmoid(cls_outputs)
                # Collect classification probabilities and labels for AUC calculation
                cls_prob_list.append(cls_probs.detach().cpu())
                cls_label_list.append(val_cls_labels.detach().cpu())

                cls_preds = (cls_probs >= 0.5).float()
                cls_correct += (cls_preds == val_cls_labels).sum().item()
                cls_total += val_cls_labels.size(0)

                tp += ((cls_preds == 1) & (val_cls_labels == 1)).sum().item()
                fp += ((cls_preds == 1) & (val_cls_labels == 0)).sum().item()
                fn += ((cls_preds == 0) & (val_cls_labels == 1)).sum().item()


                val_types = val_data['type']
                pos_labeled_idx = [i for i, t in enumerate(val_types) if t == 'labeled']
                            
                if pos_labeled_idx:
                    val_images_seg = val_images[pos_labeled_idx]
                    val_labels_seg = val_labels[pos_labeled_idx]

                    val_labels_seg = val_labels_seg.to(device)
                    val_labels_seg = F.one_hot(val_labels_seg.squeeze(1).long(), num_classes=3).permute(0, 3, 1, 2).float()  # [B,3,512,512]

                    val_outputs = model(val_images_seg, mode="seg")
                    val_preds = F.softmax(val_outputs, dim=1).argmax(dim=1, keepdim=True)  # [B,1,512,512]
                    val_preds_one_hot = F.one_hot(val_preds.squeeze(1), num_classes=3).permute(0, 3, 1, 2).float()  # [B,3,512,512]
                    dice_scores = dice_metric(y_pred=val_preds_one_hot, y=val_labels_seg)
                    class_1_dice_scores.extend(dice_scores[:, 0].tolist())  # Class 1 Dice
                    class_2_dice_scores.extend(dice_scores[:, 1].tolist())  # Class 2 Dice
            # Calculate the mean and standard deviation for class 1 and class 2 Dice scores
            class_1_mean_dice = np.mean(class_1_dice_scores)
            class_1_std_dice = np.std(class_1_dice_scores)
            class_2_mean_dice = np.mean(class_2_dice_scores)
            class_2_std_dice = np.std(class_2_dice_scores)
            # Log these statistics

            # Calculate and log the overall mean Dice and standard deviation
            all_dice_scores = class_1_dice_scores + class_2_dice_scores
            mean_dice = np.mean(all_dice_scores)
            std_dice = np.std(all_dice_scores)

            # Reset the lists for the next epoch
            class_1_dice_scores.clear()
            class_2_dice_scores.clear()
            dice_metric.reset()

            cls_accuracy = cls_correct / cls_total
            f1 , recall, precision = f1_recall_precision(tp,fp,fn)
            

        # AUC calculation for validation
        cls_prob_list = torch.cat(cls_prob_list).cpu()
        cls_label_list = torch.cat(cls_label_list).cpu()
        auc_val = roc_auc_metric(y_pred=cls_prob_list, y=cls_label_list)
        auc_val= roc_auc_metric.aggregate()

        combined_metric = 0.3 * (auc_val) + 0.7 * mean_dice
        if combined_metric > best_metric and (epoch + 1) > START_BEST:
            best_metric = combined_metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_metric_model_2Task.pth"))
            logging.info("Saved new best metric model")


        logging.info(f"Epoch {epoch + 1}, Dice_All: {mean_dice:.4f}Std_dev:{std_dice}, Dice_PS:{class_1_mean_dice} Std_dev:{class_1_std_dice}, Dice_FH:{class_2_mean_dice} Std_dev:{class_2_std_dice}, Best Combined Metric: {best_metric:.4f} at epoch {best_metric_epoch}")
        logging.info(f"Epoch {epoch + 1}, validation AUC: {auc_val:.4f}, classification accuracy: {cls_accuracy:.4f}, f1 score:{f1:.4f}, recall:{recall:.4f}, precision:{precision:.4f}")


    save_checkpoint({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 
        'best_metric': best_metric,  # Save best metric
        'scaler_state_dict': scaler.state_dict(),  # Save the scaler state
        'best_metric_epoch': best_metric_epoch,  # Save best metric epoch
        'Combined Metric': combined_metric,
        'CLS_TRAIN': CLS_TRAIN,
        'START_BEST': START_BEST
    }, os.path.join(checkpoint_dir, "last_checkpoint.pth"))


logging.info(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
