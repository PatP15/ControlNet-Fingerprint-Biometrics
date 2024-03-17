import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from dataloader import FingerDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Configs
resume_path = '../models/control_sd15_enhanced_finger.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# Prepare Dataset
full_dataset = FingerDataset("./enhanced_prompts.json")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('../models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Prepare DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# Logger
logger = ImageLogger(batch_frequency=logger_freq)

checkpoint_callback = ModelCheckpoint(
    monitor='val/loss',  # Monitor validation loss for checkpointing
    dirpath='./checkpoints/',  # Directory where checkpoints will be saved
    filename='model-{epoch:02d}-{val_loss:.2f}',  # Checkpoint file naming convention
    save_top_k=1,  # Save only the top k models according to monitored metric
    mode='min',  # Minimize the monitored metric (val_loss in this case)
    save_last=True,  # Optionally, save a checkpoint for the last epoch in addition to the best one
)

# early_stop_callback = EarlyStopping(
#     monitor='val/loss',  # Metric to monitor
#     min_delta=0.00,  # Minimum change in the monitored metric to qualify as an improvement
#     patience=2,  # Number of epochs with no improvement after which training will be stopped
#     verbose=True,
#     mode='min'  # Minimize the monitored metric (val_loss in this case)
# )

# Trainer with checkpoint callback
trainer = pl.Trainer(
    gpus=-1,  # Use all available GPUs
    precision=32,
    callbacks=[logger, checkpoint_callback],  # Include the checkpoint callback here
    strategy='ddp',  # Distributed Data Parallel
    max_epochs=10,  # Set the number of epochs here
)

# Continue with training as before
trainer.fit(model, train_dataloader, val_dataloader)
