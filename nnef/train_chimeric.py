import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch
import options
import h5py
from dataset import DatasetLocalGenCM
from model import LocalTransformer
from trainer.local_trainer import LocalGenTrainer
import pandas as pd


parser = options.get_local_gen_parser()
args = options.parse_args_and_arch(parser)

# Create Directories
root = args.save_path
if not os.path.exists(root):
    os.makedirs(root)
    os.system(f'mkdir -p {root}/models')

writer = SummaryWriter(root)

# load dataset
train_dataset = DatasetLocalGenCM(f'data/{args.data_flag}', args)

pdb_weights = pd.read_csv(f'data/{args.data_flag}')['weight'].values
datasampler = WeightedRandomSampler(weights=pdb_weights, num_samples=args.total_num_samples)
# datasampler = SequentialSampler(train_dataset)
train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=datasampler,
                               num_workers=args.num_workers)

if args.val_data_flag is not None:
    val_dataset = DatasetLocalGenCM(f'data/{args.val_data_flag}', args)
    pdb_weights = pd.read_csv(f'data/{args.val_data_flag}')['weight'].values
    val_data_sampler = WeightedRandomSampler(weights=pdb_weights, num_samples=args.batch_size)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_data_sampler,
                                 num_workers=args.num_workers, pin_memory=True)
else:
    val_data_loader = None

if args.test_data_flag is not None:
    test_dataset = DatasetLocalGenCM(f'data/{args.test_data_flag}', args)
    pdb_weights = pd.read_csv(f'data/{args.test_data_flag}')['weight'].values
    test_data_sampler = WeightedRandomSampler(weights=pdb_weights, num_samples=args.batch_size)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_data_sampler,
                                  num_workers=args.num_workers, pin_memory=True)
else:
    test_data_loader = None

#################################################
device = torch.device(args.device)

model = LocalTransformer(args)

if args.load_exp is not None:
    model.load_state_dict(torch.load(f'{args.load_exp}/models/model.pt', map_location=torch.device('cpu')))

model.to(device)

trainer = LocalGenTrainer(writer, model, device, args)

print("Training Start")
for i, epoch in enumerate(range(args.epochs)):
    trainer.train(epoch, train_data_loader)
    if (i > 0) & (i % args.save_interval == 0):
        torch.save(model.state_dict(), f"{root}/models/model_{i}.pt")
    else:
        torch.save(model.state_dict(), f"{root}/models/model.pt")
    if val_data_loader is not None:
        for j in range(10):
            trainer.test(epoch*10+j, val_data_loader, flag='Val')
    if test_data_loader is not None:
        for j in range(10):
            trainer.test(epoch*10+j, test_data_loader, flag='Test')

