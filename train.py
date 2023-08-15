import torch
import logging 
from tqdm import tqdm
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_without_label(data, model, optimizer, device):
    full_loss = []
    model.train()
    for batch_data in tqdm(data):
        h = batch_data[0].to(device)
        t = batch_data[1].to(device)
        r = batch_data[2].to(device)


        optimizer.zero_grad()
        loss, _ = model(t, r, h, True)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        l2 = loss.item()

        optimizer.zero_grad()
        loss, _ = model(h, r, t)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        l1 = loss.item()
        full_loss.append((l1+l2) / 2)
    return full_loss