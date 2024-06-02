from torch.utils.data import Dataset,DataLoader
import numpy as np
import cv2
import os
import torch
import random

PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders


class LoadImages(Dataset):
    def __init__(self,files,labels,img_size=256,batch_size=8,augment=False):
        self.img_size=img_size
        self.augment=augment
        self.batch_size=batch_size
        self.files=files
        self.labels=labels
        assert len(self.files) == len(self.labels),"Number of Images and Labels do not match"
        n=len(files)
#         self.files=list((x.replace('/',os.sep) for x in self.files))
        self.files=list(self.files)
        
        bi=np.floor(np.arange(n)/self.batch_size).astype(int)#batch index
        nb=bi[-1]+1 #number of batches
        
        self.indices=np.arange(n)
    
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self,index):
        index=self.indices[index]
        img,(h0,w0),(h,w)=self.load_image(index)
        labels=self.labels[index]
#         labels=torch.from_numpy(labels)
        
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        return torch.from_numpy(img), labels#, self.files[index]
        
    def load_image(self,i):
        f=self.files[i]
        im=cv2.imread(str(f))
        assert im is not None, f"Image Not Found {f}"
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
            im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    
#     @staticmethod
#     def collate_fn(batch):
#         """Batches images, labels, paths, and shapes, assigning unique indices to targets in merged label tensor."""
#         im, label, path = zip(*batch)  # transposed
#         for i, lb in enumerate(label):
#             lb[:, 0] = i  # add target image index for build_targets()
#         return torch.stack(im, 0), torch.cat(label, 0), path



def seed_worker(worker_id):
    """
    Sets the seed for a dataloader worker to ensure reproducibility, based on PyTorch's randomness notes.

    See https://pytorch.org/docs/stable/notes/randomness.html#dataloader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def CreateDataloader(imgs,lbls,batch_size,shuffle,workers,seed):
    
    dataset=LoadImages(imgs,lbls,batch_size=batch_size)
    batch_size=min(batch_size,len(dataset))
    print(batch_size)
    nd=torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler=None
    loader=DataLoader
    generator=torch.Generator()
    generator.manual_seed(seed)
    return loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
#         collate_fn=LoadImagesAndLabels.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    ), dataset
    

