import torchvision.transforms as T
from torch.utils.data import Dataset,dataloader
import numpy as np
import cv2
import torch
from utils.utils import seed_worker
import os
import random
from utils.augment import random_perspective,augment_hsv

PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders
IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation

class LoadImages(Dataset):
    def __init__(self,files,labels,hyp,img_size=256,batch_size=8,augment=False):
        self.img_size=img_size
        self.augment=augment
        self.batch_size=batch_size
        self.files=files
        self.labels=labels
        self.preprocess=transforms()
        assert len(self.files) == len(self.labels),"Number of Images and Labels do not match"
#         self.files=list((x.replace('/',os.sep) for x in self.files))
        self.files=list(self.files)
        n=len(self.files)
        
        bi=np.floor(np.arange(n)/self.batch_size).astype(int)#batch index
        nb=bi[-1]+1 #number of batches
        
        self.indices=np.arange(n)
        self.augment=augment
        self.hyp=hyp
    
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self,index):
        index=self.indices[index]
        img,(h0,w0),(h,w)=self.load_image(index)
        labels=self.labels[index]
#         labels=torch.from_numpy(labels)
        return img,labels
#         img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#         img = np.ascontiguousarray(img)
        
#         return torch.from_numpy(img), labels
        
    def load_image(self,i):
        f=self.files[i]
        im=cv2.imread(str(f))
        assert im is not None, f"Image Not Found {f}"
        h0, w0 = im.shape[:2]  # orig hw
        if self.augment:
            im=random_perspective( 
                    im,
                    degrees=self.hyp["degrees"],
                    translate=self.hyp["translate"],
                    scale=self.hyp["scale"],
                    shear=self.hyp["shear"])
            augment_hsv(im, hgain=self.hyp["hsv_h"], sgain=self.hyp["hsv_s"], vgain=self.hyp["hsv_v"])
            if random.random() < self.hyp["flipud"]:
                im = np.flipud(im)
            if random.random() < self.hyp["fliplr"]:
                im = np.fliplr(im)
            
        r = h0 != self.img_size or w0 != self.img_size  # ratio
        if r:  # if sizes are not equal
            interp = cv2.INTER_AREA
            im=cv2.resize(im,(self.img_size,self.img_size),interpolation=interp)
        sample=self.preprocess(im)
        
        return sample, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
def transforms():
    sample=T.Compose([ToTensor(),T.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
    return sample
        

class ToTensor:
    # YOLOv5 ToTensor class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, half=False):
        """Initializes ToTensor for YOLOv5 image preprocessing, with optional half precision (half=True for FP16)."""
        super().__init__()

    def __call__(self, im):
        """
        Converts BGR np.array image from HWC to RGB CHW format, and normalizes to [0, 1], with support for FP16 if
        `half=True`.

        im = np.array HWC in BGR order
        """
#         print(im.shape)
#         print(im)
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im=im.float()
        im /= 255.0  # 0-255 to 0.0-1.0
        return im
    
    
class InfiniteDataloader(dataloader.DataLoader):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()
    def __len__(self):
        """Returns the length of the batch sampler's sampler in the InfiniteDataLoader."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Yields batches of data indefinitely in a loop by resetting the sampler when exhausted."""
        for _ in range(len(self)):
            yield next(self.iterator)
class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        """Initializes a perpetual sampler wrapping a provided `Sampler` instance for endless data iteration."""
        self.sampler = sampler

    def __iter__(self):
        """Returns an infinite iterator over the dataset by repeatedly yielding from the given sampler."""
        while True:
            yield from iter(self.sampler)
    
def CreateDataloader(imgs,lbls,hyp,batch_size,shuffle,workers,seed,augment):
    
    dataset=LoadImages(imgs,lbls,hyp,batch_size=batch_size,augment=augment)
    batch_size=min(batch_size,len(dataset))
    nd=torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None
    loader=InfiniteDataloader
    generator=torch.Generator()
    generator.manual_seed(seed)
    return loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
#         collate_fn=LoadImages.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    ), dataset