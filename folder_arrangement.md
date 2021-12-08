## Arrange image folders to fit the needs of PyTorch Dataloader

## 1. Copy to dizhen folder

```bash
cp -r /home/dsh/AM205_Final_Project/data/manipulated_sequences/DeepFakeDetection/c40/images /home/dsh/dizhen/data/images
mv images fake

cp -r /home/dsh/AM205_Final_Project/data/original_sequences/actors/c40/images /home/dsh/dizhen/data/images
mv images real

```

## 2. Collect folder names

```bash
cd /home/dsh/dizhen/data/fake
for d in *; do
    echo "$d" >> fakedir.txt
done

cd /home/dsh/dizhen/data/real
for d in *; do
    echo "$d" >> realdir.txt
done
```

## 3. Rename each png

Dealing with Issues: same png names in different folders.

Split data into train (4/5) and val (1/5).

```python
import os

with open('fakedir.txt') as f:
    folders = f.readlines()
i = 0
for f in folders:
    
    foldername = f.replace('\n','')
    print(foldername)
    path = f'/home/dsh/dizhen/data/fake/{foldername}'
    for filename in os.listdir(path):
        r = i%5
        if r != 0:
            os.rename(os.path.join(path,filename), os.path.join(path,foldername+'_'+ filename.replace(".png","") + "_tr.png"))
        else:
            os.rename(os.path.join(path,filename), os.path.join(path,foldername+'_'+ filename.replace(".png","") + "_ts.png"))
        i += 1
import os

with open('realdir.txt') as f:
    folders = f.readlines()
i = 0
for f in folders:
    foldername = f.replace('\n','')
    print(foldername)
    path = f'/home/dsh/dizhen/data/real/{foldername}'
    for filename in os.listdir(path):
        r = i%5
        if r != 0:
            os.rename(os.path.join(path,filename), os.path.join(path,foldername+'_'+ filename.replace(".png","") + "_tr.png"))
        else:
            os.rename(os.path.join(path,filename), os.path.join(path,foldername+'_'+ filename.replace(".png","") + "_ts.png"))
        i += 1
```

## 4. Move png into data1 folders

'data1' folder contains 'train' and 'val' folders, 'train' contains 'fake' and 'real' folders, 'val' contains 'fake' and 'real' folders.

This is for the convenience of 'datasets.ImageFolder' for loading data.

```bash
cd /home/dsh/dizhen/data/fake/
find . -name '*_tr.png' -exec mv {} /home/dsh/dizhen/data1/train/fake \;

cd /home/dsh/dizhen/data/fake/
find . -name '*_ts.png' -exec mv {} /home/dsh/dizhen/data1/val/fake \;

cd /home/dsh/dizhen/data/real/
find . -name '*_tr.png' -exec mv {} /home/dsh/dizhen/data1/train/real \;

cd /home/dsh/dizhen/data/real/
find . -name '*_ts.png' -exec mv {} /home/dsh/dizhen/data1/val/real \;
```
