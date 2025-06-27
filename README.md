# Optimizarea sincronizării buzelor și reducerea variațiilor la mișcare pentru generarea bazată pe audio a animațiilor faciale 3D


## Setup

Pentru a clona repository-ul, folosiți următoarea comandă:

```shell
git clone https://github.com/thomasaugustinov/proiect-de-diploma.git
cd proiect-de-diploma
```

După clonarea repository-ului, instalați toate dependențele necesare utilizând fișierul requirements.txt. Utilizați următoarea comandă:

```shell
pip install -r setup/requirements.txt
```

Descărcați fișierele FLAME urmând aceste instrucțiuni:
  
  - De pe site-ul oficial [FLAME](https://flame.is.tue.mpg.de/):
    
    - Descărcați modelul FLAME 2020 și dezarhivați-l în `/models/data/FLAME2020`
    - Descărcați `landmark_embedding.npy` și `FLAME_masks.pkl` în `/models/data`.


Configurați environment-urile. Puteți face acest lucru rulând următoarele comenzi:
  ```shell
  bash ./setup/fetch_data.sh
  bash ./install_conda.sh
  ```


## Inferență

Oferim două [modele pre-antrenate](https://drive.google.com/drive/folders/1LATcnnGqhuik1UCI7qUip5A4KAbeF2EQ?usp=drive_link) pentru inferență: unul pentru encoder-ul de stil și celălalt pentru rețeaua de denoising. Extrageți aceste fișiere în folderul `/.experiments`.


Modelul primește ca intrare un clip audio, un șablon de parametri pentru forma feței și o trăsătură de stil, și produce animații stilizate și sincronizate cu audio-ul.

| Encoder-ul de stil         | Rețeaua de denoising      |
| -------------------------- | ------------------------- | 
| set\_1\_SE (@26k) | set\_1\_DPT (@110k) |


### Generarea animației bazată pe audio

```shell
python demo.py --exp_name <DENOISING_NETWORK_NAME> --iter <DENOISING_NETWORK_ITER> -a <AUDIO> -c <SHAPE_TEMPLATE> -s <STYLE_FEATURE> -o <OUTPUT>.mp4 -n <N_REPITIONS>
```

Iată un exemplu:

```shell
python demo.py --exp_name set_1_DPT --iter 110000 -a demo/input/audio/FAST.flac -c demo/input/coef/TH217.npy -s demo/input/style/TH217.npy -o TH217-FAST-TH217.mp4 -n 3
```