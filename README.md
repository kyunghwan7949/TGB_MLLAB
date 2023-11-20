## TGB baseline 코드
Private Repository 이므로 git clone 전 ssh key 발급 후 코드 받아주세요!


## 도커 이미지 및 가상환경
- 72/233 서버는 세팅완료
```bash
docker pull kyunghwan7949/mllab_tgb:latest
docker run -it --gpus '"device=2"' -v /media:/data -v /home/:/mount -v --ipc=host --name tgb_2 docker.io/kyunghwan7949/tgb:latest /bin/bash
conda activate tgb
```

## Train
```bash
python train_tgb_lpp.py \
--dataset_name "tgbl-flight" \
--model_name "DyGFormer" \
--batch_size "32" \
```
