# AnomalyDetection

````sh
필요 라이브러리

pytorch_lightning
mlflow
pandas
timm
opencv-python
albumentations
````

### 실행: python train.py --config config.yaml

학습결과는 lightning_logs 폴더에 버전별로 기록됨(ckpt, config, hparams)

test.py: 제출 파일 생성하는 코드
