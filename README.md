# Dacon 예술 작품 화가 분류 AI 경진대회

[Dacon 예술 작품 화가 분류 AI 경진대회 링크](https://dacon.io/competitions/official/236006/overview/description)

public 7등 / 215 팀   
private 7등 / 215 팀  

---
## 개요

이번 월간 데이콘 26은 예술 작품을 화가 별로 분류하는 대회입니다. 
더 나아가 예술 작품의 일부분만을 가지고도 올바르게 분류해낼 수 있어야합니다.
예술 작품의 일부분만 주어지는 테스트 데이터셋에 대해 올바르게 화가를 분류해낼 수 있는 예술 작품의 전문가인 AI 모델 제작을 목표로 합니다.

## 전략

### EDA : 화가 클래스 수 불균형
<p align=center>
<img width="700" height="500" src="https://user-images.githubusercontent.com/77565951/213864983-338d0203-ab6a-4763-b3ed-7d3e7e776a14.png" />
</p>
undersampling하기에는 데이터 수가 너무 적다고 판단하여 oversampling을 사용했습니다.<br>
데이터를 stratify option을 통해 나누고 가장 작은 클래스 개수(train: 15, validation: 6)만큼 각 클래스에서 15번 random sampling 진행하였습니다.
(re-sampling 전략을 택했지만 re-weighted 전략을 쓰면 더 좋은 결과가 나왔을 것 같습니다.)

### EDA : image rgb distribution & image size
아래 그래프는 화가별로 이미지 사이즈를 구한 분포입니다.
<p align=center>
<img width="300" height="500" src="https://user-images.githubusercontent.com/77565951/213866047-b558a7ca-ef0b-466d-a584-9df624d96b5d.png"/>
<img width="300" height="500" src="https://user-images.githubusercontent.com/77565951/213866102-89799691-694b-4459-a2f6-d04d0ac59c1f.png"/>
<p/>
화가별로 특정한 이미지 사이즈를 사용하는 경우가 있습니다. 여기서 h, w값을 가져와서 추가 mlp층을 거치고 기존 backbone layer의 마지막 층에 concat해줍니다.

아래 그래프는 그림의 화가의 각 그림의 평균rgb에 대한 분포입니다.
<p align=center>
<img width="230" height="180" src="https://user-images.githubusercontent.com/77565951/213866986-ebafff72-a045-4055-bcad-9516f245aa85.png"/>
<img width="230" height="180" src="https://user-images.githubusercontent.com/77565951/213866990-fbc4145c-0932-43e7-9a10-5551fd82d659.png"/>
<img width="230" height="180" src="https://user-images.githubusercontent.com/77565951/213866995-f7001b2d-e1aa-4932-ad70-20b4c2357c37.png"/>
</p>
각 화가마다 여러색의 이미지가 존재하지만 화가마다 선호하는 색이 있을것이라 생각했습니다. 실제로 색에 대한 분포가 각 화가마다 어느정도 다르고 관계가 있을것이라 생각해 size랑 같은 방법으로 mlp계층을 거치고 backbone layer의 마지막 층에 concat합니다.


### Model
- ResNext
- EfficientNetb4
- MaxViT
- SwinT
- RegNet

### Data Augmentation
train, test그림 확인시 현실에서 아무렇게나 찍은 사진이 아닌 디지털로 변환하여 정형화된(같은 그림이면 이미지 rgb값이 전부 같은 데이터 의미)데이터로 판단하여 색상을 바꾸거나 이미지 정보를 심각하게 훼손하는 augmentation으로 사용하지 않았습니다.
- VerticalFlip
- HorizontalFlip
- Resize(224, 384) : 어떤 size로 pretrained되었는지 확인하여 그에 맞게 사용
- CutMix
- test데이터가 원본 이미지의 1/4로 잘려져서 나와있기 때문에 훈련시에도 이미지를 1/4로 잘라서 사용
