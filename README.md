# DACON-CV02

[Dacon 예술 작품 화가 분류 AI 경진대회](https://dacon.io/competitions/official/236006/overview/description)

> model 폴더
> .tar 파일 저장
> 폴더 이름 : {The number of epochs}epoch{modelname}.tar

> data_analysis 폴더
> 데이터 분석 plot 저장

### Data Download
 - ./data의 내부는 git 제어에서 무시됩니다.
 - Original Link : https://drive.google.com/file/d/1SZ5j8SJTCqQQuNHTMDHXSOMShKUvB479/view?usp=sharing
 - wget
    >wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={1SZ5j8SJTCqQQuNHTMDHXSOMShKUvB479}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={1SZ5j8SJTCqQQuNHTMDHXSOMShKUvB479}" -O {open.zip} && rm -rf ~/cookies.txt

