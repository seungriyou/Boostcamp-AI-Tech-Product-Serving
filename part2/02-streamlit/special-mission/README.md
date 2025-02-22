# Special Mission 1: Streamlit으로 프로토타입 만들기
> 유승리_T3129

## 실행 방법
1. 가상 환경 생성 후 패키지를 설치합니다.
    ```bash
    conda create -n open-mmlab python=3.10 -y
    source activate open-mmlab

    conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch

    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html

    git clone https://github.com/open-mmlab/mmsegmentation.git
    cd mmsegmentation
    pip install -e .  # or "python setup.py develop"
    ```
2. `mmcv`의 일부 코드를 수정합니다.
    ```bash
    vim /opt/conda/envs/open-mmlab/lib/python3.10/site-packages/mmcv/utils/registry.py
    
    # line 253-254, 268-270 주석 처리
    ```
3. 그 외 streamlit 등의 라이브러리를 설치합니다. (자세한 내용은 `requirements.txt` 참고)

## 주의 사항
- 제출하는 파일은 `assets`나 `mmsegmentation` 내용을 제외한 `app.py`, `README.md`, `requirements.txt`, `streamlit_semantic_segmentation_유승리.mp4` 입니다.
- 실제 실행에 필요한 절차가 많기 때문에 `streamlit_semantic_segmentation_유승리.mp4` 데모 영상을 봐주시면 됩니다.

## 스크린샷
<img width="1624" alt="2" src="https://user-images.githubusercontent.com/43572543/169632998-4228c1bf-706b-4d24-8a3b-7f48f7f7f11b.png">
<img width="1624" alt="4" src="https://user-images.githubusercontent.com/43572543/169633009-6a401796-b4d6-4a20-9a42-77ea5d60271f.png">
<img width="1624" alt="5" src="https://user-images.githubusercontent.com/43572543/169633012-9380c87f-ea02-48a8-a50e-61ea80be7e8d.png">
<img width="1624" alt="6" src="https://user-images.githubusercontent.com/43572543/169633015-544e3480-d551-414c-a972-6c8b99c1e4ae.png">

## 디렉토리 구조
```
|-- README.md
|-- app.py
|-- assets
|   |-- 0511-upernet_swin_l_full_pl.py
|   |-- class_dict.csv
|   `-- epoch_46.pth
|-- mmsegmentation
|   |-- CITATION.cff
|   |-- LICENSE
|   |-- MANIFEST.in
|   |-- README.md
|   |-- README_zh-CN.md
|   |-- app.py
|   |-- configs
|   |-- demo
|   |-- docker
|   |-- docs
|   |-- mmseg
|   |-- mmsegmentation.egg-info
|   |-- model-index.yml
|   |-- model.py
|   |-- predict.py
|   |-- pytest.ini
|   |-- requirements
|   |-- requirements.txt
|   |-- resources
|   |-- setup.cfg
|   |-- setup.py
|   |-- tests
|   |-- tools
|   `-- utils.py
`-- requirements.txt
```
