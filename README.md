## 서울대학교 빅데이터 핀테크 6기 - 딥러닝과 머신러닝 파이널 프로젝트 
# DDSP-SVC를 이용한 음성 변환


## 0. 들어가며
DDSP-SVC는 새로운 오픈 소스 노래 음성 변환 프로젝트이며 개인 컴퓨터에서 대중화할 수 있는 무료 AI 음성 변환 소프트웨어 개발에 전념합니다.

이 프로젝트보다 더 유명한 Diff-SVC와 SO-VITS-SVC랑 비교해보았을때, 훈련과 합성 과정에서 조금 더 낮은 컴퓨터 사양에서도 동작하고 훈련 시간도 몇 배 더 단축 할 수 있습니다. 또한 실시간으로 음성을 변경 할 때 SO-VITS-SVC에서 요구하는 하드웨어 자원보다 요구 기준이 낮으며 Diff-SVC의 경우 실시간 음성 변경이 너무 느립니다.

본래 DDSP 의 합성 품질이 그닥 이상적이지는 않지만 (훈련 중에 TensorBoard에서 본래의 출력을 들을 수 있음) 사전훈련된 보코더 기반 enhancer를 사용하여 몇몇 데이터셋이 SO-VITS-SVC의 음질과 비슷한 수준으로 도달 할 수 있습니다.

학습 데이터의 품질이 매우 높은 경우에도 Diff-SVC의 음질이 가장 좋을 수 있습니다. 데모 출력 결과는 samples 폴더에 존재하며 관련된 체크포인트 모델들은 Releases 페이지에서 다운로드 하실 수 있습니다.

경고: DDSP-SVC를 통해 학습시키는 모델이 합법적으로 허가된 데이터로 학습되도록 해주시고 불법적인 방식으로 음성을 합성하여 사용하는 일이 없도록 해주세요. 본 저장소의 소유자는 모델 체크포인트와 오디오 이용한 권리 침해, 사기 및 기타 불법 행위에 대해 책임을 지지 않습니다.

## 1. 주제 선정 배경
We recommend first installing PyTorch from the [**official website**](https://pytorch.org/), then run:
```bash
pip install -r requirements.txt 
```
NOTE : I only test the code using python 3.8 (windows) + torch 1.9.1 + torchaudio 0.6.0, too new or too old dependencies may not work

UPDATE: python 3.8 (windows) + cuda 11.8 + torch 2.0.0 + torchaudio 2.0.1 works, and training is faster.

## 2. 이론적 배경
UPDATE:  ContentVec encoder is supported now. You can download the pretrained [ContentVec](https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr) encoder instead of HubertSoft encoder and modify the configuration file to use it.
- **(Required)** Download the pretrained [**HubertSoft**](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)   encoder and put it under `pretrain/hubert` folder.
-  Get the pretrained vocoder-based enhancer from the [DiffSinger Community Vocoders Project](https://openvpi.github.io/vocoders) and unzip it into `pretrain/` folder
## 3. 모델 구성 및 학습 방법

Put all the training dataset (.wav format audio clips) in the below directory:
`data/train/audio`. 
Put all the validation dataset (.wav format audio clips) in the below directory:
`data/val/audio`.
You can also run
```bash
python draw.py
```
to help you select validation data (you can adjust the parameters in `draw.py` to modify the number of extracted files and other parameters)

Then run
```bash
python preprocess.py -c configs/combsub.yaml
```
for a model of combtooth substractive synthesiser (**recommend**), or run
```bash
python preprocess.py -c configs/sins.yaml
```
for a model of sinusoids additive synthesiser.

You can modify the configuration file `config/<model_name>.yaml` before preprocessing. The default configuration is suitable for training 44.1khz high sampling rate synthesiser with GTX-1660 graphics card.

NOTE 1: Please keep the sampling rate of all audio clips consistent with the sampling rate in the yaml configuration file ! If it is not consistent, the program can be executed safely, but the resampling during the training process will be very slow.

NOTE 2: The total number of the audio clips for training dataset is recommended to be about 1000,  especially long audio clip can be cut into short segments, which will speed up the training, but the duration of all audio clips should not be less than 2 seconds. If there are too many audio clips, you need a large internal-memory or set the 'cache_all_data' option to false in the configuration file.

NOTE 3: The total number of the audio clips for validation dataset is recommended to be about 10, please don't put too many or it will be very slow to do the validation.

NOTE 4:  If your dataset is not very high quality, set 'f0_extractor' to 'crepe' in the config file.  The crepe algorithm has the best noise immunity, but at the cost of greatly increasing the time required for data preprocessing.

UPDATE: Multi-speaker training is supported now. The 'n_spk' parameter in configuration file controls whether it is a multi-speaker model.  If you want to train a **multi-speaker** model, audio folders need to be named with **positive integers not greater than 'n_spk'** to represent speaker ids, the directory structure is like below:
```bash
# training dataset
# the 1st speaker
data/train/audio/1/aaa.wav
data/train/audio/1/bbb.wav
...
# the 2nd speaker
data/train/audio/2/ccc.wav
data/train/audio/2/ddd.wav
...

# validation dataset
# the 1st speaker
data/val/audio/1/eee.wav
data/val/audio/1/fff.wav
...
# the 2nd speaker
data/val/audio/2/ggg.wav
data/val/audio/2/hhh.wav
...
```
If 'n_spk'  = 1, The directory structure of the **single speaker** model is still supported, which is like below:
```bash
# training dataset
data/train/audio/aaa.wav
data/train/audio/bbb.wav
...
# validation dataset
data/val/audio/ccc.wav
data/val/audio/ddd.wav
...
```
## 4. 결론
```bash
# train a combsub model as an example
python train.py -c configs/combsub.yaml
```
The command line for training other models is similar.

You can safely interrupt training, then running the same command line will resume training.

You can also finetune the model if you interrupt training first, then re-preprocess the new dataset or change the training parameters (batchsize, lr etc.) and then run the same command line.

## 5. Visualization
```bash
# check the training status using tensorboard
tensorboard --logdir=exp
```
Test audio samples will be visible in TensorBoard after the first validation.

NOTE: The test audio samples in Tensorboard are the original outputs of your DDSP-SVC model that is not enhanced by an enhancer. If you want to test the synthetic effect after using the enhancer  (which may have higher quality) , please use the method described in the following chapter.
## 6. Non-real-time VC
(**Recommend**) Enhance the output using the pretrained vocoder-based enhancer:
```bash
# high audio quality in the normal vocal range if enhancer_adaptive_key = 0 (default)
# set enhancer_adaptive_key > 0 to adapt the enhancer to a higher vocal range
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -id <speaker_id> -eak <enhancer_adaptive_key (semitones)>
```
Raw output of DDSP:
```bash
# fast, but relatively low audio quality (like you hear in tensorboard)
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -id <speaker_id> -e false
```
Other options about the f0 extractor and response threhold，see:
```bash
python main.py -h
```
(UPDATE) Mix-speaker is supported now. You can use "-mix" option to design your own vocal timbre, below is an example:
```bash
# Mix the timbre of 1st and 2nd speaker in a 0.5 to 0.5 ratio
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -mix "{1:0.5, 2:0.5}" -eak 0
```
## 7. Real-time VC
Start a simple GUI with the following command:
```bash
python gui.py
```
The front-end uses technologies such as sliding window, cross-fading, SOLA-based splicing and contextual semantic reference, which can achieve sound quality close to non-real-time synthesis with low latency and resource occupation.

Update: A splicing algorithm based on a phase vocoder is now added, but in most cases the SOLA algorithm already has high enough splicing sound quality, so it is turned off by default. If you are pursuing extreme low-latency real-time sound quality, you can consider turning it on and tuning the parameters carefully, and there is a possibility that the sound quality will be higher. However, a large number of tests have found that if the cross-fade time is longer than 0.1 seconds, the phase vocoder will cause a significant degradation in sound quality.
## 8. Acknowledgement
* [ddsp](https://github.com/magenta/ddsp)
* [pc-ddsp](https://github.com/yxlllc/pc-ddsp)
* [soft-vc](https://github.com/bshall/soft-vc)
* [DiffSinger (OpenVPI version)](https://github.com/openvpi/DiffSinger)
