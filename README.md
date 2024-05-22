# 人体キーポイントを用いて映像に整合する楽器音を生成するモデルの提案

- 2023年度 電気通信大学 卒業研究発表会 [卒論](https://github.com/harukana1435/Senior-Project/files/15397429/2010136_.pdf)
- DEIM2024 第16回 データ工学と情報マネジメントに関するフォーラム　[[学会ページ](https://confit.atlas.jp/guide/event/deim2024/subject/T4-A-2-04/tables?cryptoId=)]

## 目次
- [概要](#概要)
- [生成結果](#生成結果)
- [人体キーポイントとは](#人体キーポイントとは)
- [モデルの全体構造](#モデルの全体構造)
- [データセット](#データセット)
- [リポジトリの説明](#リポジトリの説明)
    
## 概要 
　楽器演奏の動画は音が重要な要素であるため，撮影中に騒音が入った動画や音質が劣化した昔の動画は音声の改善が求められる．その際，ノイズ除去技術は有効な手法だが，極めて劣悪な音声はノイズ除去による音声の修復が難しい．そこで，動画の視覚情報のみから演奏者が弾いている楽器音を予測して生成することを研究目的とした．先行研究([SpecVQGAN](https://v-iashin.github.io/SpecVQGAN))はコードブック表現やTransformerによって質の高い音声を生成できるが，楽器演奏という複雑な動作の映像に対しては予測精度が低いという現状がある．本研究の提案モデルは既存モデルに人体キーポイントを導入することで，楽器音の生成精度を向上させることを目指す．アンケートによる主観評価と4つの尺度を用いた客観評価を行った結果，実際の音声と生成した音声の類似性の観点で，提案モデルが既存モデルを上回った．

## 生成結果
### Cello
|本物の楽器音|生成した楽器音|
|:-:|:-:|
|<video src="https://github.com/harukana1435/Senior-Project/assets/167507629/dcecbbd6-d91f-44d0-b116-770b429c2cb0">|<video src="https://github.com/harukana1435/Senior-Project/assets/167507629/4c30bfbf-eafa-4c04-b07a-49832d5ac613">|

### Horn
|本物の楽器音|生成した楽器音|
|:-:|:-:|
|<video src="https://github.com/harukana1435/Senior-Project/assets/167507629/4861598e-08d6-4023-9a7a-a1d37150f216">|<video src="https://github.com/harukana1435/Senior-Project/assets/167507629/793bca10-46fd-421e-a89e-b1dbbfd614a5">|



## 人体キーポイントとは
人体の骨格を疑似的に表した特徴点のことであり，人間の身体の部位や関節の位置をピンポイントで示している．本研究の人体キーポイントは，右図のように47個の特徴点から構成されている．本研究では音を生成するにあたって，人体キーポイントデータから演奏者の動きを表す特徴量，すなわち人体キーポイント特徴を抽出する工程がある．
<div align="center" style="line-height: 0;">
  <img src="https://github.com/harukana1435/Senior-Project/assets/167507629/6a337c71-1e3d-4c85-bc22-46cc2f641883" alt="Image 1" width="277" style="vertical-align: middle;"/>
  <img src="https://github.com/harukana1435/Senior-Project/assets/167507629/a0d2be7e-ceb3-4755-9b08-347e7cc7aeff" alt="Image 2" width="300" style="vertical-align: middle;"/>
  <img src="https://github.com/harukana1435/Senior-Project/assets/167507629/60b974a7-afb4-4624-89b6-d15bc9eb008a" alt="Image 3" width="300" style="vertical-align: middle;"/>
</div>

## モデルの全体構造
本研究のモデルは[SpecVQGAN](https://v-iashin.github.io/SpecVQGAN)をベースとして開発した．[SpecVQGAN](https://v-iashin.github.io/SpecVQGAN)は全般的な音を対象として映像に整合する音を生成した研究であり，生成するときに用いる映像特徴として，色情報を示す**RGB特徴**と物体の動きを示す**オプティカルフロー特徴**を導入していた．一方，提案モデルは楽器演奏という複雑な動きを捉えるために，新たに**人体キーポイント特徴**を処理するモジュールを追加した．人体キーポイント特徴を計算する際には，人体キーポイントを事前学習済みの[ST-GCN](https://arxiv.org/abs/1801.07455)(**※1**)で処理した．
<div align="center" style="line-height: 0;">
  <img src="https://github.com/harukana1435/Senior-Project/assets/167507629/7c7c3a80-4d69-4d56-8701-22e97fa2fe58" alt="Image 1" width="800" style="vertical-align: middle;"/>
</div>

#### ※1
[Spatial Temporal Graph Convolutional Networks(ST-GCN)](https://v-iashin.github.io/SpecVQGAN)とは人体キーポイントから空間的及び時間的パターンを学習するモデルのことであり，人間の動作認識タスクで用いられている．人体の骨格をグラフとして扱っており，人間の関節や体の部位がノードに対応している．

## データセット
本研究の提案モデルを学習させるためには，楽器演奏の映像と音声のペアが必要である．そこで，楽器演奏の映像に加えて人体キーポイントデータを提供している[Solos](https://juanmontesinos.com/Solos/)データセットを使用した．データセット内から以下の6つの楽器を生成対象として選択した．
<div align="center" style="line-height: 0;">
  <img src="https://github.com/harukana1435/Senior-Project/assets/167507629/49e9379f-c393-4552-9304-f5651c910eb3" alt="Image 1" width="600" style="vertical-align: middle;"/>
</div>

## リポジトリの説明
### configs
モデルを学習させる際のコンフィグファイルが格納されている．
### data
モデルに入力する動画の特徴量が格納されている．
### keypoint_feature
人体キーポイントデータを処理するプログラムやST-GCNのプログラムが格納されている．  
`extract_solos_keypoint.py`，`generate_stgcn_data.py`:人体キーポイントデータからST-GCNの学習データを作成する．  
`stgcn_data.py`，`stgcn_model.py`，`stgcn_train_test_generate.py`:ST-GCNの学習を行い，人体キーポイント特徴を計算する．  
### logs
提案モデルの学習結果が格納されている．
### preprocess
動画の前処理を行うプログラムが格納されている．  
`extract_solos_keypoint.py`:solosデータセットに含まれている楽器演奏の動画をyoutubeからダウンロードする．  
`convert_solos_25fps.py`:動画を25fpsに変換する．  
`generate_10s_video.py`:動画を10秒間隔で切り分ける．  
`extract_25fps_audio.py`:25fpsの動画から音声のみを抽出する．  
`tempo_change.py`:音声のテンポを変更する．  
`concatenate_video_and_audio.py`:音声と動画を結合する．  
`generate_solos_json.py`:切り分けた10秒の動画に対する情報をjsonに書き出す．  
`generate_train_valid_txt.py`:データセット内の動画を訓練用とテスト用に分割する．  
### solos_data
前処理の過程で作成されたデータが格納されている．
### specvqgan
既存モデルのソースコードが格納されている．
### vocoder
生成したメルスペクトログラムを音声波形に変換するボコーダのプログラムが格納されている．
### conda_env.yml
anacondaの仮想環境を構築するプログラム．以下のコマンドで実行．  
`conda env create -f conda_env.yml`
### train.py
コードブック及びTransformerを訓練するプログラム．以下のコマンドで実行． 
#### コードブック
`python train.py --base configs/solos_codebook.yaml -t True --gpus 0,`
#### Transformer
`python train.py --base configs/solos_transformer.yaml -t True --gpus 0, \
    model.params.first_stage_config.params.ckpt_path=./logs/solos_codebook_52/checkpoints/last.ckpt`

### generate_samples.py
学習したモデルを用いてテストデータを対象に楽器音を生成するプログラム．以下のコマンドで実行．  
`python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=62374 \
    --use_env \
        evaluation/generate_samples.py \
        sampler.config_sampler=evaluation/configs/sampler.yaml \
        data.params.spec_dir_path="./data/solos/features/*/melspec_10s_22050hz/" \
        data.params.rgb_feats_dir_path="./data/solos/features/*/feature_rgb_bninception_dim1024_21.5fps/" \
        data.params.flow_feats_dir_path="./data/solos/features/*/feature_flow_bninception_dim1024_21.5fps/" \     data.params.keypoint_feats_dir_path="./data/solos/features/*/feature_keypoint_dim64_21.5fps/" \
        sampler.now=`date +"%Y-%m-%dT%H-%M-%S"``


