# PRIISMインストール手順書 @ADC

PRIISMのインストール手順をまとめます。CASAは6.6.4、 Pythonのバージョンは3.10とします。コンパイラはシステムの `gcc` (`/usr/bin/gcc`) を使います。

## PRIISM向け仮想環境の作成

PRIISMはCASAをベースにしており、CASAはPython 3.8とPython 3.10をサポートしています。新システム（2024年7月〜）の`python3`はPython 3.10ですので、以下ではPython 3.10をベースとした仮想環境を構築します。ただし、下記「CASAを仮想環境とみなす」方法の方が簡便ですので、可能であればそちらをお試しください。

```
# Python 3.10 ベースの仮想環境
python3 -m venv priism

# 仮想環境の有効化
source priism/bin/activate

# pipをアップグレードしておきます。またwheelもインストールしておきます。
python3 -m pip install --upgrade pip wheel
```

または、CASAを仮想環境とみなしてPRIISMをCASAに直接インストールすることもできます。Python 3.10版のCASAを使うようにしてください。

```
# ADCの共有エリアからCASAを丸ごとコピー（この例では CASA 6.6.4）
cp -r /usr/local/casa/casa-6.6.4-34-py3.10.el8 .

# パスを通す
# which python3などとしてCASAにパスが通っていることを確認してください。
export PATH=$PWD/casa-6.6.4-34-py3.10.el8/bin:$PATH

# pipをアップグレードしておきます。またwheelもインストールしておきます。
python3 -m pip install --upgrade pip wheel
```

## PRIISMのクローン

もしまだクローンしていない場合はクローンしてください。すでにクローン済みの場合はスキップして構いません。

```
git clone https://github.com/tnakazato/priism.git
```

必要に応じて最新版にアップデートします。

```
# クローンしたPRIISMのディレクトリに移動
cd path/to/priism
git pull
```

## PRIISMのインストール

`pip`によるインストールをサポートしていますが、動作が不安定なため現時点ではおすすめできません。`setup.py`によるインストール方法を下記に示します。仮想環境がCASAベースの場合とシステムの`python3`ベースの場合でビルドオプションが異なりますのでご注意ください。

#### システムの`python3`ベースの場合


```
# 依存パッケージのインストール
python3 -m pip install -r requirements.txt

# ビルド
# 本来Python 3.10のヘッダファイルが必要なのですが、
# 見つからなかったためPython 3.11のヘッダファイルを使う
# というトリッキーなことをやっています
python3 setup.py build --python-library=/usr/lib64/libpython3.so --python-include-dir=/usr/include/python3.11/

# インストール
python3 setup.py install
```

#### CASAベースの場合
```
# 依存パッケージのインストール
python3 -m pip install -r requirements.txt

# ビルド
python3 setup.py build

# インストール
python3 setup.py install
```

## PRIISMの動作確認

下記のどちらかのJupyter Notebookを実行します。PRIISMリポジトリに同梱さています。

* TW Hya imaging tutorial: `priism/tutorial_twhya.ipynb`
* HL Tau imaging tutorial: `priism/tutorial_hltau.ipynb`

```
# PRIISM仮想環境にJupyter Notebookをインストール
python3 -m pip install jupyter

# ノートブックの中でAstropyを使うのでインストール
python3 -m pip install astropy

# 作業ディレクトリを作成（パスは適宜読み替えてください）
mkdir ~/work/HLTau-notebook
cd ~/work/HLTau-notebook

# Jupyterを起動
# 以後はJupyter Notebook上でセルを実行していただきます
jupyter-notebook
```

## TODO

この手順書の今後のアップデート項目です。もし要望があれば takeshi.nakazato@nao.ac.jp までお願いします。
