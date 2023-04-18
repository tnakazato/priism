# PRIISMインストール手順書 @ADC

PRIISMのインストール手順をまとめます。Pythonのバージョンは3.8とします。コンパイラはインテルコンパイラ`icpc`か、または`/usr/local/gcc`以下にインストールされたバージョンの新しい`gcc`のいずれかが利用可能です。なお、`icpc`を使う場合でも、sparseimaging以外のc++コードは常にgccを使ってコンパイルします。

## PRIISM向け仮想環境の作成

PRIISM専用の仮想環境を作ります。仮想環境のディレクトリは`~/pyenv/priism`としていますが、適宜読み替えてください。

```
# PRIISM用の仮想環境を作ります
python3.8 -m venv ~/pyenv/priism

# 仮想環境を有効化
source ~/pyenv/priism/bin/activate

# pipをアップグレードしておきます。またwheelもインストールしておきます。
python3 -m pip install --upgrade pip wheel
```

仮想環境を新規に作成する代わりに、CASAを仮想環境とみなしてPRIISMをCASAに直接インストールすることもできます。Python 3.8版のCASAを使うようにしてください。

```
# ADCの共有エリアからCASAを丸ごとコピー（この例では CASA 6.5.2）
cp -r /usr/local/casa/casa-6.5.2-26-py3.8 .

# パスを通す
# which python3などとしてCASAにパスが通っていることを確認してください。
export PATH=$PWD/casa-6.5.2-26-py3.8/bin:$PATH

# pipをアップグレードしておきます。またwheelもインストールしておきます。
python3 -m pip install --upgrade pip wheel
```

## インテルコンパイラのための設定

2023年1月以降、ADCでインテルコンパイラを使うためには`LD_LIBRARY_PATH`の設定が必要になりました。詳しくは[ADCのFAQ](https://www.adc.nao.ac.jp/cgi-bin/cfw/wiki.cgi/FAQ/FAQJ?page=Intel+C%2B%2B+Compiler+%27icc%27+%A4%CE%BB%C8%CD%D1%A4%CB%A4%C4%A4%A4%A4%C6+%282023%C7%AF1%B7%EE20%B0%CA%B9%DF%29)を参照してください。
以下はbashの場合の設定例です。

```
# bash系の場合
export LD_LIBRARY_PATH=/usr/local/gcc/12.2/lib64:$LD_LIBRARY_PATH
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

```
# 依存パッケージのインストール
python3 -m pip install -r requirements.txt

# ビルド: インテルコンパイラを使う場合はこちらを実行
python3 setup.py build --use-intel-compiler=yes

# ビルド: gccを使う場合はこちらを実行（gccのバージョンが変わっていたら適宜変更してください）
PATH=/usr/local/gcc/12.2/bin:$PATH python3 setup.py build 

# インストール
python3 setup.py install
```

## PRIISMの動作確認

下記のどちらかのJupyter Notebookを実行します。

* TWHya imaging tutorial: PRIISM同梱 `priism/cvrun.ipynb`
* HL Tau imaging demo (Gist): https://gist.github.com/tnakazato/be0888d153eef2a76a3c260d794bf052

```
# PRIISM仮想環境にJupyter Notebookをインストール
python3 -m pip install jupyter

# ノートブックの中でAstropyを使うのでインストール
python3 -m pip install astropy

# 作業ディレクトリを作成（パスは適宜読み替えてください）
mkdir ~/work/HLTau-notebook
cd ~/work/HLTau-notebook

# 上記ノートブックのいずれかをダウンロード
# ここではHL Tau Imaging demoを使います
wget https://gist.githubusercontent.com/tnakazato/be0888d153eef2a76a3c260d794bf052/raw/ab4f72efa0026f85caaaafdc0939a4ea6602cc1f/HLTau_demo.ipynb

# Jupyterを起動
# 以後はJupyter Notebook上でセルを実行していただきます
jupyter-notebook
```

## TODO

この手順書の今後のアップデート項目です。もし要望があれば takeshi.nakazato@nao.ac.jp までお願いします。
