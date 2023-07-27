# Diffusion Learning
学習の例は`execute_example.sh`及び`multiprocess_example.sh`を参考．
ログを見たい場合`tensorboard --logdir=runs`などを実行(デフォルトのポートは6006)．

## ディレクトリの設計について
ディレクトリについてシェルスクリプトによって実験のパラメタを確実に保存し，
実験の性質ごとにディレクトリを大分類として分けることを推奨する．
また，Tensorboardではlogdirの配下のパラメタについて和集合を取って表示するため，ディレクトリの最下層は
データの種別に分離すると見やすくなるだろう．

## コードの実行と計算資源
データの種類，数，ステップ数によってかかる時間のボトルネックが異なり，GPUの効率のよい使い方が変わるので適宜調整すること．

特に`--sample_multi_process`について，これは推論時にGPUが余っているが，ステップ数が大きく，改善の余地があるときに使用される．
並列処理を行うときに異なるエポックのモデルをマルチプロセスで処理するために，モデルの保存及びプロセスの分岐のためのレイテンシが発生する．
当然GPUやCPUの排他制御によりうまく行かない部分があるかもしれない．なお，このとき，使用しなくても良いが`nvidia-cuda-mps-control`を使用すると
GPUの並列処理がそれを使用しないマルチプロセスよりもより効率良くなる．なお，最後にまとめてサンプルをしたほうが効率がより良くなるが，学習中にも確認したいという受容に対応するためそのようには実装していない．

## コードの変更
diffusion.pyそのものを変更するのは，argparserを変更する以外はおすすめしない．過去の実験がどのような条件で行われたかという情報が失われる可能性があるため．
モデルやデータの変更はファイルに追加することで実装する．


## 謝辞
本プロジェクトは[https://github.com/cloneofsimo/minDiffusion](minDiffusion)を大いに参考にして開発されました．
