# A Simple PyTorch deep learning REST API

This repository contains the code of [如何用flask部署pytorch模型](https://zhuanlan.zhihu.com/p/35879835)

## Starting the pytorch server

```bash
python run_pytorch_server.py 
```

<img src='https://ws1.sinaimg.cn/large/006tNc79gy1fqi1tz84vtj30r603emxe.jpg' width='400'>

You can now access the REST API via `http://127.0.0.1:5000/predict`

## Submitting requests to pytorch server

```bash
python simple_request.py --file='file_path'
```

<img src='https://ws3.sinaimg.cn/large/006tNc79gy1fqi206fd7qj30i803g74b.jpg' width='400'>

## Acknowledgement
This repository refers to [jrosebr1/simple-keras-rest-api](https://github.com/jrosebr1/simple-keras-rest-api), and thank the author again.