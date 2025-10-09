# Qwen2.5-Omni-7B 多模態 AI 服務

## 專案概述
本專案使用 Qwen2.5-Omni-7B 模型提供多模態 AI 服務，支援：
- 音訊轉錄 (Audio Transcription)
- 音訊對話 (Audio Chat)
- 文字、圖像、音訊、視訊的多模態輸入處理

## 要安裝的套件，請務必增加在requirements.txt中。
## 環境變數使用
因為本機無法執行，主要就是有env.example的範例，到伺服器端會在docker run指令中直接指定.env檔，在此情境下，請用最單純的load_dotenv()讀取.env的內容即可

若在.env檔有設定以下的環境變數，如：
GOOGLE_API_KEY=你的_API_Key
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

並於要使用的程式中使用以下範例程式

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
database_url = os.getenv("DATABASE_URL")

## 使用 Logging 機制
logging模組的初始化已經於app.py中做了，若要修改log level與format，請到.env中修改。

在新程式中使用 logging 機制的作法如下：
1. 在程式的開頭引入 `logging` 模組。
2. 建立 logger 物件，並在程式中使用該 logger 進行日誌記錄。

以下是範例程式碼：

```python
import logging

# 建立 logger 物件
logger = logging.getLogger(__name__)

# 使用 logger 進行日誌記錄
logger.info("這是一條 INFO 級別的日誌")
logger.debug("這是一條 DEBUG 級別的日誌")
logger.error("這是一條 ERROR 級別的日誌")
```

## 外部資料存取
若在開發時，程式中有要存取的資料，請在根目錄建立一個data資料夾(有設定排除，不會同步到git上)，目前有設
docker run指令中可設定存取此外部資料夾。若開發與部署時不想一直改路徑，則建議可以使用以下較簡單的方法：
import os
data_path = os.getenv('DATA_PATH', './data/')
file_path = os.path.join(data_path, 'example_file.txt')

若要比較複雜的作法，則也可考慮在開發環境也建立環境變數或.env檔案，並使用python-dotenv讀取：
.env檔內容
DATA_PATH=../data/

from dotenv import load_dotenv
load_dotenv()

## 編輯完程式後照以下步驟更新docker container：
1. 從Github同步最新的程式碼
因為已經有先儲存我的github帳號，所以不用重複輸入帳號密碼，但密碼現在是personal api key。若未來有要調整，則需刪掉~/.git-credentials檔案，然後上github的個人帳號重新生成一個。

若是初次建立
cd ~/marco_files/models
git clone https://github.com/marcochang1028/qwen2.5-omni-7b.git
進到模型中的docker資料夾，執行以下指令，將此兩檔變成執行檔
chmod +x docker_run.sh
git add docker_run.sh
git commit -m "設定成執行檔"
git push origin main
最後再回到vs code sync修改後的檔案
新增src底下.env檔

之後則
cd ~/marco_files/models/qwen2.5-omni-7b
git pull origin main

1. 用以下指令停掉container
docker stop qwen25-asr 
2. 進到docker目錄後執行以下構建image的指令
docker build -t pgi-asr-2 -f Dockerfile ..
3. 執行以下指令確認image成功建立
docker images
4. 執行以下指令刪除舊的image
docker rmi [image id]
5. 將docker_run.sh複製到要跑container的資料夾
6. 用env.example複製到同一個資料夾中的config/.env，並看是否有需要修改參數
7. 執行以下指令建立container
./docker_run.sh
