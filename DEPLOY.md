# Cloud Run 部署手冊

這個專案的正式部署方式就是跑：`./scripts/deploy_cloud_run.sh`。

同一條指令可用在：

- 第一次部署
- 你修改程式後再次部署
- 想要清掉舊 revision / 舊 image 的重部署

它現在會自動完成：**檢查 env、確認 GCP API、送出 Cloud Build、在雲端跑測試、建 image、部署到同一個 Cloud Run service、smoke check、清理舊 revision / image**。

---

## 第一次設定

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

之後就可以直接用：

```bash
./scripts/deploy_cloud_run.sh
```

如果你沒有先 `gcloud config set project`，也可以這樣：

```bash
./scripts/deploy_cloud_run.sh --project-id YOUR_PROJECT_ID
```

---

## 部署前要準備什麼

先建立 `.env`：

```bash
cp .env.example .env
```

至少要有：

- `LINE_CHANNEL_SECRET`
- `LINE_CHANNEL_ACCESS_TOKEN`
- `OPENROUTER_API_KEY`
- `GCP_PROJECT_ID`

建議同時確認這幾個部署欄位：

- `CLOUD_RUN_SERVICE_NAME`
- `CLOUD_RUN_REGION`
- `DEPLOY_KEEP_REVISIONS`
- `DEPLOY_KEEP_IMAGES`
- `DEPLOY_ENABLE_APIS`

如果你有開：

- `SCHEDULED_MESSAGES_ENABLED=true`

那要確認：

- `LINE_PUSH_FALLBACK_ENABLED=true`
- `SCHEDULED_GROUP_ID`
- `SCHEDULED_WEEKLY_MESSAGES` 或 `SCHEDULED_YEARLY_MESSAGES` 至少有一個有內容

而 `LINE_PUSH_MONTHLY_LIMIT`：

- 設成 `0` = 排程 direct push 不設上限
- 設成 `>0` = 幫 direct push 加每月上限

部署腳本會自動處理這兩件事：

- `min instances`：只有啟用排程且有實際 job 時才設為 `1`，其他情況 `0`
- runtime service account：沿用目前 Cloud Run 設定；第一次部署則使用預設 service account

---

## 標準部署

最推薦、最簡單的方式：

```bash
./scripts/deploy_cloud_run.sh
```

如果你想明確指定專案、區域、service：

```bash
./scripts/deploy_cloud_run.sh \
  --project-id YOUR_PROJECT_ID \
  --service-name linebot-cloud-agent \
  --region us-west1
```

## 修改程式後再次部署

你改完程式後，**還是跑同一條**：

```bash
./scripts/deploy_cloud_run.sh
```

這就是這個專案現在整理好的「一鍵重部署」方式。

預設行為：

- 自動產生唯一 image tag
- 部署到同一個 Cloud Run service
- `min-instances` 會依排程需求自動決定（也可手動覆寫）
- 將 100% 流量切到最新 revision
- 在 Cloud Build 內跑 `pytest` 和 `compileall`
- 部署後跑 smoke check
- 預設只保留最新 **1 個 revision**
- 預設保留最新 **3 個 image digest**

也就是說，**舊的 revision 不會一直堆著**，舊的 docker image 也不會一直累積。

---

## 如果你想更乾淨

如果你連舊 image 也只想留 1 個：

```bash
./scripts/deploy_cloud_run.sh --keep-revisions 1 --keep-images 1
```

這會更乾淨，但回滾空間也更少。

---

## 快速部署選項

如果你只是小改動，想加快速度：

### 跳過雲端測試

```bash
./scripts/deploy_cloud_run.sh --skip-tests
```

### 跳過 smoke check

```bash
./scripts/deploy_cloud_run.sh --skip-smoke
```

### 不重新 enable API

```bash
./scripts/deploy_cloud_run.sh --skip-enable-apis
```

一般情況不建議同時跳過測試和 smoke，除非你很確定這次只是極小改動。

---

## 這支腳本實際做了什麼

1. 讀 `.env`
2. 驗證必要 env key 是否存在
3. 自動確保 `cloudbuild.googleapis.com`、`run.googleapis.com`、`artifactregistry.googleapis.com`、`iamcredentials.googleapis.com`、`cloudresourcemanager.googleapis.com` 已啟用
4. 產生暫時的 Cloud Run env-vars YAML
5. 送出 `cloudbuild.yaml`
6. Cloud Build 內執行測試與語法檢查
7. 建置並推送新 image
8. 部署到 Cloud Run
9. 若有設定 `GCS_BUCKET_NAME`，部署流程會自動把 Cloud Run runtime service account 補上 `roles/iam.serviceAccountTokenCreator`，並檢查 / 補上 bucket 的 3 天 lifecycle 保底刪除規則
10. 對 `/health` 和 `/webhook` 做 smoke check
11. 清理舊 revision / image

---

## 部署完成後確認

先載入目前部署設定：

```bash
PROJECT_ID="$(python scripts/envfile.py get --file .env GCP_PROJECT_ID)"
SERVICE_NAME="$(python scripts/envfile.py get --file .env CLOUD_RUN_SERVICE_NAME)"
REGION="$(python scripts/envfile.py get --file .env CLOUD_RUN_REGION)"
```

### 取得 service URL

```bash
gcloud run services describe "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format='value(status.url)'
```

### 取得 webhook URL

```bash
echo "$(gcloud run services describe "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format='value(status.url)')/webhook"
```

### 查看 health

```bash
SERVICE_URL="$(gcloud run services describe "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format='value(status.url)')"

curl -fsS "$SERVICE_URL/health" | python -m json.tool
```

### 查看部署後目前設定的排程 push 訊息內容

```bash
gcloud run services describe "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format=json | python -c 'import json,sys; svc=json.load(sys.stdin); env={e["name"]: e.get("value","") for e in svc["spec"]["template"]["spec"]["containers"][0].get("env", [])}; out={"SCHEDULED_MESSAGES_ENABLED": env.get("SCHEDULED_MESSAGES_ENABLED"), "SCHEDULED_GROUP_ID": env.get("SCHEDULED_GROUP_ID"), "SCHEDULED_WEEKLY_MESSAGES": json.loads(env.get("SCHEDULED_WEEKLY_MESSAGES", "[]") or "[]"), "SCHEDULED_YEARLY_MESSAGES": json.loads(env.get("SCHEDULED_YEARLY_MESSAGES", "[]") or "[]")}; print(json.dumps(out, ensure_ascii=False, indent=2))'
```

### 查看最新 revision

```bash
gcloud run services describe "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format='value(status.latestReadyRevisionName)'
```

### 查看目前 revisions

```bash
gcloud run revisions list \
  --service "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION"
```

### 查看目前 images

```bash
gcloud artifacts docker images list \
  "${REGION}-docker.pkg.dev/${PROJECT_ID}/linebot-cloud-agent/app" \
  --project "$PROJECT_ID" \
  --sort-by='~UPDATE_TIME'
```

---

## 實際測試時怎麼看 log

最推薦做法：先開一個 terminal 跑即時 log，再去 LINE 實際傳訊息、圖片、測搜尋 / 語音。

### 即時追全部 log

優先用這條：

```bash
gcloud beta run services logs tail "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION"

gcloud beta run services logs tail linebot-cloud-agent \
  --project terabanana \
  --region us-west1

```

如果這台機器沒有 `beta` 的 tail 指令，就改用下面這個**輪詢式監看**（實測可用）：

```bash
while true; do
  gcloud logging read \
    "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE_NAME}\"" \
    --project "$PROJECT_ID" \
    --limit 50 \
    --order=desc \
    --format='table(timestamp,severity,textPayload)'
  echo "----- refresh in 5s -----"
  sleep 5
done
```

### 查看最近全部 log

```bash
gcloud run services logs read "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --limit 200
```

```bash
gcloud logging read \
   "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE_NAME}\" AND textPayload:\"Input:\" AND textPayload:\"user_text=\" AND timestamp>=\"2026-04-02T00:00:00Z\" AND timestamp<=\"2026-04-06T23:59:59Z\"" \
   --project "$PROJECT_ID" \
   --limit 200 \
   --order=asc \
   --format='table(timestamp,textPayload)'
```

### 只看錯誤 log

```bash
gcloud logging read \
  "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE_NAME}\" AND severity>=ERROR" \
  --project "$PROJECT_ID" \
  --limit 50 \
  --format='table(timestamp,severity,textPayload)'
```

### 只看 webhook / 回覆相關

```bash
gcloud logging read \
  "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE_NAME}\" AND textPayload=~\"POST /webhook|Reply sent|Reply failed|Push failed|Input:\"" \
  --project "$PROJECT_ID" \
  --limit 50 \
  --format='table(timestamp,textPayload)'
```

### 只看聊天記憶摘要 log

```bash
gcloud logging read \
  "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE_NAME}\" AND textPayload:\"Memory summary\"" \
  --project "$PROJECT_ID" \
  --limit 50 \
  --format='table(timestamp,textPayload)'
```

### 只看警告以上

```bash
gcloud logging read \
  "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE_NAME}\" AND severity>=WARNING" \
  --project "$PROJECT_ID" \
  --limit 50 \
  --format='table(timestamp,severity,textPayload)'
```

實際測試時，如果看到 `ERROR`、`Reply failed`、`Push failed`、`/health` 非 `200`，就代表有功能要進一步排查。

---

## 不會自動清掉的東西

以下是正常保留，不影響你這個專案的乾淨部署：

- Cloud Build 歷史紀錄
- Cloud Logging 紀錄
- 你手動建立的其他 GCP 資源

這些不屬於 `deploy_cloud_run.sh` 的清理範圍。

---

## Endpoints

| Endpoint | Method | 用途 |
| --- | --- | --- |
| `/health` | `GET` | Readiness、provider 狀態、成本控制狀態、agent 呼叫計數 |
| `/webhook` | `POST` | LINE webhook（HMAC-SHA256 驗簽） |

---

## 成本控制

| 資源 | 控制方式 |
| --- | --- |
| LLM 呼叫 | 每個模型的 RPM/RPD 追蹤，429 時自動 fallback |
| LINE push | `LINE_PUSH_MONTHLY_LIMIT` 控制月度 push 上限 |
| 網路搜尋 | 不做 app 內配額限制；實際額度以 Tavily free plan / API 回應為準（`WEB_SEARCH_MONTHLY_QUOTA` 僅保留相容舊設定） |
| GCS 媒體 | Signed URL 48 小時過期，app 2 天後清理，部署腳本驗證 3 天 lifecycle 保底 |
| 使用者請求 | 每人滑動視窗 rate limit |
| Cloud Run | `max-instances=1`，`min-instances` 自動判斷（一般 `0`，有排程時 `1`） |

---

## 運行注意事項

- `/health` 可能回 `200` 但 `ready_for_webhook=false`，要看 payload 不能只看 HTTP status。
- 語音和圖片生成需要 `GCS_BUCKET_NAME`，沒設定時文字回覆仍正常。
- 目前只支援**語音輸出**，不支援把使用者傳來的音訊自動轉文字。
- 主 prompt 記憶是「1 份長期摘要 + 最多 5 則近期文字訊息」；目前仍是 **in-memory / per-instance**，重部署後會重置，多 instance 之間也不共享。
- `!new` 會清空當前 user / group / room 的近期記憶與長期摘要，並回覆 `Let's start a new chat!`。
- 長期摘要內容會直接寫進 log（loaded / updated / cleared），請自行評估 Cloud Logging 的敏感資訊風險。
- 引用訊息快取、LINE push 計數與模型 rate state 仍是 **in-memory / per-instance**；重部署後會重置，多 instance 之間也不共享。
- 網路搜尋目前不做 app 內配額限制；真正可用額度以 Tavily free plan / API 回應為準。
- Tavily 搜尋查詢會自動帶入目前日期時間，幫助「今天 / 最近 / 現在 / 最新」這類相對時間詞對齊到當下時點。
- 排程訊息需要同時設定 `SCHEDULED_MESSAGES_ENABLED=true`、`LINE_PUSH_FALLBACK_ENABLED=true`、`SCHEDULED_GROUP_ID`、以及至少一個排程 job。
- Prompts 從 `prompts/*.md` 載入，調整路由或語氣不需要改 Python 程式碼。

---

## 直接 Cloud Build 提交

如果不想用 wrapper 腳本，可以直接提交 Cloud Build：

```bash
gcloud builds submit . \
  --project YOUR_PROJECT_ID \
  --config cloudbuild.yaml \
  --substitutions=_SERVICE_NAME=linebot-cloud-agent,_REGION=us-west1,_IMAGE_TAG=manual-$(date -u +%Y%m%d%H%M%S)
```

這種方式適用於 Cloud Run service 已有環境變數設定，或你另外提供 env-vars 檔案（透過 `_ENV_VARS_FILE`）的情境。
