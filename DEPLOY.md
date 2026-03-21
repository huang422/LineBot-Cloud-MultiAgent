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
- `CLOUD_RUN_MIN_INSTANCES`
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

`CLOUD_RUN_MIN_INSTANCES` 留空時，部署腳本會自動判斷：

- 有啟用排程且有設定 job 時，預設用 `1`
- 其他情況預設用 `0`

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

如果你要指定 Cloud Run runtime service account：

```bash
./scripts/deploy_cloud_run.sh \
  --service-account YOUR_SERVICE_ACCOUNT_EMAIL
```

---

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
9. 若有設定 `GCS_BUCKET_NAME`，部署流程會自動把 Cloud Run runtime service account 補上 `roles/iam.serviceAccountTokenCreator`
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
