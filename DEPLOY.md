# Cloud Run 部署手冊

這個專案的正式部署方式就是跑：`./scripts/deploy_cloud_run.sh`。

同一條指令可用在：

- 第一次部署
- 你修改程式後再次部署
- 想要清掉舊 revision / 舊 image 的重部署

它會自動完成：**檢查 env（含已下架模型 / 舊版 key）、檢查媒體 bucket 是否在免費額度內、確認不會上傳任何密鑰檔、確認 GCP API、Artifact Registry 與 Cloud Tasks queue、送出 Cloud Build（ruff + pytest + 建置並推送 image）、把 `.env` 同步到 Secret Manager 並部署到同一個 Cloud Run service、設定 Cloud Scheduler、smoke check、清理舊 revision / image / secret 版本**。

第一次使用（或還沒做過 GCP 強化）請先跑一次〈GCP 佈建與強化（第一次）〉的 `./scripts/provision_gcp.sh`：它建立最小權限的 service account、Cloud Build 暫存 bucket 與存放 `.env` 的 secret。網路搜尋只用 Tavily，不需要 VM 或 VPC，整個專案維持在 GCP 免費額度內。還沒佈建時部署腳本仍可運作，但會用舊方式（預設 service account、暫存 env 檔）並印出警告。設計理由見 [ADR-0008](docs/adr/0008-secrets-and-identity.md)。

> 從 v2（Qwen3.5 / nv-embedqa-e5-v5 / `ORCHESTRATOR_MODEL`）升級？照下面的〈上線前檢查清單〉依序做一次；〈從 v2 升級到 v3〉有其中幾步的細節。

---

## 上線前檢查清單（需要你本人操作）

下面這些步驟只有你能做（主控台操作、瀏覽器同意、會花錢或無法復原的變更），請**依序**完成；每一步的細節在後面各節。各項設計理由的索引見 [README 的 Architecture decisions](README.md#architecture-decisions)。

1. **輪替外洩的密鑰**：v2 的部署把 `.env` 的所有值打包進 `gs://<專案>_cloudbuild/source/` 的 tarball（共 58 份）。到各主控台重新產生：LINE channel secret 與 channel access token（LINE Developers Console）、`NVIDIA_API_KEY`（build.nvidia.com）、`OPENROUTER_API_KEY`（openrouter.ai/keys）、`TAVILY_API_KEY`（tavily.com），並用 `openssl rand -hex 32` 產生新的 `INTERNAL_CRON_SECRET`（`INTERNAL_TASKS_SECRET` 若曾用舊方式部署過，也一起換），全部寫進 `.env`。LINE channel secret 重新產生後舊的立即失效：從這一步到第 5 步部署完成之前，bot 會驗簽失敗，請在同一個時段連續做完；其他服務的舊 key 等第 5 步 `/health` 正常後再撤銷（〈密鑰輪替〉）。舊 tarball 由第 4 步的 `purge-cloudbuild` 清除。
2. **建立預算警報（約 30 TWD）**：帳單幣別是 TWD；門檻為實際支出 50%、90%、100% 與預測支出 100%，只寄 email，**不設花費上限**。步驟見〈預算警報〉（`./scripts/provision_gcp.sh --step budget` 也會印出）。
3. **清理 `.env`**：`python3 scripts/envfile.py migrate-v3 --file .env`（先備份成 `.env.bak.<時間>`，只印 key 名稱），再跑 `python3 scripts/envfile.py check-v3 --file .env`，應該通過。`MODEL_ROLES` 不會被改動，裡面若有已下架的 id 要自己改。2026-10-01 起 push 不再計數或限制，`LINE_PUSH_FALLBACK_ENABLED`、`LINE_REPLY_FALLBACK_PUSH_ENABLED`、`LINE_PUSH_RESERVE`、`LINE_PUSH_MONTHLY_LIMIT` 已不再讀取（`check-v3` 只提醒、不擋部署，`migrate-v3` 會刪掉）；同日起每個請求只有一個回答期限 `REPLY_WINDOW_SECONDS`（預設 90 秒），`COLAB_REPLY_WINDOW_SECONDS` 也不再讀取（同樣只提醒，`migrate-v3` 會刪掉）。`.env` 裡若還寫著 `REPLY_WINDOW_SECONDS=50`，請改成 90 或刪掉這一行。
4. **GCP 佈建**：先跑 `./scripts/provision_gcp.sh`（dry-run：只做唯讀檢查，印出會執行的指令）看過一遍，再照〈GCP 佈建與強化（第一次）〉的順序**一步一步** `--step <名稱> --apply`（每一步都先看它自己的 dry-run）。`purge-cloudbuild` 要另外加 `--yes-purge`，`ar-cleanup-enforce` 等 `ar-cleanup` 一天後再跑。
5. **部署**：建議先跑 `.venv312/bin/python scripts/live_gate.py`（全部 PASS 或 SKIP），再跑 `./scripts/deploy_cloud_run.sh`。確認 `/health` 的 `ready_for_webhook` 是 `true`，`/health/detail` 的 `roles` 裡沒有 `dead: true`、`config_warnings` 是空的。
6. **遷移 episodic memory 向量**（`FIRESTORE_ENABLED=true` 才需要）：`gcloud auth application-default login` 後，先跑 `python3 scripts/migrate_embeddings.py`（dry-run，只印數量），數量合理再跑 `python3 scripts/migrate_embeddings.py --apply`。它和線上的 bot 共用 NVIDIA 每分鐘 40 次的額度，bot 忙碌時請在離峰執行，或加上 `--max-rpm 20`。說明見〈從 v2 升級到 v3〉第 4 步與 [ADR-0011](docs/adr/0011-embedding-migration.md)。
7. **開啟 LINE webhook redelivery**：LINE Developers Console → 你的 channel → Messaging API → Webhook settings，開啟 **Webhook redelivery**（Cloud Tasks 與 `webhookEventId` 去重都已上線，重送的事件會被丟掉，不會重複回覆）。
8. **（選用）Colab 層**，細節見〈Colab 層〉、[colab/README.md](colab/README.md) 與 [ADR-0010](docs/adr/0010-colab-ollama-tier.md)：
   1. 用**專用 Google 帳號**（只持有 Colab 方案、在這個 GCP 專案沒有任何角色）執行 `./scripts/mint_colab_adc.sh`（dry-run），再跑 `./scripts/mint_colab_adc.sh --apply`；
   2. `.env` 設定 `FIRESTORE_ENABLED=true`、`COLAB_SHARED_SECRET`、`INTERNAL_CRON_SECRET`（控制器由每分鐘的 `/internal/cron` 執行）、`LINE_ADMIN_USER_IDS`，以及試跑用的 `COLAB_ENABLED=true`、`COLAB_DRY_RUN=true`、`MODEL_PATH_PREFERENCE=api`（這樣只有 `!colab on` 會開機，一般訊息照舊走 API），然後部署；
   3. **CPU 乾跑**：`!colab on` 等到 READY，`.venv312/bin/python scripts/live_gate.py --path colab` 的 `colab.probe` 要 PASS，最後 `!colab off` 並確認 runtime 已釋放；
   4. **一次真正的 A100**（見〈Colab 層〉第 6 步）：`COLAB_DRY_RUN=false`、部署、`!colab on`，READY 後 `live_gate.py --path colab` 四項都要 PASS，記下開機時間、CU 每小時用量與 GPU，最後 `!colab off`；
   5. **對所有人開啟**：保留 `COLAB_ENABLED=true`，把 `MODEL_PATH_PREFERENCE` 改回 `colab` 並部署。
9. **最後才移除 `roles/editor`**：以上都正常之後（Cloud Build 以 `linebot-builder`、Cloud Run 以 `linebot-runtime` 執行，已經沒有東西在用 compute 預設 service account），用 `./scripts/provision_gcp.sh --step drop-editor` 先 dry-run（檢查替代帳號，印出 Policy Simulator 的位置與還原指令），在 Policy Simulator 確認沒有影響後再加 `--apply`，移除 compute 預設 service account 的 `roles/editor`。這個帳號本身**不要刪除**。

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

## GCP 佈建與強化（第一次）

`./scripts/provision_gcp.sh` 只需要跑一次（之後重跑也安全：每一步都先 describe，已存在就略過；只有 gcloud 回報 NOT_FOUND 才算「不存在」，權限、API 未啟用、登入過期或網路錯誤會直接停止，不會把讀取失敗當成「已刪除」或「沒東西可清」；唯一的例外是 dry-run 時 `apis` 步驟會啟用的 API（Cloud Tasks、Secret Manager）還沒啟用：gcloud 回報 `PERMISSION_DENIED`（reason `SERVICE_DISABLED`）時當成「還不存在（apis 步驟會啟用這個 API）」繼續規劃，`--apply` 時則停止並提示先跑 `--step apis --apply`）。gcloud 一律不會詢問（腳本設定 `CLOUDSDK_CORE_DISABLE_PROMPTS=1`），所以不會卡在看不見的「enable API? (y/N)」提示。**預設是 dry-run**：會真的執行唯讀檢查（describe、list、get-iam-policy），但所有會改變專案的指令只印出來（`[dry-run]` 開頭）；加上 `--apply` 才會執行（`[run]` 開頭）。設定從 `.env` 讀（`GCP_PROJECT_ID`、`CLOUD_RUN_REGION`、`GCS_BUCKET_NAME`、`CLOUD_TASKS_QUEUE`、`CLOUD_TASKS_LOCATION`）。`.env`（或 `--env-file`）不存在、讀不到、不是 UTF-8、或有 python-dotenv 讀不了的行（與 `secrets` 步驟相同的檢查，只列 key 名稱與行號）時，任何步驟都在呼叫 gcloud 之前停止；專案只取 `GCP_PROJECT_ID`，**不會**退回 `gcloud config` 的預設專案（空的就停止）。`--project-id` 必須與 `GCP_PROJECT_ID` 一致；只有明確給 `--project-id` 時才能不帶 env 檔執行（其他設定用預設值）。`mint_colab_adc.sh` 的規則相同。

```bash
./scripts/provision_gcp.sh --list      # 列出所有步驟
./scripts/provision_gcp.sh             # dry-run 全部步驟，先看一遍會做什麼
```

確認輸出沒問題後，**建議一步一步加 `--apply`**（每一步都先看 dry-run，再執行）：

```bash
./scripts/provision_gcp.sh --step apis --apply
./scripts/provision_gcp.sh --step runtime-sa --apply
./scripts/provision_gcp.sh --step staging-bucket --apply
./scripts/provision_gcp.sh --step builder-sa --apply
./scripts/provision_gcp.sh --step purge-cloudbuild --apply --yes-purge
./scripts/provision_gcp.sh --step secrets --apply
./scripts/provision_gcp.sh --step colab-secret --apply   # 選用：Colab token 已用 mint_colab_adc.sh 存好時
./scripts/provision_gcp.sh --step ar-cleanup --apply
./scripts/provision_gcp.sh --step budget      # 只印主控台步驟
./scripts/provision_gcp.sh --step summary     # 還需要你手動處理的事
```

| 步驟 | 內容 |
| --- | --- |
| `apis` | 啟用 `cloudtasks.googleapis.com`、`secretmanager.googleapis.com`（**不會**啟用 containerscanning / ondemandscanning，那是按 image 計費的） |
| `runtime-sa` | 建立 `linebot-runtime`：專案層級 `roles/datastore.user`、媒體 bucket 的 `roles/storage.objectUser`、對自己的 `roles/iam.serviceAccountTokenCreator`（簽 GCS URL）、`CLOUD_TASKS_QUEUE` 與 `CLOUD_TASKS_COLAB_QUEUE` 的 `roles/cloudtasks.enqueuer`（queue 已存在時；否則部署建立 queue 時補上）、每個已存在 secret 的 `roles/secretmanager.secretAccessor` |
| `staging-bucket` | 建立 `gs://<專案>-cb-src-usw1`：us-west1、STANDARD、uniform access、public access prevention、關閉 soft delete、3 天 lifecycle 刪除（`infra/gcp/lifecycle-delete-3d.json`）。bucket 名稱是全域唯一且可預測的：同名 bucket 若屬於別的專案（或讀不到），這一步會停止，不會沿用、授權或放原始碼進去；部署也只在本專案列得出這個 bucket 時才使用它 |
| `builder-sa` | 建立 `linebot-builder`：repository 的 `roles/artifactregistry.writer`、`roles/logging.logWriter`、staging bucket 的 `roles/storage.objectViewer`（Artifact Registry repository 不存在時一併建立） |
| `purge-cloudbuild` | 舊部署把 `.env` 所有密鑰打包進 `gs://<專案>_cloudbuild/source/` 的 tarball。依序：`--clear-soft-delete`（**必須先做**，否則刪掉的檔案 7 天內仍可復原）→ `gcloud storage rm .../source/**` → 1 天 lifecycle 規則。會印出醒目警告，`--apply` 時**必須另外加 `--yes-purge`** 才會刪除；全部步驟一起 `--apply` 而沒有 `--yes-purge` 時會略過這一步 |
| `secrets` | 用 `python3 scripts/envfile.py to-dotenv --file .env` 產生 dotenv blob，**經由 pipe**（不寫檔）建立 `linebot-dotenv`（automatic replication），並授權 `linebot-runtime` 讀取；secret 已存在就略過（之後的版本由部署腳本新增） |
| `colab-secret` | 讓 `linebot-runtime` 讀取 Colab 層的 token（`linebot-colab-adc` 的 `roles/secretmanager.secretAccessor`）。這一步**不會**建立 secret（token 只能由人在瀏覽器同意後產生，見〈Colab 層〉的 `scripts/mint_colab_adc.sh`）；secret 還不存在就略過並提示 mint 腳本。mint 腳本存好 token 後會自己執行這一步 |
| `ar-cleanup` | Artifact Registry 清理政策（`infra/gcp/ar-cleanup.json`：保留最新 2 個、刪除超過 1 天的其他 image），**先以 dry-run 模式設定**，不會刪任何東西 |
| `ar-cleanup-enforce` | 只能單獨用 `--step` 執行：約一天後（Artifact Registry 的 Data Access 稽核 log 會記錄 dry-run 本來會刪哪些）改成真正刪除（`--no-dry-run`）。**之後只能回滾到最新 2 個 image**：它不看 revision，其他 image 一天後就刪除（見〈回滾〉） |
| `budget` | 只印出主控台建立預算警報的步驟（見〈預算警報〉） |
| `summary` | 列出還需要你手動處理的事：輪替密鑰、預算警報、LINE webhook redelivery、purge、ar-cleanup-enforce、Colab（選用）的第一次設定，以及最後的 `drop-editor`（見下） |

佈建完成後照常執行 `./scripts/deploy_cloud_run.sh`：它偵測到 `linebot-builder` 與 staging bucket 就用它們跑 Cloud Build，偵測到 `linebot-runtime` 與 `linebot-dotenv` 就改用 secret 部署（見〈標準部署〉）。

### 最後一步：`drop-editor`（Phase 9）

只能單獨用 `--step` 執行，全部步驟一起跑時不會執行。它移除 Compute Engine 預設 service account（`<專案編號>-compute@developer.gserviceaccount.com`）的專案層級 `roles/editor`，**帳號本身不會刪除**。移除前先確認替代的帳號都已就位，缺任何一項就全部列出並停止（dry-run 也一樣）：

- `linebot-runtime`、`linebot-builder` 都存在，staging bucket 也在本專案（否則部署的 Cloud Build 會改用預設 service account）
- Cloud Run service `linebot-cloud-agent`（`CLOUD_RUN_SERVICE_NAME`）以 `linebot-runtime` 執行
- 沒有其他 Cloud Run service（所有 region；沒指定 service account 的也算）、`CLOUD_RUN_REGION` 的 Cloud Scheduler job 或 VM（包括已停止的）以 compute 預設 service account 執行

以 compute 預設 service account 執行的舊 VM 會讓它停止：例如舊的 `instance-gcp-banana`，或先前自架搜尋時佈建的 VM（見 [ADR-0009](docs/adr/0009-search-backends-and-ssrf-safe-fetching.md)）。這個 bot 不需要任何 VM，開機中的 VM 還會按時計外部 IPv4 費用，所以先確認裡面沒有要留的資料，再手動刪除，然後重跑 `drop-editor`：

```bash
gcloud compute instances list --project YOUR_PROJECT_ID
gcloud compute instances delete VM_NAME --zone ZONE --project YOUR_PROJECT_ID
```

專案的 Compute Engine API 若是關閉的，`drop-editor` 會把它讀成「沒有 VM」，不會因此停止。

檢查通過後，它先印出 Policy Simulator 的位置（主控台 IAM 頁面：編輯該帳號、移除 Editor，按 **Test changes** 而不是儲存）與還原指令，再移除這個 binding；已經沒有這個 binding 時只印出 `nothing to do`，重跑是安全的。

```bash
./scripts/provision_gcp.sh --step drop-editor           # dry-run：檢查、Policy Simulator 網址、還原指令
./scripts/provision_gcp.sh --step drop-editor --apply
# 還原：
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member=serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com \
  --role=roles/editor --condition=None
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
- `INTERNAL_TASKS_SECRET`（`openssl rand -hex 32` 產生；`TASKS_MODE=inline` 時才可省略，見下方〈Webhook 之後的工作：Cloud Tasks〉）

建議同時確認這幾個部署欄位：

- `CLOUD_RUN_SERVICE_NAME`
- `CLOUD_RUN_REGION`
- `DEPLOY_KEEP_REVISIONS`
- `DEPLOY_KEEP_IMAGES`
- `DEPLOY_ENABLE_APIS`

模型設定建議**直接沿用程式預設的模型角色表**（`.env` 不設 `MODEL_ROLES`）。v3 預設組合（`src/runtime/model_roles.py`，理由見 ADR-0007）：

| 角色 | 用途 | 各層（依序） |
| --- | --- | --- |
| `router` | LLM 路由（一律關閉思考） | Colab `nimble:9b` → NVIDIA `nvidia/nemotron-3-super-120b-a12b`（json_schema，temperature 0.1）→ NVIDIA `nvidia/nemotron-3-ultra-550b-a55b`（沒有結構化輸出，只靠 prompt 要求 JSON；temperature 0.1）→ OpenRouter `nvidia/nemotron-3-super-120b-a12b:free`（json_object，effort none） |
| `chat` | 對話、網搜回答 | Colab `qwen3.8:27b-mtp-q4_K_M`（`COLAB_CHAT_THINK`，70 秒；後面有 API 層時最多用到期限前 20 秒）→ NVIDIA `nvidia/nemotron-3-ultra-550b-a55b`（關閉思考）→ `qwen/qwen3.8-27b:free`（effort none）→ `nvidia/nemotron-3-ultra-550b-a55b:free`（effort none） |
| `thinking` | router 判定需要思考時先走這裡，之後接 chat 各層（關閉思考） | Colab qwen3.8（`COLAB_DEEP_THINK`）→ NVIDIA Ultra（開啟思考，`medium_effort`；Ultra 收到 `reasoning_budget` 會回 400，所以不送，`NVIDIA_THINKING_BUDGET` 只放寬 `max_tokens`）→ `qwen/qwen3.8-27b:free`（effort medium）→ `nvidia/nemotron-3-ultra-550b-a55b:free`（effort medium） |
| `vision` | 有圖片的請求（不論哪個 agent 回答） | Colab qwen3.8（70 秒）→ NVIDIA `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`（關閉思考，temperature 0.2，max_tokens 1024，15 秒）→ NVIDIA `meta/muse-glimmer-30b`（關閉思考，35 秒）→ `qwen/qwen3.8-27b:free`（effort none，15 秒）。API 層一律不思考，需要思考時只有 Colab 思考；引用的圖片若是 URL（訊息快取 / GCS），Colab 依設計略過（摘要記 `skip(by design: url image)`），直接由 API 層回答；有圖片的 prompt 只帶 3000 字的上下文（`IMAGE_CONTEXT_CHARS`：profile、最近對話由新到舊整輪保留、剩下的給長期摘要），整體約 4k tokens（2026-10-01 實測 muse-glimmer 在 6.2k tokens 要約 40 秒，4k 約 8.5–19 秒）。Nano Omni 忙碌時回 503（不到 1 秒），直接換下一層。Nemotron Ultra 只收文字，所以不在這個角色 |
| `utility` | 記憶摘要、搜尋查詢規劃、生圖提示改寫與說明文字 | NVIDIA `nvidia/nemotron-3-super-120b-a12b`（關閉思考，temperature 0.2）→ NVIDIA `nvidia/nemotron-3-ultra-550b-a55b`（關閉思考）→ `nvidia/nemotron-3-ultra-550b-a55b:free`（effort none） |
| `embed` | episodic recall | NVIDIA `nvidia/nemotron-3-embed-1b`（取前 1024 維） |
| `image` | 生圖 | NVIDIA `black-forest-labs/flux.2-klein-4b` → `black-forest-labs/flux.1-dev`（50 steps） |

注意：

- 帶 `:free` 的模型走 OpenRouter，不帶的走 build.nvidia.com；`OPENROUTER_ALLOW_PAID=false`（預設）時角色表裡的付費 OpenRouter id 會讓啟動直接失敗
- Colab 層要設定 `COLAB_ENABLED=true` 才會啟用（見下方〈Colab 層〉），否則一律停用（`/health/detail` 會列出原因）；`MODEL_PATH_PREFERENCE=colab`（預設）讓各角色先走 Colab、nimble 在 API router 之前路由，而且通過限流的觸發訊息會記錄需求並自動開機；`api` 則先走 NVIDIA → OpenRouter 與原本的 API router，Colab 層排到最後、只在 session 已經 READY（例如管理員 `!colab on`）時使用，nimble 只在 API router 失敗時使用，訊息不會自動開機。Colab 隨需開機、閒置 15 分鐘釋放；開機完成（READY）前 Colab 層一律立刻略過（不發任何網路請求），每則訊息都由 NVIDIA → OpenRouter 直接回覆，不會等 Colab；READY 之後 Colab 先回答，期限和其他請求一樣是事件時間 + `REPLY_WINDOW_SECONDS`（90 秒，私訊與群組都一樣），後面還有 API 層的 Colab 層最多用到期限前 20 秒（`COLAB_API_RESERVE_SECONDS`；期限短時最多一半，不論思考等級），Colab 變慢時 API 層仍有時間回覆；送出一律先試 reply，LINE 拒絕（reply token 已失效）才改用 push
- 取樣：層自己的 `temperature` 優先，其次是模型卡建議值（NVIDIA Nemotron Ultra / Super：temperature 1.0、top_p 0.95；muse-glimmer：0.6），最後才是 `CHAT_TEMPERATURE` 等 agent 設定（只影響 OpenRouter 這些沒有模型卡取樣的層）。Colab 的 qwen3.8 固定用 Qwen 建議的取樣（不思考 0.7 / top_p 0.8 / top_k 20 / presence penalty 1.5，思考 1.0 / 0.95 / 20），除非該層自己設 `temperature`。router 的 Super 與 Ultra 層、utility 的 Super 層刻意維持低溫（0.1 / 0.1 / 0.2），讓路由 JSON、搜尋查詢與記憶摘要穩定
- OpenRouter 的 `nvidia/nemotron-3-*:free` 由 NVIDIA 託管，可能與 NIM 同時失效，所以排在各角色最後
- 2026-10-01 起預設不再使用 Gemma（NIM 的 Gemma 4 31B 實測每次都卡住，OpenRouter 的 Gemma `:free` 每次都回 429）：router、chat、utility 改由 Nemotron 3 Ultra 備援；Ultra 只收文字、沒有結構化輸出，所以 vision 的備援改為 OpenRouter qwen3.8，router 的 Ultra 層只靠 prompt 要求 JSON，由 orchestrator 的容錯解析器（加一次修復提示）讀取。`MODEL_ROLES` 若還寫著 Gemma 的 id，`check-v3` 會提醒（不擋部署），請自行改掉
- 要換某個角色的模型，用 `MODEL_ROLES`（JSON，只取代列出的角色；格式見 `.env.example`）。JSON 錯誤、未知角色 / provider、付費或已下架的 id、模型做不到該角色時，啟動會直接失敗，不會默默略過
- `FORCE_TIER`（例如 `nvidia`、`openrouter`）**只供測試**：停用其他所有層；若讓 router / chat / vision / utility 沒有可用的層，啟動會失敗。正式環境請留空
- 已下架、會回 HTTP 410 的 `qwen/qwen3.5-397b-a17b`、`qwen/qwen3.5-122b-a10b`、`nvidia/nv-embedqa-e5-v5`（包括寫在 `MODEL_ROLES` 裡），以及已停用的 `ORCHESTRATOR_MODEL` / `ORCHESTRATOR_FALLBACK_MODEL`、`REQUIRE_REASONING_MODELS=true`，部署腳本會直接拒絕（`python3 scripts/envfile.py check-v3 --file .env`）；修正指令：`python3 scripts/envfile.py migrate-v3 --file .env`（`MODEL_ROLES` 需自行修改）
- 舊版的逐模型 key（`NVIDIA_MODEL`、`ROUTER_MODEL`、`*_FALLBACK_MODEL`、`IMAGE_GEN_*`、`NVIDIA_EMBEDDING_MODEL`、各種 reasoning 開關等）已不再讀取；啟動時會列出仍存在的 key 名稱，`migrate-v3` 會把它們刪掉
- thinking 參數（`chat_template_kwargs.enable_thinking` 與 Ultra 的 `medium_effort`、Super 的 `reasoning_budget`、muse-glimmer 的 `reasoning_effort`、OpenRouter `reasoning`）由應用程式依模型與角色層自動套用，不需要改 deploy 指令
- 每天第一次 `/internal/cron` 會排入一個 `eol` 任務，用公開的模型目錄檢查每個設定中的模型是否下架、即將停用或沒有 endpoint（不帶 API key）；狀態改變時記 ERROR log，結果在 `/health/detail`

如果你有開：

- `SCHEDULED_MESSAGES_ENABLED=true`

那要確認：

- `SCHEDULED_GROUP_ID`
- `SCHEDULED_WEEKLY_MESSAGES` 或 `SCHEDULED_YEARLY_MESSAGES` 至少有一個有內容
- `INTERNAL_CRON_SECRET`（建議 `openssl rand -hex 32`）— 提供給 Cloud Scheduler 用的共用密鑰

LINE 訊息的送法（`src/services/line_service.py`）：

- 每則回答，不論私訊或群組，一律先用免費的 reply；只有 reply **確定**失敗（reply token 過期或無效、請求根本沒送出）才改用 push。送出後結果不明的 reply（逾時、5xx）視為已送出，不會再 push，避免重複
- push 不計數、不設上限、不保留額度也不會被拒絕：唯一的限制是 LINE 自己的每月額度（私訊每則算 1，群組 / 多人聊天室依成員數計；免費方案用完後 LINE 會拒絕所有訊息，包括排程提醒）。2026-10-01 起舊的 `LINE_PUSH_*` 設定已移除
- 排程訊息一律直接 push
- 每則訊息的摘要 log 記 `delivery=reply` 或 `delivery=push`（見〈只看每則訊息的摘要〉），可用來觀察 push 用量

部署腳本會自動處理這幾件事：

- `min-instances` 永遠是 `0`（scale-to-zero，吃 Cloud Run 免費額度），`max-instances` 與 service 層級上限都是 `1`
- 啟用 `cloudscheduler.googleapis.com`；只要 `INTERNAL_CRON_SECRET` 有值（即使沒開排程訊息）就建立 / 更新 `<service>-cron` Cloud Scheduler job，secret 留空時刪除該 job
- 該 Cloud Scheduler job 每分鐘 POST `/internal/cron`，由應用程式自行決定哪些 weekly/yearly job 該觸發，並順便寫入超過 `PASSIVE_GROUP_FLUSH_DELAY_SECONDS` 仍未達門檻的群組訊息緩衝（沒有排程訊息也會做，所以建議一律設定 `INTERNAL_CRON_SECRET`）
- 查詢、建立、更新與刪除 job 的 `gcloud` 呼叫都會關閉 gcloud 的檔案 log（`CLOUDSDK_CORE_DISABLE_FILE_LOGGING=true`），避免 `X-Cron-Token`（旗標值，以及 describe 回傳的 header）被寫進 `~/.config/gcloud/logs`；執行期間它仍會短暫出現在本機的 process list
- runtime service account：已佈建時用 `linebot-runtime`（`CLOUD_RUN_SERVICE_ACCOUNT` 可覆寫，但該帳號需要 `linebot-dotenv` 的讀取權）；尚未佈建時沿用目前 Cloud Run 設定，第一次部署則使用預設 service account

---

## Webhook 之後的工作：Cloud Tasks

Cloud Run 用 request-based 計費（`--cpu-throttling`）：只有在處理 request 時才有 CPU。舊版在回 200 之後用 `asyncio.create_task` 繼續處理，實際上幾乎拿不到 CPU，回覆會很慢，reply token 也常常過期而改用計入 LINE 每月額度的 push。

現在 webhook 只做驗簽、去重（`webhookEventId`）、群組訊息緩衝與文字快取，其餘工作都變成 Cloud Tasks 任務，由 Cloud Tasks 再 POST 回 `/internal/tasks/{kind}`（這是一個真正的 request，有 CPU）：

| kind | 內容 |
| --- | --- |
| `event` | 處理一則觸發的訊息（路由、模型、回覆、寫入記憶） |
| `passive_flush` | 群組訊息緩衝達到 `PASSIVE_GROUP_FLUSH_THRESHOLD`，或最舊一則已超過 `PASSIVE_GROUP_FLUSH_DELAY_SECONDS` 又有新訊息時一次寫入；若排入後 `PASSIVE_GROUP_FLUSH_DELAY_SECONDS` 內沒寫入（任務遺失），下一則訊息會再排一次 |
| `cache_image` | 下載並快取未觸發的圖片，讓之後引用時看得到 |
| `compact` | 近期訊息超過上限時的長期記憶摘要 |
| `eol` | 每個台北時間日第一次 `/internal/cron` 排入的模型下架檢查（查公開的模型目錄，不帶 API key） |
| `colab` | Colab 控制器的步驟（assign、launch、heartbeat、unassign、reconcile），排進專用 queue `CLOUD_TASKS_COLAB_QUEUE`；只在 `COLAB_ENABLED=true` 時出現（見〈Colab 層〉） |

部署腳本會：

- 啟用 `cloudtasks.googleapis.com`
- 建立 / 更新 queue `CLOUD_TASKS_QUEUE`（預設 `linebot-events`，location 為 `CLOUD_TASKS_LOCATION`，預設 `us-west1`）：`--max-attempts=1`、`--max-concurrent-dispatches=4`、`--max-dispatches-per-second=5`，**不設** `--max-retry-duration`（Cloud Tasks 要兩個條件都滿足才停止重試，設了反而會重試）
- `COLAB_ENABLED=true` 時再建立 / 更新 Colab 控制器專用的 queue `CLOUD_TASKS_COLAB_QUEUE`（預設 `linebot-colab`）：同樣 `--max-attempts=1`，`--max-concurrent-dispatches=2`、`--max-dispatches-per-second=1`。`colab` 任務（assign、launch 等，每個數秒到數分鐘）因此不會佔用 LINE 事件 queue 的 4 個派送名額；它與 `CLOUD_TASKS_QUEUE` 同名時停止部署
- 給 runtime service account 每個 queue 的 `roles/cloudtasks.enqueuer`
- `TASKS_MODE` 不是 `inline` 但 `INTERNAL_TASKS_SECRET` 是空的就直接停止部署
- queue 不是 `RUNNING`（例如被暫停成 `PAUSED`）就停止部署並印出 `gcloud tasks queues resume` 指令：暫停的 queue 仍會接受任務，但一個都不會送出
- `TASK_HANDLER_BUDGET_SECONDS` 不小於 `TASK_DISPATCH_DEADLINE_SECONDS` 就停止部署；Cloud Run 的 `--timeout` 設成 max(300, `TASK_DISPATCH_DEADLINE_SECONDS`)

相關設定：

- `TASKS_MODE=auto`（預設）：在 Cloud Run 上且 `INTERNAL_TASKS_SECRET` 有值就用 Cloud Tasks，否則在 request 內直接處理（本機開發就是這樣）
- `INTERNAL_TASKS_SECRET`：Cloud Tasks 帶在 `X-Tasks-Token` 的共用密鑰，**與 `INTERNAL_CRON_SECRET` 分開**；留空時 `/internal/tasks` 一律 403
- `PUBLIC_BASE_URL`：Cloud Tasks 回呼的網址，必須是 `https://`；留空即可，程式會從第一個通過驗證的 request 的 Host 推得（一律用 https）
- `TASK_DISPATCH_DEADLINE_SECONDS=300`、`TASK_HANDLER_BUDGET_SECONDS=280`：Cloud Tasks 等待上限與單一任務的處理預算；預算必須小於等待上限
- `REPLY_WINDOW_SECONDS=90`：回答期限，LINE 事件時間加上這個秒數；每個不需思考的回答（私訊與群組，不論由 Colab 或 API 層回答）都要在期限內完成（見〈運行注意事項〉的回覆期限），必須小於 `TURN_DEADLINE_SECONDS`（否則啟動失敗）。送出一律先試 reply，LINE 拒絕才改用 push。任務開始時已超過期限會記錄警告（reply token 可能已過期，多半會改用 push）

注意：

- enqueue 失敗時 webhook 會在 request 內直接處理並記錄 ERROR，此時可能超過 LINE 的 2 秒 webhook 期限
- Cloud Tasks 與 LINE redelivery 都可能重送，程式用 `webhookEventId` 在程序內去重，這只在 `max-instances=1` 時成立
- 請在 LINE Developers Console 開啟 **Webhook redelivery**（預設關閉）
- `colab` 以外的任務（`event`、`passive_flush`、`cache_image`、`compact`、`eol`）共用 `CLOUD_TASKS_QUEUE` 的 4 個併發名額；模型服務卡住時，`compact`（最長 `MEMORY_SUMMARY_TIMEOUT_SECONDS`）可能佔住名額，讓 `event` 任務排隊超過 reply window 而改用 push
- 費用：request-based 計費會把每個任務 request 的整段時間（含等待模型回應）算進 Cloud Run 的 vCPU-s／GiB-s（免費額度每月 180,000 vCPU-s、360,000 GiB-s）；例如每天 300 則觸發訊息、每則 30 秒，約 270,000 vCPU-s／月，會超出免費額度（約 US$2／月）。要完全不產生費用，每月要留在 180,000 vCPU-s 以內（每則 30 秒計約每天 200 則，另有每分鐘 cron 等少量用量；摘要 log 的 `total=` 是每則的處理秒數）。建議設定帳單預算提醒（〈預算警報〉）
- 生成的圖片 / 語音不再由程式延遲刪除，改由 bucket 的 3 天 lifecycle 規則刪除；只有傳送失敗的物件會立即刪除。舊的 `GCS_MEDIA_CLEANUP_DELAY_SECONDS` 已停用，可從 `.env` 刪除（留著只會在啟動時出現提醒）

---

## 標準部署

最推薦、最簡單的方式：

```bash
./scripts/deploy_cloud_run.sh
```

設定怎麼進到 Cloud Run（已佈建時）：

- `.env` 去掉部署用的 key（`GCP_PROJECT_ID`、`CLOUD_RUN_*`、`DEPLOY_*`、`PORT`、`HOST`、`GOOGLE_APPLICATION_CREDENTIALS`）後轉成單引號 dotenv blob（`python3 scripts/envfile.py to-dotenv`），比對 `linebot-dotenv` 最新版本的 SHA-256（不印出任何一方），**只有內容改變時才新增版本**
- Cloud Run 以 `linebot-runtime` 執行，secret 掛載在 `/secrets/app/app.env` 並**釘住版本號**（`--set-secrets=/secrets/app/app.env=linebot-dotenv:N`）；唯一的一般環境變數是 `APP_VERSION`（`--set-env-vars`，它會清掉舊版部署留下的所有明文環境變數）
- 程式依序讀 `.env`（本機）→ `/secrets/app/app.env`（後者優先，不存在就略過），真正的環境變數永遠最優先
- `.env` 的值若含 `${`（python-dotenv 會把它當變數展開）、以反斜線結尾或含 CR，或引號寫壞（引號沒有關上、關閉引號後面還有非註解文字；python-dotenv 會直接略過這個 key），部署會在建置前停止並列出 key 名稱
- 讀取最新版本比對時關閉 gcloud 的檔案 log（`CLOUDSDK_CORE_DISABLE_FILE_LOGGING=true`）：gcloud 會把自己印出的所有內容複製到 `~/.config/gcloud/logs`（保留約 30 天）
- smoke check 通過後，銷毀「任何 region 的任何 Cloud Run service 的 revision 都沒有釘住、也不在最新 max(2, `DEPLOY_KEEP_REVISIONS`) 個之內」的 secret 版本（釘住的版本直接從 revision 讀出：先用 `gcloud run services list` 找出所有有 service 的 region，再逐一讀它們的 revision，讀不到就整個略過清理。所以部署失敗留下的版本不會把上一個 revision 回滾要用的版本擠掉；`linebot-dotenv` 整個專案只有一個，用 `--service-name` 或在其他 region 部署的另一個 service 也不會銷毀別的 service 釘住的版本）；`--skip-smoke` 時不會銷毀任何版本，有效版本超過 3 個時會警告。停用（disabled）的版本仍會計費，只有銷毀才免費
- 部署不接 VPC（不加 `--network`、`--subnet`、`--vpc-egress`）：LINE、NVIDIA、OpenRouter、Tavily 都走 Cloud Run 自己的對外連線。先前曾以 Direct VPC egress 部署過的 service 不會因此斷開，需要時手動 `gcloud run services update SERVICE --region REGION --clear-network`
- `.env` 有 `COLAB_ENABLED=true` 時（見〈Colab 層〉）：把 `linebot-colab-adc` 最新一個 **enabled** 版本**釘住版本號**掛載在 `/secrets/colab/adc.json`（與 dotenv 放在同一個 `--set-secrets`，例如 `--set-secrets=/secrets/app/app.env=linebot-dotenv:N,/secrets/colab/adc.json=linebot-colab-adc:M`），回滾時 token 跟著 revision 走；部署本身**從不讀取** token 內容。secret 不存在、沒有 enabled 版本或尚未佈建時只警告（Colab 層維持停用，服務照常部署）；`linebot-runtime` 讀不到它時在建置前停止（Cloud Run 會拒絕這個 revision），提示 `provision_gcp.sh --step colab-secret --apply`。`COLAB_ENABLED` 關閉時不掛載、也不碰這個 secret。smoke check 通過後，同樣銷毀沒有 revision 釘住、也不在最新 max(2, `DEPLOY_KEEP_REVISIONS`) 個之內的 `linebot-colab-adc` 版本
- 尚未佈建時：改用 `$TMPDIR` 裡只有自己可讀的 env-vars YAML（`--env-vars-file`，含 `APP_VERSION`，部署完立即刪除），並印出「Run ./scripts/provision_gcp.sh first」警告。只有 gcloud 回報 NOT_FOUND 才會退回舊方式；權限、API、登入或網路錯誤會直接停止部署。Cloud Run service 已經掛載 `linebot-dotenv` 時（例如 secret 或 `linebot-runtime` 被誤刪），部署會拒絕改回明文環境變數，除非加上 `--allow-env-fallback`

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
- `min-instances` 永遠 `0`（free-tier scale-to-zero）；排程改由 Cloud Scheduler 觸發
- 將 100% 流量切到最新 revision
- 在 Cloud Build 內跑 `ruff`、`pytest` 和 `compileall`（Cloud Build 只測試、建置、推送 image，不部署也拿不到任何密鑰）
- 部署後跑 smoke check（簽章 webhook 檢查送的是空的 `events`，不會寫入正式記憶）
- 預設保留最新 **2 個 Ready 的 revision**（`DEPLOY_KEEP_REVISIONS`；從未 Ready 的失敗 revision 不佔名額，等到有這麼多個 Ready revision 比它新才刪除）
- 預設保留最新 **2 個 image digest**（`DEPLOY_KEEP_IMAGES`；同一 region 任何 service 的 revision 還在用的 image 不刪）
- 已佈建時：`.env` 沒變就沿用同一個 `linebot-dotenv` 版本，有變才新增；smoke check 通過後銷毀回滾用不到的舊版本

也就是說，**舊的 revision 不會一直堆著**，舊的 docker image 也不會一直累積。

---

## 如果你想更乾淨

如果你連舊 image 也只想留 1 個：

```bash
./scripts/deploy_cloud_run.sh --keep-revisions 1 --keep-images 1
```

這會更乾淨，但回滾空間也更少。想每次部署都這樣清，就在 `.env` 設 `DEPLOY_KEEP_REVISIONS=1`、`DEPLOY_KEEP_IMAGES=1`：只留目前的 revision 與它的 image，無法用切流量回滾（要回到舊版只能用舊程式碼重新部署）；`linebot-dotenv` 仍至少保留最新 2 個版本。

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

### 刻意改回明文環境變數

```bash
./scripts/deploy_cloud_run.sh --allow-env-fallback
```

只在 `linebot-dotenv` 或 `linebot-runtime` 真的不存在、而你確定要讓目前掛載 secret 的 service 暫時改用明文環境變數時使用（所有 `.env` 值會對能檢視 service 的人可見）。一般應先用 `./scripts/provision_gcp.sh --step runtime-sa --apply` / `--step secrets --apply` 補回來。

---

## 這支腳本實際做了什麼

1. `.env` 檢查：`python3 scripts/envfile.py check-v3 --file .env`；仍指定已下架模型（含 `MODEL_ROLES` 裡的）、`MODEL_ROLES` 不是 JSON 物件、舊版 `ORCHESTRATOR_*` key、`REQUIRE_REASONING_MODELS=true`，或有 python-dotenv 讀不懂的引號值就停止（只印 key 名稱）；不是 KEY=VALUE 的行（例如 key 後面漏了 `=`，值裡又有 `=`）只印行號，不印內容
2. 驗證必要 env key 是否存在
3. 上傳檢查：`gcloud meta list-files-for-upload` 列出的檔案只要含 `.env*`（`.env.example` 除外）、`.cloudrun-env*`、`settings.local.json`、`.claude/`、`credentials.json`、`*.bak` 或 migrate-v3 的備份檔（`<檔名>.bak.<時間>`）就停止；`--env-file` 指到 repo 裡的檔案（不論檔名，例如 `prod.env`）時，它也不能在上傳清單裡，而且必須被 git 忽略，否則停止（放到 repo 外面，或加進 `.gitignore` 與 `.gcloudignore`）
4. 若有設定 `GCS_BUCKET_NAME`：bucket 必須是 `US-WEST1` / `US-CENTRAL1` / `US-EAST1` 的 STANDARD bucket（Always Free 5 GB），否則停止並印出建立新 bucket 的指令
5. 偵測佈建結果：`linebot-builder` 存在且本專案列得出 `gs://<專案>-cb-src-usw1` → Cloud Build 以 `linebot-builder` 執行並把原始碼放到該 bucket；`linebot-runtime`（或 `CLOUD_RUN_SERVICE_ACCOUNT`）與 `linebot-dotenv` 都存在 → secret 部署（並先檢查 `.env` 能否轉成 dotenv blob）。gcloud 回報 NOT_FOUND 時各自退回舊方式並警告（service 已掛載 `linebot-dotenv` 時則拒絕，除非 `--allow-env-fallback`）；其他查詢錯誤直接停止
6. 自動確保 `cloudbuild.googleapis.com`、`run.googleapis.com`、`artifactregistry.googleapis.com`、`iamcredentials.googleapis.com`、`cloudresourcemanager.googleapis.com`、`cloudscheduler.googleapis.com`、`cloudtasks.googleapis.com` 已啟用，並確保 Artifact Registry repository 與 Cloud Tasks queue 存在（`TASKS_MODE=inline` 時略過 queue）
7. 送出 `cloudbuild.yaml`：在與 image 相同的 `python:3.12-slim`（digest 固定）上以 `--require-hashes` 安裝 `requirements-dev.lock`，跑 `ruff` + `pytest` + `compileall`，然後建置並推送 image（不上傳密鑰、不部署）
8. secret 部署：`.env` 有變才新增 `linebot-dotenv` 版本，`gcloud run deploy --service-account linebot-runtime@... --set-secrets=/secrets/app/app.env=linebot-dotenv:N --set-env-vars=APP_VERSION=<tag>`（`COLAB_ENABLED=true` 時同一個旗標再加上 `,/secrets/colab/adc.json=linebot-colab-adc:M`，建置前先確認版本與讀取權限）；尚未佈建：在 `$TMPDIR` 產生只有自己可讀的 env-vars YAML，`gcloud run deploy --env-vars-file ... --clear-secrets`（關閉 gcloud 的檔案 log：gcloud 會把解析後的每個旗標寫進 `~/.config/gcloud/logs` 的 DEBUG log，`--env-vars-file` 解析出來就是 `.env` 的每個值），部署完立即刪除。
9. 將 100% 流量切到最新 revision
10. 若有設定 `GCS_BUCKET_NAME`，檢查 / 補上 bucket 的 3 天 lifecycle 保底刪除規則；簽 URL 需要 runtime service account 對自己的 `roles/iam.serviceAccountTokenCreator`：`linebot-runtime` 由 `provision_gcp.sh --step runtime-sa` 授權一次，其他帳號（`CLOUD_RUN_SERVICE_ACCOUNT`，或尚未佈建時 service 原本的帳號）缺少時由部署補上（沒有它，圖片與語音回覆會退回文字）
11. 給 runtime service account queue 的 `roles/cloudtasks.enqueuer`，開放 LINE webhook 的公開存取，建立 / 更新或刪除 Cloud Scheduler job
12. 對 `/health` 和 `/webhook` 做 smoke check
13. 清理舊 revision（只數 Ready 的）/ image（Artifact Registry 清理政策已強制執行、而保留中的 revision 用到的 image 不在最新 2 個之內時會警告）；smoke check 通過後銷毀任何 region 都沒有 revision 釘住的舊 `linebot-dotenv` 版本（有掛載時也包括 `linebot-colab-adc`）

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

curl -fsS "$SERVICE_URL/health" | python -m json.tool     # 公開：status / ready_for_webhook / version

# 詳細狀態（各模型層、額度、config 警告、每日下架檢查、記憶、儲存、排程、Colab）需要 cron 密鑰
CRON_SECRET="$(python3 scripts/envfile.py get --file .env INTERNAL_CRON_SECRET)"
curl -fsS -H "X-Cron-Token: $CRON_SECRET" "$SERVICE_URL/health/detail" | python -m json.tool
```

### 查看部署後目前設定的排程 push 訊息內容

排程從舊版的 in-process APScheduler 改成 Cloud Scheduler 每分鐘觸發 `/internal/cron`。可以從下面幾個角度查：

**A. 看「線上 app 實際載入的排程」（最直觀，含每筆 next_run 自動換算到 Asia/Taipei）**

```bash
PROJECT_ID="$(python3 scripts/envfile.py get --file .env GCP_PROJECT_ID)"
SERVICE_NAME="$(python3 scripts/envfile.py get --file .env CLOUD_RUN_SERVICE_NAME)"
REGION="$(python3 scripts/envfile.py get --file .env CLOUD_RUN_REGION)"
SERVICE_URL="$(gcloud run services describe "$SERVICE_NAME" \
  --project "$PROJECT_ID" --region "$REGION" \
  --format='value(status.url)')"

CRON_SECRET="$(python3 scripts/envfile.py get --file .env INTERNAL_CRON_SECRET)"
curl -fsS -H "X-Cron-Token: $CRON_SECRET" "$SERVICE_URL/health/detail" | python3 -c "
import json, sys
jobs = sorted(json.load(sys.stdin)['scheduler']['jobs'], key=lambda j: j['next_run'])
print('{:<28} {}'.format('NEXT RUN (Asia/Taipei)', 'ID'))
print('-' * 60)
for j in jobs:
    print('{:<28} {}'.format(j['next_run'], j['id']))
"
# /health/detail 只列 id 與 next_run、不含訊息文字；訊息內容請看 .env 的 SCHEDULED_*_MESSAGES
```

**B. 看「部署用的 `.env` 原始設定」（SCHEDULED_WEEKLY_MESSAGES / SCHEDULED_YEARLY_MESSAGES 的字串內容）**

已佈建時，Cloud Run 讀的是由 `.env` 產生、掛載成檔案的 `linebot-dotenv` secret，service 的環境變數只有 `APP_VERSION`，所以 `gcloud run services describe` 看不到這些值；直接看 `.env`（部署後改過 `.env` 但還沒重新部署時，線上的仍是上一次部署的內容）：

```bash
for KEY in SCHEDULED_MESSAGES_ENABLED SCHEDULED_GROUP_ID SCHEDULED_MESSAGES_TIMEZONE SCHEDULED_WEEKLY_MESSAGES SCHEDULED_YEARLY_MESSAGES; do
  printf '%s=%s\n' "$KEY" "$(python3 scripts/envfile.py get --file .env "$KEY")"
done
```

**C. 看「Cloud Scheduler 觸發器本身」（只含每分鐘 HTTP POST 設定，不含訊息內容）**

```bash
gcloud scheduler jobs describe "${SERVICE_NAME}-cron" \
  --project "$PROJECT_ID" \
  --location "$REGION" \
  --format='yaml(schedule,timeZone,state,httpTarget.uri,httpTarget.httpMethod,attemptDeadline,lastAttemptTime)'
```

**D. 查最近 cron 是否真的有觸發 `/internal/cron`（10 分鐘內每分鐘應該都有一筆 200）**

```bash
gcloud logging read \
  'resource.type=cloud_run_revision AND resource.labels.service_name="'"$SERVICE_NAME"'" AND textPayload:"POST /internal/cron"' \
  --project "$PROJECT_ID" --limit 10 --freshness=10m \
  --format='value(timestamp,textPayload)'
```

**E. 查 app 端有沒有真的 fire 過任何排程（每週一 21:00 之後可看）**

```bash
gcloud logging read \
  'resource.type=cloud_run_revision AND resource.labels.service_name="'"$SERVICE_NAME"'" AND textPayload:"Scheduled send"' \
  --project "$PROJECT_ID" --limit 20 --freshness=7d \
  --format='value(timestamp,textPayload)'
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

先設定三個變數（從 `.env` 讀這三個非密鑰值；開新 terminal 時要重設）：

```bash
cd ~/Documents/github/LineBot-Cloud-MultiAgent
PROJECT_ID="$(python3 scripts/envfile.py get --file .env GCP_PROJECT_ID)"
SERVICE_NAME="$(python3 scripts/envfile.py get --file .env CLOUD_RUN_SERVICE_NAME)"
REGION="$(python3 scripts/envfile.py get --file .env CLOUD_RUN_REGION)"
```

優先用這條（需要 gcloud 的 `beta` 與 `log-streaming` 元件；沒有的話先跑 `gcloud components install beta log-streaming`，按 Ctrl+C 結束）：

```bash
gcloud beta run services logs tail "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION"
```

### 即時只看重點

app 的每一行只出現一次（`… | INFO | linebot | …`；修正前的舊版本會再多印一行 `INFO:linebot:…`，同一筆內容）；下面的 `grep ' | linebot | '` 只留這個格式。

```bash
# 只看 app 自己的 log（去掉重複與 HTTP 存取紀錄）
gcloud beta run services logs tail "$SERVICE_NAME" --project "$PROJECT_ID" --region "$REGION" \
  | grep --line-buffered ' | linebot | '

# 每則訊息一行摘要：走 Colab 還是 API、路由由誰決定、nimble 機率、最後由哪個模型回答
gcloud beta run services logs tail "$SERVICE_NAME" --project "$PROJECT_ID" --region "$REGION" \
  | grep --line-buffered '] summary pref='

# 只看警告與錯誤（例如「AI 服務暫時不可用」背後的 fallback 失敗原因）
gcloud beta run services logs tail "$SERVICE_NAME" --project "$PROJECT_ID" --region "$REGION" \
  | grep --line-buffered -E ' \| (WARNING|ERROR) +\| linebot \| '

# 只看 Colab 開機、狀態變化與 nimble 路由
gcloud beta run services logs tail "$SERVICE_NAME" --project "$PROJECT_ID" --region "$REGION" \
  | grep --line-buffered -E ' \| linebot \| .*(Colab|colab|nimble)'

# 追單一則訊息的完整過程：把 9e4000b2 換成 summary／Input 那行開頭 [ ] 裡的編號
gcloud beta run services logs tail "$SERVICE_NAME" --project "$PROJECT_ID" --region "$REGION" \
  | grep --line-buffered '9e4000b2'
```

事後查（不是即時）同樣的內容，用下面〈只看每則訊息的摘要〉與〈只看錯誤 log〉的 `gcloud logging read` 指令。

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
  "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE_NAME}\" AND textPayload=~\"POST /webhook|Reply sent|Reply failed|Push failed|outcome unknown|Input:\"" \
  --project "$PROJECT_ID" \
  --limit 50 \
  --format='table(timestamp,textPayload)'
```

### 只看 Cloud Tasks 任務

每個任務結束時會記一行 `Task done {"kind": ..., "ok": ..., "ms": ...}`；排不進 queue 而改在 request 內處理時會有 `could not be queued` 的 ERROR。

```bash
gcloud logging read \
  "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE_NAME}\" AND textPayload=~\"Task done|could not be queued|not queued\"" \
  --project "$PROJECT_ID" \
  --limit 50 \
  --format='table(timestamp,severity,textPayload)'
```

### 只看每則訊息的摘要（模式、路由、模型）

每則有回覆（或回了錯誤訊息）的訊息，在回覆送出後記一行 INFO `summary`（`!new`、`!colab` 與被擋下的訊息不記）。原本逐次呼叫的 log 照舊；這一行把它們收成一句：

```text
[a615fdd7] summary pref=colab path=colab route=colab:nimble:9b agent=chat out=text think=off answer=colab:qwen3.8:27b-mtp-q4_K_M 2.8s answer_think=low delivery=reply total=3.4s router_p=[agent={chat:0.93,web_search:0.04,vision:0.02,image_gen:0.01} conf=0.84 gate=0.75 ok | format={text:0.97,voice:0.02,image:0.01} conf=0.90 ok | think p=0.05 (on>=0.85 off<=0.15) -> off] tiers=[colab:qwen3.8:27b-mtp-q4_K_M ok 2.8s]
```

| 欄位 | 意思 |
|---|---|
| `pref` | `MODEL_PATH_PREFERENCE`：`colab` 或 `api`；`COLAB_ENABLED=false` 時是 `api-only` |
| `path` | 實際寫出回答的是哪一種 tier：`colab`、`api`，沒有回答是 `none` |
| `route` | 誰決定 agent：`rules`（快速規則）、`colab:nimble:9b`（Colab 路由）、回答的 API 路由 tier（例如 `nvidia:nvidia/nemotron-3-super-120b-a12b`），或 `fallback`（路由都失敗，預設 chat） |
| `agent` / `out` / `think` | 路由結果：agent、輸出形式（`text`／`voice`／`image`）、是否需要思考 |
| `answer` | 回答的 tier 與該次呼叫秒數；工具迴圈呼叫多次時看最後一次呼叫（真正送出的那個回答），最後一次失敗就是沒有回答，即使前面幾輪成功。沒有回答時 `answer=none`，出錯時加 `error=<例外類別>(<最後的 provider 錯誤> <狀態碼>)` |
| `answer_think` | 回答的 tier 實際怎麼思考：`off`、`on`，或 Colab／OpenRouter 的等級。Colab chat 預設以 `COLAB_CHAT_THINK=low` 思考，所以 `think=off` 時仍可能是 `answer_think=low` |
| `prompt` | 只有 image_gen：寫英文 prompt 的 utility tier；失敗而改用原文時是 `raw`（`answer` 是 FLUX 那一層） |
| `tts` | 只有語音：語音送出是 `edge-tts`；合成失敗改傳文字是 `edge-tts(failed->text)`，語音送不出去改傳文字是 `edge-tts(send failed->text)` |
| `delivery` | 怎麼送出的：`reply`（免費）或 `push`（reply 確定失敗後改用；同一則訊息只要有一次是 push 就記 `push`）；什麼都沒送出時不出現 |
| `sent=no` | 回覆沒送達（reply 確定失敗，push 也失敗） |
| `total` | 從開始處理到這一行的秒數 |
| `router_p` | nimble 每個問題的機率（只列定義的選項，高到低、取兩位）、信心值對門檻（`gate`，`COLAB_ROUTER_CONFIDENCE`）的 `ok`／`fail`，以及 `needs_thinking` 的機率與 `COLAB_ROUTER_THINK_ON`／`_OFF` 得出的結果（`rules->on` 表示落在中間，由規則決定）。沒被採用時結尾是 `gate fail -> API router`。沒問 nimble（快速規則、API 路由先答）是 `n/a`；問了但沒用上時寫原因，例如 `n/a(nimble not READY)`、`n/a(nimble busy)` |
| `tiers` | 回答這一段走過的每個 tier 與結果：`ok`、`tools`（只要求呼叫工具、沒寫回答：工具迴圈中間的一輪）、`skip(<原因>)`、`timeout`、`rate_limited`、`empty`、`error(<類別> <狀態碼>)`，後面是秒數。常見的 `skip`：`skip(Colab session not READY)`、`skip(by design: url image)`（引用的圖片是 URL，Colab 只收內嵌圖片，依設計交給 API 視覺層，不算失敗）；生圖被 FLUX 安全過濾擋下是 `error(ContentFilteredError 200)`，會改用 FLUX.1-dev 再試一次。依 tier 分組：tier 按第一次用到的順序，各自的結果按發生順序，同一結果連續出現寫成 `x<次數>`（秒數相加），所以思考與不思考兩輪走過同一批 tier 時，看不出中間跑了哪一個。呼叫不只一次時加 `calls=<次數>` |

這一行只有 tier 名稱、固定的結果字樣與數字：不含使用者文字、prompt、URL 或錯誤訊息內容（使用者文字另外記在 `Input:` 那一行，已截斷）。啟動時另有一行 `Model roles (MODEL_PATH_PREFERENCE=...)` 列出每個角色的 tier 順序。

```bash
gcloud logging read \
  "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE_NAME}\" AND textPayload:\"] summary pref=\"" \
  --project "$PROJECT_ID" \
  --limit 50 \
  --format='value(timestamp,textPayload)'
```

只看 Colab 回答的（`path=api` 看 API）、或 nimble 沒被採用的：

```bash
gcloud logging read \
  "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE_NAME}\" AND textPayload:\"] summary pref=\" AND textPayload:\" path=colab \"" \
  --project "$PROJECT_ID" --limit 50 --format='value(timestamp,textPayload)'

gcloud logging read \
  "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE_NAME}\" AND textPayload:\"gate fail -> API router\"" \
  --project "$PROJECT_ID" --limit 50 --format='value(timestamp,textPayload)'
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

實際測試時，如果看到 `ERROR`、`Reply failed`、`Push failed`、`outcome unknown`、`/health` 非 `200`，就代表有功能要進一步排查。

---

## 不會自動清掉的東西

以下是正常保留，不影響你這個專案的乾淨部署：

- Cloud Build 歷史紀錄（原始碼壓縮檔在 `gs://<PROJECT_ID>-cb-src-usw1`，3 天後由 lifecycle 刪除）
- Cloud Logging 紀錄
- 你手動建立的其他 GCP 資源

這些不屬於 `deploy_cloud_run.sh` 的清理範圍。v2 的部署會把含所有密鑰的 `.cloudrun-env.generated.*.yaml` 一起上傳到 `gs://<PROJECT_ID>_cloudbuild/source/`，請用 `./scripts/provision_gcp.sh --step purge-cloudbuild --apply --yes-purge` 清掉（見〈從 v2 升級到 v3〉第 6 步）。

---

## 從 v2 升級到 v3

v2 使用的 `qwen/qwen3.5-397b-a17b`（2026-07-27 起）與 `nvidia/nv-embedqa-e5-v5`（2026-08-25 起）已被 NVIDIA 下架，呼叫會回 HTTP 410。下面是升級各步的細節；整體順序以上面的〈上線前檢查清單〉為準（例如第 6 步的密鑰輪替要在第一次部署**之前**做）：

1. **清理 `.env`**（會先備份成 `.env.bak.<時間>`，權限 0600；只印 key 名稱）：

   ```bash
   python3 scripts/envfile.py migrate-v3 --file .env
   python3 scripts/envfile.py check-v3 --file .env   # 應該通過
   ```

   它會刪掉指定已下架模型的 key、`ORCHESTRATOR_MODEL` / `ORCHESTRATOR_FALLBACK_MODEL`、`REQUIRE_REASONING_MODELS`，以及所有已不再讀取的舊 key（逐模型設定、reasoning 開關等，完整清單見 `src/config.py` 的 `REMOVED_ENV_KEYS`），讓模型角色表的預設生效；`MODEL_ROLES` 不會被改動，裡面若有已下架的 id 請自行修改。確認沒問題後可刪除備份檔（`.env.bak.*` 符合 `.env.*`，不會被上傳到 Cloud Build 或打包進 image）。
2. **（建議）跑 live gate**，用你的 key 驗證 bot 實際送出的每條路徑（全部約 NVIDIA 33 次、OpenRouter 15-18 次請求，其中 `--path tiers` 依預設角色表約 NVIDIA 11 次、OpenRouter 7 次）：

   ```bash
   .venv312/bin/python scripts/live_gate.py                 # 全部（含 tiers）
   .venv312/bin/python scripts/live_gate.py --path nvidia   # 只測 NVIDIA
   .venv312/bin/python scripts/live_gate.py --path roles    # 每個角色第一個啟用的非 Colab 層
   .venv312/bin/python scripts/live_gate.py --path tiers    # 角色表的每一層各呼叫一次
   .venv312/bin/python scripts/live_gate.py --only tier.router   # 只測 router 角色的每一層
   ```

   模型檢查的 tier 選項來自實際的角色表（含 `MODEL_ROLES`、`FORCE_TIER`）；角色表裡沒有或被 `FORCE_TIER` 停用的模型會顯示 SKIP；`--path colab` 需要一個 READY 的 Colab session（見〈Colab 層〉），沒有時顯示 SKIP 與原因。Ultra 思考檢查確認送出 `medium_effort` 且沒有 `reasoning_budget`；router 的 Ultra 檢查（`nvidia.ultra.router`）確認沒有送 `response_format`（Ultra 沒有結構化輸出，只靠 prompt 要求 JSON），回覆由 orchestrator 的容錯解析器讀取；視覺檢查（`nvidia.omni.vision` 15 秒、`nvidia.muse.vision` 35 秒）用紅色圓形 PNG，而且只給該層的時間上限，和 bot 的 fallback chain 一致；Nano Omni 忙碌時回 503 會 FAIL（bot 會立刻改用 muse-glimmer），用 `--only nvidia.omni.vision` 重跑即可；路由與工具檢查走 bot 的 fallback chain，超過該層 `timeout_s` 的模型一樣 FAIL；`nvidia.ultra.plain` 會標出回覆是否帶 Markdown / LaTeX（送 LINE 前會被清掉）。

   `--path tiers`（`all` 也包含）用 `.env` 的設定建出和 bot 完全相同的角色表（含 `MODEL_ROLES`、`MODEL_PATH_PREFERENCE`、`FORCE_TIER`），依呼叫順序對**每個角色的每一層**透過 bot 自己的 provider 與 payload 各送一次最小請求，檢查 id 為 `tier.<角色>.<第幾層>.<provider>:<model>`：

   - `router`：要回出 orchestrator 解析器讀得懂的繁中訊息路由決定；Colab 的 nimble 走 `SystemOneRouter`
   - `chat`：繁中短回答（思考依該層設定）；`thinking`：一題繁中推理題，開啟思考（budget 依該層），必須有推理內容
   - `vision`：紅色圓形 PNG 要被描述成紅色、圓形；`utility`：一段繁中記憶摘要（被 `max_tokens` 截斷算 FAIL）
   - `embed`：向量維度要等於 `EMBEDDING_DIMENSIONS`；`image`：要拿到圖片 bytes

   每項最多等該層的 `timeout_s` 再加 2 秒；請求會自動間隔，維持在 NVIDIA 每分鐘 40 次、OpenRouter 每分鐘 20 次以內；OpenRouter 一旦回 `free-models-per-day`（帳號每日上限），本次執行後面的 OpenRouter 檢查都不再送出，顯示 SKIP「daily cap」。Colab 層沿用 `--path colab` 的 session 探查，沒有 READY 的 session 時 SKIP；被 `FORCE_TIER` 或 `COLAB_ROUTER_ENABLED=false` 關掉的層 SKIP，缺 key 而停用的層 FAIL。最後另外印一張表：角色、第幾層、provider:model、PASS/FAIL/SKIP、延遲、備註（最多約 80 字，失敗時含 HTTP 狀態碼，不含 key 或完整回覆）。

   需要 Python 3.12+ 與專案依賴，請用「本機測試（Python 3.12）」建立的 `.venv312` 執行（系統的 `python3` 若是 3.11 以下會直接 SyntaxError）。

   任一檢查 FAIL 會回傳 exit code 1；輸出只有 PASS/FAIL/SKIP 與簡短證據，不會印出 key 或完整回覆。
3. **部署**：`./scripts/deploy_cloud_run.sh`
4. **遷移 episodic memory 向量**（有開 `FIRESTORE_ENABLED=true` 才需要）。舊的 e5 向量和新模型不在同一個向量空間，v3 的 `recall_memory` 會忽略它們，遷移前舊 episodes 不會被召回。用本機 ADC 執行：

   ```bash
   gcloud auth application-default login
   python3 scripts/migrate_embeddings.py           # dry run：只印數量
   python3 scripts/migrate_embeddings.py --apply   # 重新 embed；沒有摘要文字的舊 episode 會被刪除
   ```

   腳本以每批 16 筆、每分鐘最多 40 次請求重新 embed（模型取自 embed 角色；文字和 app 寫入新 episode 時相同：摘要加上對話片段，舊 episode 用它保存的 `messages`），只印數量，不印任何對話內容。若角色表無效（例如 `MODEL_ROLES` 的 embed 層仍是已下架的模型），會直接以 exit code 2 停止，請先修正。

   - 分類：已標記目前模型與維度的保留；其他有摘要文字的（沒有標記的舊 e5 向量、下架期間沒有向量的 episode、其他模型或維度）就地重新 embed 並更新 `embedding` / `embedding_model` / `embedding_dim`（`expires_at` 不變，90 天 TTL 照舊）；沒有摘要文字的，整筆 episode 刪除。
   - 某一批 embedding 失敗時該批不會被改動、腳本以 exit code 1 結束，重跑即可補上；已完成的會被判定為保留，重跑是安全的。
   - 它和線上的 bot 共用 NVIDIA 每分鐘 40 次的額度（額度屬於 API key），bot 忙碌時請在離峰執行或加上 `--max-rpm 20`。
   - 沒跑遷移時，舊向量仍留在索引裡但不會被召回；有開 TTL 的話，最後一批 e5 向量（2026-08-25 以前寫入）約在 2026-11-23 前隨 episode 一起被刪除。設計與回滾方式見 [ADR-0011](docs/adr/0011-embedding-migration.md)。

   同一時間建議也照〈啟用 Firestore 持久化記憶〉第 6 步，替大欄位加上單欄位索引豁免（降低儲存用量；不影響功能）。
5. **確認**：`/health` 的 `ready_for_webhook` 是 `true`；`/health/detail`（帶 `X-Cron-Token`）的 `roles` 裡沒有 `dead: true`、`config_warnings` 是空的，log 沒有 `Config:` 開頭的警告。
6. **輪替密鑰並清掉舊的上傳檔**：v2 的部署把含所有 `.env` 值的 `.cloudrun-env.generated.*.yaml` 上傳到 Cloud Build 的原始碼 bucket，舊版部署也可能把 `INTERNAL_CRON_SECRET` 寫進本機 `~/.config/gcloud/logs`。建議：
   - 輪替 LINE channel secret / access token、`OPENROUTER_API_KEY`、`NVIDIA_API_KEY`、`TAVILY_API_KEY`、`INTERNAL_CRON_SECRET`、`INTERNAL_TASKS_SECRET`，更新 `.env` 後重新部署（步驟見〈密鑰輪替〉）；
   - 刪除舊的原始碼壓縮檔（會先關閉 soft delete，否則 7 天內仍可復原）：

     ```bash
     ./scripts/provision_gcp.sh --step purge-cloudbuild            # dry-run：印出指令與檔案數
     ./scripts/provision_gcp.sh --step purge-cloudbuild --apply --yes-purge
     ```

   - 刪除 repo 根目錄殘留的 `.cloudrun-env.generated.*`（部署腳本發現時會警告），以及不需要的本機 gcloud log：`rm -rf ~/.config/gcloud/logs`。
7. **在 LINE Developers Console 開啟 Webhook redelivery**（Cloud Tasks 與 `webhookEventId` 去重已上線）。

---

## 密鑰輪替

`.env` 是唯一的來源；Cloud Run 讀的是 `linebot-dotenv` 的某個**釘住的版本**。

1. 在各服務的主控台重新產生密鑰（LINE channel secret / access token、NVIDIA、OpenRouter、Tavily），或用 `openssl rand -hex 32` 產生新的 `INTERNAL_CRON_SECRET` / `INTERNAL_TASKS_SECRET`。
2. 更新 `.env`。
3. `./scripts/deploy_cloud_run.sh`：偵測到內容改變 → 新增一個 secret 版本 → 部署釘住新版本 → Cloud Scheduler job 換成新的 cron token → smoke check（用新的 LINE channel secret 簽 webhook）→ 通過後銷毀更舊的版本。
4. 確認 `/health` 正常後，再到舊服務撤銷舊密鑰（LINE 的 channel secret 重新產生後舊的立即失效，所以第 1～3 步要連續做）。

注意：

- 部署腳本只印 key 名稱，不會印出任何值或 hash；blob 只經由 pipe 傳給 `gcloud secrets versions add --data-file=-`，不會寫成檔案；讀取最新版本時關閉 gcloud 的檔案 log，payload 不會進 `~/.config/gcloud/logs`
- 想手動確認目前版本：`gcloud secrets versions list linebot-dotenv --filter='state!=DESTROYED'`
- 同一帳單帳戶的 Secret Manager 免費額度是 **6 個有效版本**（enabled 和 disabled 都算）與每月 10,000 次存取；請不要改用 disable 代替銷毀，也不要建立有 `--version-destroy-ttl` 的 secret
- Colab 層的 token（`linebot-colab-adc`）不在 `.env` 裡：換新的方式是重跑 `./scripts/mint_colab_adc.sh --apply`，再部署（見〈Colab 層〉的「重新產生 token」）
- 換掉 `INTERNAL_CRON_SECRET` 之後，Cloud Scheduler job 只帶新的 token：回滾到換之前的 revision 時，job 的 header 也要一起換回舊值（見〈回滾〉）

---

## 預算警報

預算與警報本身免費（只有 Pub/Sub 通知要錢）；帳單幣別是 **TWD**，金額要用 TWD 輸入。`./scripts/provision_gcp.sh --step budget` 會印出同樣的步驟。

1. 開啟 `https://console.cloud.google.com/billing/budgets?project=<專案>` → **建立預算**
2. 名稱 `linebot-monthly`；範圍：專案 `<專案>`、所有服務；**保留「包含抵免額」**（Always Free 的用量是以抵免額抵銷的）
3. 金額：指定金額 **30 TWD**（約 US$1），每月
4. 門檻：實際支出 **50%、90%、100%**，以及預測支出 **100%**
5. 通知：寄 email 給帳單管理員與使用者；不設 Pub/Sub、**不設花費上限**（Cloud Run 的 spend cap 會直接暫停 bot，而且以扣抵免額前的金額計算）

---

## 回滾

每個 revision 都釘住自己的 image 與 `linebot-dotenv` 版本，所以回滾只要切流量：

```bash
gcloud run revisions list --service "${SERVICE_NAME}" --project "${PROJECT_ID}" --region "${REGION}"
gcloud run services update-traffic "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" --region "${REGION}" \
  --to-revisions=<上一個 revision>=100
```

- 部署腳本保留最新 `DEPLOY_KEEP_REVISIONS`（預設 2）個 Ready 的 revision（部署失敗、從未 Ready 的 revision 不算，所以不會把上一個正常的 revision 擠掉），以及所有 region 的 revision 釘住的 secret 版本（再加上最新 max(2, `DEPLOY_KEEP_REVISIONS`) 個作為下限）；更舊的 revision 會被刪除、它的 secret 版本會被銷毀，無法再回滾
- `ar-cleanup-enforce` 之後，Artifact Registry 只留最新 2 個 image，其他的一天後刪除（不管 revision 還用不用）；Cloud Run 只在 revision 有流量時保留自己的 image 副本。所以回滾只到最新 2 個 image：把 `DEPLOY_KEEP_REVISIONS` / `DEPLOY_KEEP_IMAGES` 調到 2 以上不會多出回滾空間，失敗的部署或只 build 不 deploy 的 image 也會佔掉一格。部署發現保留中的 revision 的 image 落在最新 2 個之外時會警告
- 回滾後下一次 `./scripts/deploy_cloud_run.sh` 會重新把流量切到最新 revision（`--to-latest`）
- 只想回到舊設定：把 `.env` 改回去再部署（會新增一個版本）；已銷毀的 secret 版本無法復原
- **Cloud Scheduler job 的 `X-Cron-Token` 不跟著 revision 走**：它一直是最近一次部署時 `.env` 的 `INTERNAL_CRON_SECRET`。回滾到換掉這個密鑰**之前**的 revision 後，每分鐘的 `/internal/cron` 都會 403（排程訊息、群組訊息緩衝的逾時寫入、每日下架檢查與 Colab 控制器都會停）。這時請改成用舊的程式碼重新部署（新 revision 拿到現在的 `.env`，不必回滾），或者回滾後把 job 的 header 換成那個 revision 的舊密鑰：先暫時把 `.env` 的 `INTERNAL_CRON_SECRET` 改回舊值，再執行下面的指令（關閉 gcloud 的檔案 log；值經由 `$(...)` 傳入，不會留在 shell history）。下一次部署會再把 header 換成當時 `.env` 的值。

  ```bash
  CLOUDSDK_CORE_DISABLE_FILE_LOGGING=true gcloud scheduler jobs update http "${SERVICE_NAME}-cron" \
    --project "${PROJECT_ID}" --location "${REGION}" \
    --update-headers="X-Cron-Token=$(python3 scripts/envfile.py get --file .env INTERNAL_CRON_SECRET)"
  ```

---

## Endpoints

| Endpoint | Method | 用途 |
| --- | --- | --- |
| `/health` | `GET` | 公開、精簡：`status`、`ready_for_webhook`、`version`（`APP_VERSION` 環境變數，未設定為 `dev`） |
| `/health/detail` | `GET` | 各角色各模型層（啟用 / 停用原因、健康狀態、最後錯誤、下架日）、provider 額度、config 警告、每日下架檢查、tasks mode、記憶 / 儲存統計、網路搜尋（`web_search`：Tavily 是否設定、模式與本程序用掉的 credits、冷卻秒數、最後錯誤，不含 key）、Colab（`colab`：狀態與起始時間、是否 READY、最後錯誤、CU 餘額與每小時用量、今日用量；不含 tunnel 網址、bearer 或 Colab id）、排程 id 與 next_run（不含訊息文字）；`X-Cron-Token` 驗證（`INTERNAL_CRON_SECRET`），不符或密鑰為空一律 403 |
| `/webhook` | `POST` | LINE webhook（HMAC-SHA256 驗簽） |
| `/internal/cron` | `POST` | Cloud Scheduler 每分鐘觸發排程訊息與群組訊息緩衝的逾時寫入，並在每個台北時間日第一次排入 `eol` 下架檢查；`X-Cron-Token` 共用密鑰驗證（`INTERNAL_CRON_SECRET`） |
| `/internal/tasks/{kind}` | `POST` | Cloud Tasks 回呼（`event` / `passive_flush` / `cache_image` / `compact` / `eol` / `colab`）；`X-Tasks-Token` 共用密鑰驗證（`INTERNAL_TASKS_SECRET`），驗證通過後一律回 200 |
| `/internal/colab/register` | `POST` | Colab VM 回報 `ready` / `failed` / `stopping`；公開路由，以 `X-Colab-Signature`（HMAC，由 `COLAB_SHARED_SECRET` 與該次 session 導出的金鑰）驗證，時間差超過 300 秒、不是目前 session、簽章錯誤或重送一律 403；body 上限 8 KB；`COLAB_ENABLED=false` 時 404。見 ADR-0010 |

---

## 成本控制

| 資源 | 控制方式 |
| --- | --- |
| LLM 呼叫 | provider 額度（NVIDIA 40 次/分、10,000 次/天，含 embedding；OpenRouter 免費 20 次/分、`OPENROUTER_FREE_DAILY_LIMIT` 次/天）與模型冷卻，429 時自動 fallback（OpenRouter 單一模型的上游 429 只冷卻該模型；帳號層級的 `free-models-per-day` / `per-min` 才冷卻所有 `:free` 模型 10 分鐘）；每次回應的 log 帶 prompt / completion / reasoning token 數 |
| LINE push | 一律先用免費的 reply，reply 確定失敗才 push；程式不計數也不限制，只受 LINE 的每月額度限制（群組依人數計）；用量看摘要 log 的 `delivery=push` 與 LINE Official Account Manager |
| 網路搜尋 | 只用 Tavily（basic 深度每次 1 credit，關閉 auto_parameters 避免自動升級成 2 credits；讀網頁每 5 頁 1 credit，且只讀本機讀不到的頁面）；Tavily 429 依 Retry-After 冷卻、額度用完冷卻 1 小時。沒有 app 內配額（`WEB_SEARCH_MONTHLY_QUOTA` 已停用）；Tavily 回報的 credits 顯示在 `/health/detail` |
| GCS 媒體 | Signed URL 48 小時過期；bucket 3 天 lifecycle 規則刪除（部署腳本驗證），傳送失敗的物件立即刪除 |
| Cloud Tasks | 每則訊息約 2 次 billable operation（建立 + 投遞），每月前 100 萬次免費；payload 超過 32 KB 改在 request 內處理 |
| 使用者請求 | 每人滑動視窗 rate limit |
| Colab | 只在需要時開機（`MODEL_PATH_PREFERENCE=colab` 下通過限流與擋訊息檢查的觸發訊息，每分鐘最多記錄一次需求；`api` 下只有 `!colab on`），閒置 `COLAB_IDLE_TTL_SECONDS`（900）或滿 `COLAB_MAX_SESSION_SECONDS`（39600）釋放；CU 餘額低於 `COLAB_MIN_CU_BALANCE`（10）或當日用量達 `COLAB_DAILY_CU_CAP`（30）不開機；`!colab off` 立即停用 |
| Cloud Run | `max-instances=1`、`min-instances=0`（scale-to-zero），request-based billing 走免費額度；背景工作改走 Cloud Tasks，不需要 instance-based billing |
| Secret Manager | 一個 `linebot-dotenv`（automatic replication 算一個 location），只在 `.env` 改變時新增版本，部署後銷毀舊版本；`COLAB_ENABLED=true` 時另有 `linebot-colab-adc`（只在重新 mint token 時新增版本，同樣銷毀沒有 revision 釘住的舊版本）。所有 secret 合計 6 個有效版本以內免費 |
| Cloud Build | 原始碼放在 us-west1 的 `gs://<專案>-cb-src-usw1`（3 天刪除、無 soft delete）；以 `linebot-builder` 執行 |
| Artifact Registry | 清理政策保留最新 2 個 image、刪除 1 天以上的其他 image（先 dry-run；強制執行後回滾只到最新 2 個 image）；部署腳本也會刪掉超過 `DEPLOY_KEEP_IMAGES` 且沒有 revision 使用的 image |

免費額度估算（每天約 100 則觸發、300 則群組被動訊息；皆以帳單帳戶計）。網路搜尋只用 Tavily，沒有 VM、外部 IP 或 VPC 流量，整體是 **$0**。

| 項目 | 每月用量 | 免費額度 | 費用 |
| --- | --- | --- | --- |
| Cloud Run requests | 約 64K | 2M | $0 |
| Cloud Run vCPU-s | 約 60–120K（Cloud Tasks 讓等待模型的時間也計費） | 180K | $0 |
| Cloud Tasks | 約 14K 次 operation | 1M | $0 |
| Cloud Scheduler | 1 個 job | 3 個 | $0 |
| Secret Manager | `linebot-dotenv` ≤ 3 個有效版本（平時 2 個；部署中或部署失敗後 3 個，超過 3 個會警告）；開啟 Colab 時 `linebot-colab-adc` 再加 1–2 個（最多 3 個），合計不超過 6 個；每次冷啟動約 2–3 次存取 | 6 個版本 / 10K 次存取 | $0 |
| Firestore（default DB） | 每天約 2K 讀、< 1K 寫 | 每天 50K / 20K | $0 |
| GCS 媒體（US-WEST1 / CENTRAL1 / EAST1 STANDARD、3 天 lifecycle） | 很少 | 5 GB | $0 |
| Cloud Build | 每次部署約 3–4 分鐘 | 2,500 分鐘 | $0 |
| Artifact Registry | 2 個 image | 0.5 GB | $0 |
| 預算警報 | — | 免費 | $0 |
| Cloud Run 對外流量 | 約 0.2–0.5 GiB | 1 GiB（北美） | $0 |
| Tavily（唯一的搜尋服務） | 每次搜尋 1 credit、讀網頁每 5 頁 1 credit | 每月 1,000 credits 或 keyless | $0 |

---

## 運行注意事項

- `/health` 可能回 `200` 但 `ready_for_webhook=false`，要看 payload 不能只看 HTTP status。ready 的條件是 LINE 憑證齊全，且 chat 角色至少有一個不依賴 Colab 的可用層（`NVIDIA_API_KEY` 或 `OPENROUTER_API_KEY` 其一即可）；原因列在 `/health/detail` 的 `warnings`。
- **回覆期限（reply window）**：每個請求的一般回答（不需思考）都要在 LINE 事件時間 + `REPLY_WINDOW_SECONDS`（90 秒）內完成，私訊與群組都一樣，不論由 Colab 或 API 層回答；Colab 是第一個可用層時最多用到期限前 20 秒，留給 API 備援層。需要思考的回答與生圖，不論私訊或群組，都可以用到 `TURN_DEADLINE_SECONDS`（從開始處理起算 240 秒）；開始回答時期限只剩不到 `MIN_TIER_SECONDS` 的一般回答（延遲或重送的事件，或路由太慢用掉了期限）也改用這個 turn 期限。LINE 的 reply token 只保證約 1 分鐘有效，但送出一律先試免費的 reply，不會因為已經過了多久就略過；LINE 確定拒絕（4xx，例如 reply token 無效或過期，或請求根本沒送出）才改用 push，結果不明（送出後逾時、5xx）視為已送出、不再 push，push 不計數也不設上限。期限內每一層模型只拿「自己的上限」與「剩餘時間」中較小的那個，思考層合計最多只用剩餘時間的一半；網路搜尋必須在期限前 20 秒結束。生圖：FLUX.2 klein 被安全過濾擋下時改用 FLUX.1-dev 再試一次，仍被擋下就用繁中告訴使用者是圖片模型的安全過濾擋下、請換個描述，不回一般的失敗訊息。Colab 還沒 READY 時一律略過，不會讓回覆等它。
- 語音和圖片生成需要 `GCS_BUCKET_NAME`，沒設定時文字回覆仍正常。
- 目前只支援**語音輸出**，不支援把使用者傳來的音訊自動轉文字。
- 主 prompt 記憶是「1 份長期摘要 + 上次壓縮後的近期文字訊息」（累積到 6 則開始壓縮；上次壓縮不到 `COMPACTION_DEBOUNCE_SECONDS`＝120 秒時，等下一則訊息再壓縮，累積到安全上限的一半則立即壓縮）。近期訊息依 LINE 收到的時間排序，壓縮完成時只移除被摘要的那幾則（依訊息 id），壓縮期間進來的訊息不會遺失。預設使用 **in-memory backend**（per-instance），重部署 / cold start 會重置；要長期保存請啟用下方的 **Firestore backend**。
- 發訊息者的 user profile facts（最新 10 筆、500 字內）會以 `[使用者 profile，僅供參考]` 區塊提供給 chat / vision / web_search（路由與生圖提示改寫不會看到）；profile 在程序內快取 10 分鐘。顯示名稱與 last seen 在回覆送出後才更新，且只在名稱改變或距上次寫入滿 24 小時才寫入。
- `!new` 會清空當前 user / group / room 的近期記憶、長期摘要與 episodic recall 記錄（`!new` 之後才送出的訊息會保留），並回覆 `Let's start a new chat!`；跨聊天室的 `linebot_user_profile/{user_id}` 不會被 `!new` 刪除，需要時請在 Firestore 手動刪除。
- 長期摘要內容會直接寫進 log（loaded / updated / cleared），請自行評估 Cloud Logging 的敏感資訊風險。
- 引用訊息快取、模型健康狀態與每日下架檢查結果仍是 **in-memory / per-instance**；重部署後會重置，多 instance 之間也不共享。
- 模型回 HTTP 410（已下架）時，該模型在本次程序內停用；回 HTTP 404 時停用 6 小時後自動重試（每分鐘的 cron 可能讓同一個 instance 活很多天，所以 404 不會永久停用）。每日下架檢查看到模型目錄回 410 也會先停用它，目錄回 200 則提早解除停用。`/health/detail` 各層的 `health` / `dead` / `eol` / `last_error` 可看到原因；OpenRouter 的 404（no endpoints）則冷卻 10 分鐘。
- 網路搜尋使用 Tavily（有 `TAVILY_API_KEY` 用 key，沒有則在 `TAVILY_KEYLESS_ENABLED=true` 時用 keyless 模式，見〈網路搜尋：Tavily〉）→ 不行時，模型依既有知識回答，開頭加上「（目前無法使用搜尋服務，以下依模型既有知識回答，可能不是最新資訊）」，不回固定錯誤訊息。
- 問題裡的「今天 / 最近 / 這個月 / 今年」這類時間詞會換成對應的時間範圍（day / week / month / year）；指過去一段日曆期間的詞依台北時間換成涵蓋整段期間的最短範圍（昨天 → week、上週 → month、上個月 → year、去年不設範圍），昨天、上個月、去年還會在查詢後補上明確日期（2026 年問「去年」會加上「2025年」）。Tavily 的 `country` 只在一般主題時送出。查無結果時拿掉主題、時間與國家篩選重搜一次（最多一次）；Tavily 失敗（連不上、拒絕、限流，而不是查無結果）時不重搜，免得再等一次逾時。
- 搜尋到結果後，會讀取前 `SEARCH_FETCH_TOP_N`（3）筆的網頁全文，所有網頁共用 `SEARCH_FETCH_BUDGET_SECONDS`（6 秒），時間到就用已讀到的；使用者傳來的網址也先在本機讀取，讀不到的才交給 Tavily extract；還是讀不到時，回覆開頭註明連結無法開啟，模型不會猜測該網頁內容。整個搜尋階段必須在回覆期限前 20 秒結束，留給模型回答；讀網頁會在搜尋階段結束前 0.5 秒停下，所以已讀到的網頁一定會用上，沒讀完的會取消。可以用整個 turn 期限的回答（思考模式的回答，不論私訊或群組，或開始時回覆期限只剩不到 `MIN_TIER_SECONDS` 的回答；reply token 失效後改用 push 送出）在回覆期限不夠搜尋時，改以它自己的期限計算，不會因此跳過搜尋。
- 本機讀網頁會擋掉所有內部位址：只允許 http / https 的 80 / 443 埠；`localhost`、`metadata.google.internal`、`*.internal`、`*.local` 直接拒絕；解析出的每個 IP 都必須是公開位址（私有網段、loopback、169.254.169.254 等 link-local、CGNAT 100.64.0.0/10、multicast、IPv4-mapped / 6to4 包著的內部位址都拒絕），連線直接連到檢查過的 IP；redirect 最多 5 次、每次重新檢查；只讀 HTML / 純文字，每頁最多 `SEARCH_FETCH_MAX_BYTES`（2 MB）。
- 續問（例如問完台北天氣再問「那高雄呢」）會先由 utility 模型依最近對話與調度摘要改寫成完整搜尋字串；只有短而獨立的查詢（中文少於 8 字、英文少於 4 個詞）直接照原文搜尋。
- 排程訊息需要同時設定 `SCHEDULED_MESSAGES_ENABLED=true`、`SCHEDULED_GROUP_ID`、`INTERNAL_CRON_SECRET`，以及至少一個排程 job。觸發機制：Cloud Run 不再用 in-process timer，改由 Cloud Scheduler 每分鐘 POST `/internal/cron`，因此 Cloud Run 可以維持 `min-instances=0`、走免費額度。
- 群組訊息緩衝、引用訊息快取、事件去重與「摘要是否已排入」都是 **in-memory / per-instance**，依賴 `max-instances=1`；緩衝會在達到門檻（或逾時後有新訊息）、機器人回答該群組之前、cron 逾時掃描或 instance 關閉時寫入；寫入失敗的訊息會留在緩衝等下次再寫。
- Colab 的忙碌閘門（每個 Colab 模型一個 `asyncio.Semaphore(1)`：模型正在回答時直接跳過，不排隊）與斷路器（qwen3.8 連續 3 次失敗跳過 120 秒；nimble 的失敗只讓 nimble 冷卻，不會跳過 Colab 對話）都在程序內，前提是 `max-instances=1`；session 狀態本身存在 Firestore `{prefix}_colab/state`，冷啟動後讀一次。
- `!colab on` / `off` / `status` 只限 `LINE_ADMIN_USER_IDS`（其他人回「此指令僅限管理員」），在群組不需 @ 也會觸發；`!whoami` 只在私訊回覆自己的 userId，在群組只提醒改用私訊。
- Prompts 從 `prompts/*.md` 載入，調整路由或語氣不需要改 Python 程式碼。
- LINE 不會渲染 Markdown：每則送出的文字（回覆、圖片說明、朗讀的文字）都會先去掉 `**粗體**`、`##` 以下的標題、程式碼圍欄那兩行、行內程式碼、LaTeX（`\( … \)`、`\[ … \]`、`$$…$$`）、`[文字](http…)` 連結語法與表格分隔列（網址、emoji 與程式碼內容保留，圍欄內的程式碼原樣保留，`src/utils/line_text.py`），再做 OpenCC 簡轉繁。

---

## Colab 層

Colab A100 上的 Ollama 0.35.0 跑 `qwen3.8:27b-mtp-q4_K_M`（對話、視覺、思考）與 `nimble:9b`（`/v1/systemone` 路由），所有使用者都先走 Colab（Colab 條款禁止提供網路服務與讓第三方使用 runtime，這個風險已由你接受，見 [ADR-0010](docs/adr/0010-colab-ollama-tier.md)）。**任何訊息都不會等 Colab**：READY 前一律由 NVIDIA → OpenRouter 立即回覆。READY 後由 Colab 先回答，期限和其他請求一樣是事件時間 + `REPLY_WINDOW_SECONDS`（90 秒），Colab 層最多用到期限前 20 秒；送出一律先試 reply，reply token 已失效才改用 push。bot 放棄一個 Colab 請求（時間用完）時會關閉連線，VM 上的 auth proxy 隨即中止對應的 Ollama 請求，GPU 不會繼續算沒人要的回答；`auth_proxy.log` 的 `cancelled after N s` 紀錄從**下一次 Colab 開機**起才有（VM 的檔案在開機時上傳），也可以用它確認 Cloudflare 有把斷線傳到 VM（見 [colab/README.md](colab/README.md)）。開機、保活、釋放都由每分鐘的 cron 決定，慢的步驟走 `colab` Cloud Tasks；VM 端的程式與疑難排解見 [colab/README.md](colab/README.md)。

### 第一次設定

這些步驟需要你操作：token 只能由人在瀏覽器同意後產生，第一次實跑也要有人看著。前提：已跑過 `./scripts/provision_gcp.sh`（`linebot-runtime` 與 `linebot-dotenv` 存在），而且 Firestore 已啟用（〈啟用 Firestore 持久化記憶〉）。

1. **準備專用 Google 帳號**：只持有 Colab 方案（要能用 A100、有 CU 餘額）、在這個 GCP 專案**沒有任何角色**，也不拿來做其他 gcloud / colab-cli 登入。原因：gcloud 產生的 token 一定帶 `cloud-platform` scope，專案 Owner 的 token 放進 Secret Manager 就等於 Owner 權限；而同一個 Cloud SDK OAuth client 每個帳號最多 100 個有效 token，超過時最舊的會被默默撤銷（可能就是 bot 用的那個）。
2. **產生並儲存 token**：用你平常部署用的 gcloud 登入執行（Secret Manager 的操作以它的身分進行），Colab 專用帳號只在瀏覽器裡登入。

   ```bash
   ./scripts/mint_colab_adc.sh           # dry-run：檢查權限，印出會做的事，不產生任何東西
   ./scripts/mint_colab_adc.sh --apply   # 瀏覽器同意 → 存成 linebot-colab-adc 的新版本 → 授權 linebot-runtime
   ```

   `--apply` 在暫存的 `CLOUDSDK_CONFIG`（`mktemp -d`，權限 700，不會蓋掉你本機的 ADC 與 gcloud 登入）執行 `gcloud auth application-default login --no-launch-browser --disable-quota-project`（scope：openid、cloud-platform、userinfo.email、colaboratory）。用**只登入專用帳號的無痕視窗**打開它印出的網址，同意後把授權碼貼回終端機。腳本確認檔案是帶 refresh token 的 `authorized_user` 憑證後，經 stdin 存成 secret 的新版本（第一次會建立 secret，automatic replication），再執行 `./scripts/provision_gcp.sh --step colab-secret --apply`，結束時刪除暫存目錄（有 `shred` 時先覆寫）。token 內容不會印出；權限、API 或 gcloud 登入的問題會在瀏覽器登入**之前**就停止，不會白白多產生一個 token。專案與 `provision_gcp.sh` 一樣只取 `.env` 的 `GCP_PROJECT_ID`（env 檔壞掉或 `GCP_PROJECT_ID` 是空的就停止，不會用 `gcloud config` 的預設專案）。
3. **設定 `.env`**：
   - `FIRESTORE_ENABLED=true`（session 狀態存在 `{prefix}_colab/state`）
   - `COLAB_SHARED_SECRET`：至少 32 字元的隨機字串，例如 `python3 -c "import secrets; print(secrets.token_urlsafe(48))"`。每次 session 的註冊金鑰與 Ollama bearer 都由它和 session id 導出，其他密鑰不會存在任何地方；換掉它會讓正在跑的 session 失效
   - `LINE_ADMIN_USER_IDS`：可以使用 `!colab` 的 LINE userId（逗號分隔）；私訊 bot `!whoami` 就能看到自己的 userId（可以先部署一次、傳 `!whoami`，再補上）
   - `COLAB_ADC_PATH` 維持預設 `/secrets/colab/adc.json`（部署把 token 掛載在這裡；不一致時部署會警告）
   - `COLAB_ENABLED=true`，加上第一次實跑用的 `COLAB_DRY_RUN=true` 與 `MODEL_PATH_PREFERENCE=api`：這樣只有 `!colab on` 會開機，一般訊息照舊走 API、不會觸發開機
   - Cloud Tasks 必須可用（`INTERNAL_TASKS_SECRET` 有值、`TASKS_MODE` 不是 `inline`）：控制器的 `colab` 任務一律排進 Cloud Tasks、從不在 cron request 內執行；部署會另建專用 queue `CLOUD_TASKS_COLAB_QUEUE`（預設 `linebot-colab`，2 個並行名額），不佔 LINE 事件的 queue
   - 選用：`COLAB_TUNNEL_MODE=named` 加 `COLAB_TUNNEL_HOST` / `COLAB_TUNNEL_TOKEN`（自己的 Cloudflare 網域；空白時只接受 `https://<名稱>.trycloudflare.com`）、`COLAB_ROUTER_ENABLED`（nimble 路由開關）、`COLAB_ROUTER_CONFIDENCE` / `COLAB_ROUTER_THINK_ON` / `COLAB_ROUTER_THINK_OFF`（路由門檻，預設 0.75 / 0.85 / 0.15 是暫定的嚴格值，要用 `scripts/evals/routing_eval.py` 重新調整）、`COLAB_CHAT_THINK` / `COLAB_DEEP_THINK`、`COLAB_ACCOUNT_DEDICATED=true`（帳號真的只給 bot 用時，帳號上不認得的 runtime 也會被釋放）

   缺 Firestore、`COLAB_SHARED_SECRET` 太短、沒有 `INTERNAL_CRON_SECRET`、讀不到 token 或在 Cloud Run 上沒有 Cloud Tasks 時，服務照常啟動但 Colab 層維持停用，原因記在 WARNING log 與 `/health/detail` 的 `colab.reason`。
4. **部署**：`./scripts/deploy_cloud_run.sh`。輸出應有 `Mounting linebot-colab-adc version N at /secrets/colab/adc.json`；部署後 `/health/detail` 的 `colab.enabled` 是 `true`。
5. **CPU 乾跑**（`COLAB_DRY_RUN=true`）：控制器改申請 CPU runtime（`Variant.DEFAULT`），VM 以 `DRYRUN=1` 啟動，跳過 GPU、Ollama 與模型，由一個假的 Ollama 通過就緒檢查，所以整條控制流程（assign、上傳、silent env 檔、launcher 標記、註冊、READY、heartbeat、釋放、對帳）都會真的跑一遍，只花極少 CU。假的 Ollama 對推論一律回忙碌 503，這段期間使用者照常由 API 回覆。
   1. 私訊 bot `!colab on`，每隔一兩分鐘 `!colab status`：`IDLE` → `ASSIGNING` → `BOOTING` → `READY`（通常幾分鐘內）；`!colab status` 也會顯示 CU 餘額與每小時用量。
   2. `.venv312/bin/python scripts/live_gate.py --path colab`：`colab.probe`（經 tunnel 與 auth proxy 跑 bot 的 readiness probe）應為 PASS，其他三項 SKIP（乾跑沒有推論）。gate 以本機 ADC 讀 Firestore 的 session 文件，先執行 `gcloud auth application-default login`，用**能讀這個專案 Firestore 的帳號**（通常是你自己），不是 Colab 專用帳號；bearer 由 `.env` 的 `COLAB_SHARED_SECRET` 推導，不會印出。
   3. `!colab off`：`DRAINING` → `IDLE`，runtime 被釋放。另外建議再跑一次、不下 `!colab off`，確認控制器在 `COLAB_IDLE_TTL_SECONDS`（15 分鐘）後自己釋放。
   4. 確認沒有留下 runtime：`!colab status` 是 `IDLE`；用專用帳號在 Colab 網頁的「管理工作階段」看不到 runtime。
6. **一次真正的 A100**：兩個模型的 digest 都已釘在 `colab/versions.json`（路由模型在 2026-09-30 換成 `nimble:9b`，見 ADR-0010）。qwen3.8 與釘住的不一致、或沒有完全載入 GPU 時，每次開機都會失敗並自我釋放，連續 3 次後 Colab 層停用；nimble 沒拉到、不一致或沒有完全載入 GPU 時照常 READY（載入 nimble 時若踢掉了 qwen3.8，VM 會卸載 nimble、重新載入 qwen3.8），但這個 session 不用 nimble，路由改由規則與 API 路由決定（見 [colab/README.md](colab/README.md)〈釘住的 digest〉與〈VRAM〉）。`COLAB_DRY_RUN=false`、部署、`!colab on`，約 4–8 分鐘後 READY，再跑 `.venv312/bin/python scripts/live_gate.py --path colab`，四項都要 PASS：readiness probe（兩個模型都完全在 VRAM）、qwen3.8 的繁中回覆（think=`COLAB_CHAT_THINK`）、紅色圓形 PNG 的視覺、nimble 路由決定。記下 assign 到 READY 的時間、`!colab status` 的 CU 每小時用量（約 5.4）、`/health/detail` 的 `colab.gpu`（應是 40 GB 的 A100，不是 80 GB 的 high-mem；`chat_on_gpu`、`router_on_gpu`、`used_mib` 是 warm-up 後的 VRAM，照 colab/README.md〈VRAM〉記錄），再用 `scripts/evals/routing_eval.py` 重新調整 `COLAB_ROUTER_*` 門檻；可以的話在群組同時送兩則訊息確認忙碌閘門會改走 API，最後 `!colab off`。
7. **對所有人開啟**：A100 那次正常後，把 `MODEL_PATH_PREFERENCE` 改回 `colab`（預設，也可以刪掉這一行）並部署。之後通過限流的觸發訊息會讓控制器開機（CU 餘額扣掉一次開機加一次閒置（約 2.7 CU）仍 ≥ `COLAB_MIN_CU_BALANCE`、當日用量加上它仍 ≤ `COLAB_DAILY_CU_CAP` 時），閒置 15 分鐘或滿 11 小時釋放。

關閉：`!colab off` 立即生效（session 會釋放，`!colab on` 恢復）；要完全停用就設 `COLAB_ENABLED=false` 再部署，部署會同時卸下 token 的掛載。

### 重新產生 token

token 在被撤銷、6 個月沒用或被 100 個上限擠掉時失效。控制器看到 `invalid_grant` 或 `Reauthentication is needed` 會進入 `DISABLED` 並記錄 ERROR（`!colab status` 會寫原因），回覆不受影響（全部走 API 層）。處理方式：

1. `./scripts/mint_colab_adc.sh --apply`（同樣用專用帳號；新增一個版本）
2. `./scripts/deploy_cloud_run.sh`（釘住最新的 enabled 版本；smoke check 通過後，銷毀沒有 revision 釘住、也不在最新 2 個之內的舊版本）
3. `!colab on`

token 外洩時：用專用帳號登入 https://myaccount.google.com/connections 移除「Google Cloud SDK」的存取權。這會撤銷這個帳號透過 gcloud 產生的**所有** token（包括 bot 正在用的，這也是要用專用帳號的原因），接著照上面的步驟產生新的。

### 註冊與檢查

VM 開機後以 HMAC 簽章 POST `/internal/colab/register`；bot 檢查 tunnel 網址（只接受 https + trycloudflare 或 `COLAB_TUNNEL_HOST`，且解析出的 IP 全是公開位址），再經 tunnel 探測 Ollama 版本 0.35.0、qwen3.8 已載入且完全在 VRAM、qwen3.8 的 digest 與釘住的一致、qwen3.8 的 think 等級，全部通過才標成 READY（nimble 沒載入、digest 不一致或沒記錄到、或沒有完全在 VRAM 時仍會 READY，但這個 session 不用 nimble 路由，`/health/detail` 的 `colab.unpinned`／`colab.off_gpu` 會列出；之後的健康檢查發現 nimble 被踢出 VRAM 時也一樣，qwen3.8 對話照常）。探測失敗回 503，VM 可重試；原因寫在 `!colab status` 與 `/health/detail` 的 `colab.last_error`。

### 排查

先看 `!colab status`（狀態、開機步驟、最後錯誤、冷卻時間、CU 餘額與用量）。常見狀況（assign 412／503／400、DENYLISTED、`invalid_grant`、digest 不符、tunnel 連不上、524）的原因與處理見 [colab/README.md](colab/README.md)〈疑難排解〉。

```bash
# session 狀態（不含網址與金鑰）
curl -fsS -H "X-Cron-Token: $CRON_SECRET" "$SERVICE_URL/health/detail" \
  | python3 -c 'import json, sys; print(json.dumps(json.load(sys.stdin)["colab"], indent=2, ensure_ascii=False))'

# Colab 相關 log（READY、註冊被拒、斷路器、nimble 不確定或停用、控制器的狀態轉換）
gcloud logging read \
  "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE_NAME}\" AND textPayload=~\"Colab|nimble\"" \
  --project "$PROJECT_ID" \
  --limit 50 \
  --format='table(timestamp,severity,textPayload)'
```

### 路由評測（routing eval）

`scripts/evals/routing_eval.py` 拿 `scripts/evals/routing_zh_tw.jsonl`（約 270 則人工標註的繁中 LINE 訊息：四個 agent、三種輸出、要不要思考，含 `tests/test_routing_rules.py` 與 `prompts/orchestrator.md` 的所有已知誤判、續問、引用、英文與中英混合、prompt injection）比較規則（離線）、Colab 的 nimble、NVIDIA 與 OpenRouter 的路由層，並用 nimble 的原始機率掃描門檻，建議 `COLAB_ROUTER_CONFIDENCE` / `COLAB_ROUTER_THINK_ON` / `COLAB_ROUTER_THINK_OFF`：在設定允許的範圍內，gated precision 達到 `--target-precision`（預設 0.95）時覆蓋率最高的值。

```bash
.venv312/bin/python scripts/evals/routing_eval.py --dry-run     # 只印計畫，不送任何請求
.venv312/bin/python scripts/evals/routing_eval.py               # 只跑規則（離線）
.venv312/bin/python scripts/evals/routing_eval.py --env-file .env --candidates rules,colab
.venv312/bin/python scripts/evals/routing_eval.py --env-file .env --candidates rules,nvidia,openrouter
```

- `colab`、`nvidia`、`openrouter` 會用真的 key 送請求，**要你同意才跑**，而且一定要明確給 `--env-file`（不會自動讀 `.env`），沒給就 SKIP。
- `colab` 需要一個 READY、能推論的 session（不是 `COLAB_DRY_RUN`）：先 `!colab on`，`live_gate.py --path colab` 通過後再跑。它和 live gate 一樣用本機 ADC 讀 Firestore 的 session 文件、由 `COLAB_SHARED_SECRET` 推導 bearer，一次送一則；READY 只維持到上次健康檢查後 120 秒，所以跑得比這久時它會自己再做健康檢查，檢查失敗就停下。
- 任何候選停下而留下沒跑的訊息（`not_run`）時，它標為 `partial`，整個評測以 exit code 1 結束，也不給 nimble 門檻建議；確定要用這些部分結果時加 `--allow-partial`。
- NVIDIA 每分鐘最多 40 次、OpenRouter 20 次，和線上的 bot 共用額度，bot 忙碌時請在離峰執行或加上 `--max-rpm`。OpenRouter 免費模型每天只有 `OPENROUTER_FREE_DAILY_LIMIT` 次（50；買過 credits 是 1,000），完整一輪要 270–540 次，不夠時用 `--limit` 取平均分布的樣本。
- 報告（Markdown 與 JSON）寫在 `tmp/routing-eval/`：各路由的覆蓋率與準確率（全部訊息，以及規則放行給模型路由的訊息）、agent 混淆表、各標籤準確率、誤判清單，和 nimble 的門檻曲線與建議值（每個點附 95% Wilson 下界，另有 output_format 混淆表；思考門檻只在通過建議信心門檻的訊息上掃描）；不含 key、bearer 或 tunnel 網址。把建議值寫進 `.env` 再部署；nimble 換 checkpoint 或量化版本時要重跑。
- 資料集只有約 270 則、模型路由實際會看到的約 167 則，而且八成是 chat：建議值只能當**暫定**，報告裡的 95% 下界常常遠低於目標 precision。每個門檻至少要有 20 則通過才會被建議（20 則全對的下界約 0.84）；累積更多實際訊息的標註後再重跑、再收緊。

---

## 網路搜尋：Tavily

網路搜尋只用 Tavily：有 `TAVILY_API_KEY` 時用 key（每月 1,000 credits 免費；basic 深度每次搜尋 1 credit），沒有 key 時在 `TAVILY_KEYLESS_ENABLED=true` 下用 keyless 模式（免費、有速率限制）。搜尋不需要 VM、VPC 或任何 GCP 資源，整個專案維持在 GCP 免費額度內。Tavily 無法使用（沒有設定、額度用完、被限速）或查無結果時，模型依既有知識回答，開頭註明「（目前無法使用搜尋服務，以下依模型既有知識回答，可能不是最新資訊）」。設計理由見 [ADR-0009](docs/adr/0009-search-backends-and-ssrf-safe-fetching.md)。

注意：

- `/health/detail` 的 `web_search` 會顯示 Tavily 的模式（`key` / `keyless` / `off`）、本程序用掉的 credits、冷卻秒數與最後錯誤。Tavily 回 429 時依 `Retry-After` 冷卻（沒有時 1 分鐘），額度用完（432 / 433）或 key 被拒（401 / 403）冷卻 1 小時；冷卻期間的搜尋直接改為不搜尋回答。
- `SEARCH_RERANK_ENABLED=true` 會用 embedding 依語意重排搜尋結果（每次搜尋多 2 次 NVIDIA embedding 呼叫），預設關閉。

---

## 啟用 Firestore 持久化記憶（選用）

Cloud Run 預設 `min-instances=0`，沒人聊天時 instance 會被回收，**in-memory 記憶會清空**。
若希望跨 cold start / 多 instance 共享長期摘要與 user profile，可啟用 Firestore Native backend：

### 1. 建立 Firestore Native database（與 Cloud Run 同 region 避免跨區流量費）

```bash
gcloud firestore databases create \
  --project YOUR_PROJECT_ID \
  --database='(default)' \
  --location=us-west1 \
  --type=firestore-native
```

> ℹ️ 一個 GCP project 只能有一個 `(default)` Firestore database；若已存在可直接沿用，
> 或用具名 database（例：`--database=linebot-memory`）並把 `FIRESTORE_DATABASE` 設為相同名稱。

### 2. 授權 Cloud Run service account 讀寫 Firestore

已跑過 `./scripts/provision_gcp.sh --step runtime-sa --apply` 的話，`linebot-runtime` 已經有 `roles/datastore.user`，可以跳過這步。尚未佈建時：

```bash
SA="$(gcloud run services describe ${SERVICE_NAME} \
  --project YOUR_PROJECT_ID --region us-west1 \
  --format='value(spec.template.spec.serviceAccountName)')"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:${SA}" \
  --role="roles/datastore.user"
```

### 3. 設定環境變數（Cloud Run / `.env`）

```bash
FIRESTORE_ENABLED=true
FIRESTORE_PROJECT_ID=YOUR_PROJECT_ID
FIRESTORE_DATABASE="(default)"
FIRESTORE_COLLECTION_PREFIX=linebot
MEMORY_CACHE_TTL_SECONDS=60
LINE_PROFILE_CACHE_TTL_SECONDS=86400
```

部署後可從 `/health/detail` 的 `memory.backend` 欄位確認目前使用的 backend（`in_memory` 或 `firestore`）。
若 `google-cloud-firestore` 套件缺失或憑證不足，啟動時會印出 warning 並自動退回 in-memory，不會中斷服務。

### 4. 免費額度提醒

- Firestore Native 每天免費 **50K reads / 20K writes / 1 GiB 儲存**。
- 本 bot 每輪對話約 1~2 reads + 1~2 writes（chat memory；同一 chat 連續訊息有 `MEMORY_CACHE_TTL_SECONDS` 讀取快取）。user profile 每位使用者最多每 10 分鐘讀 1 次、每天寫 1 次（改名時另寫）；episode 每個 chat 最多每小時 1 筆（`EPISODE_MIN_INTERVAL_SECONDS`）。個人用量遠低於免費額度。
- 最容易先用完的是 **1 GiB 儲存**：大欄位的預設單欄位索引也算儲存量，請依下方〈6. 大欄位的單欄位索引豁免〉關掉。
- 文件 ID 格式：`{source_type}::{chat_id}`（chat memory）、`{user_id}`（user profile），**不含 LINE token / 個資**，但長期摘要本身可能含使用者對話內容，請自行評估。
- 想清空可在 Firestore console 直接刪除 collection，或用 `gcloud firestore` CLI；app 端傳 `!new` 則只清當前 chat scope（含對應的 episodes 子集合）。

### 5. 啟用 Episodic Memory（向量檢索 + 自動 TTL）

每次壓縮完成時，若這個 chat 上一筆 episode 已超過 `EPISODE_MIN_INTERVAL_SECONDS`（預設 3600 秒），就把「上一筆之後被壓縮的對話（`snippet`，最新 2000 字，每則訊息最多 300 字）+ 當時的摘要（`summary`）」一起做成 1024 維向量（連同 `embedding_model` / `embedding_dim` 標記）寫入
`{prefix}_episodes/{chat_key}/items/{auto_id}`；未滿間隔的壓縮內容先存在 chat 文件的 `episode_pending_text`，併入下一筆 episode（同樣只保留最新 2000 字；忙碌的聊天室在一小時內更早被壓縮的對話只留在長期摘要裡）；embedding 失敗或被略過（額度、冷卻）時不寫 episode，對話留在 `episode_pending_text` 併入下一筆。chat agent 透過 `recall_memory`
工具用語意檢索找出較舊的對話片段（回傳對話片段最多 600 字、摘要最多 300 字）。要啟用向量搜尋必須建立 Vector Index：

```bash
# 建立 cosine 距離的向量索引（embedding 欄位 = nemotron-3-embed-1b 的前 1024 維）
gcloud firestore indexes composite create \
  --project=YOUR_PROJECT_ID \
  --database='(default)' \
  --collection-group=items \
  --query-scope=COLLECTION \
  --field-config=vector-config='{"dimension":"1024","flat":"{}"}',field-path=embedding
```

> ℹ️ 索引建立會花數分鐘；建立完成前 `recall_memory` 會回傳空結果，但 chat agent
> 會自動 fallback，不會中斷對話。

設定 90 天自動清理舊 episodes（避免長期累積爆量）。程式會在每筆 episode 寫入 `expires_at = now + 90 days`：

```bash
gcloud firestore fields ttls update expires_at \
  --project=YOUR_PROJECT_ID \
  --database='(default)' \
  --collection-group=items \
  --enable-ttl
```

> Firestore TTL 會以 `expires_at` timestamp 欄位為準自動刪除過期文件。
> 若要改保留週期，請在程式端調整 episode 寫入時的 `expires_at` 或自行設不同 TTL 策略。

### 6. 大欄位的單欄位索引豁免（建議）

Firestore 預設會替每個欄位建立單欄位索引（array / map 裡的每個值也會），索引項目算在儲存量裡。下面這些欄位只會整筆讀寫、從不拿來查詢，關掉它們的索引可以明顯降低儲存用量：

| collection group | 欄位 |
| --- | --- |
| `{prefix}_chat_memory` | `recent_messages`、`summary_text`、`episode_pending_text` |
| `items`（`{prefix}_episodes/{chat_key}/items`） | `summary`、`snippet`、`messages`（舊版 episode 的欄位） |

請自行執行（`linebot` 換成你的 `FIRESTORE_COLLECTION_PREFIX`，每個欄位一個長時間作業，約數分鐘）：

```bash
PROJECT=YOUR_PROJECT_ID
DATABASE='(default)'
PREFIX=linebot

for FIELD in recent_messages summary_text episode_pending_text; do
  gcloud firestore indexes fields update "$FIELD" \
    --project="$PROJECT" --database="$DATABASE" \
    --collection-group="${PREFIX}_chat_memory" \
    --disable-indexes
done

for FIELD in summary snippet messages; do
  gcloud firestore indexes fields update "$FIELD" \
    --project="$PROJECT" --database="$DATABASE" \
    --collection-group=items \
    --disable-indexes
done

# 確認：列出有非預設索引設定的欄位
gcloud firestore indexes fields list --project="$PROJECT" --database="$DATABASE"
```

> 不要豁免 `embedding`（向量索引另外建立）或 `expires_at`（TTL 欄位）。`items` 這個 collection group 名稱也適用於同一個 database 裡其他叫 `items` 的子集合；若你的 database 還有其他用途，請先確認。
> 要還原某個欄位的預設索引：`gcloud firestore indexes fields update FIELD --collection-group=... --clear-exemption`。

### 7. NVIDIA Embedding 模型

- 使用免費的 `nvidia/nemotron-3-embed-1b`（多語言），共用既有 `NVIDIA_API_KEY`。API 一律回傳 2048 維，程式只保留前 `EMBEDDING_DIMENSIONS`（預設 1024）維並重新 L2 正規化，所以沿用同一個 1024 維索引。
- 每筆 episode 會記錄 `embedding_model` 與 `embedding_dim`；`recall_memory` 向索引要 2 倍的候選，再丟掉不是目前模型產生的向量。換 embedding 模型後請跑 `scripts/migrate_embeddings.py`（見〈從 v2 升級到 v3〉第 4 步），否則舊 episodes 不會被召回；設計、回滾與改維度的做法見 [ADR-0011](docs/adr/0011-embedding-migration.md)。
- 若要改其他 embedding 模型，用 `MODEL_ROLES` 覆寫 `embed` 角色（只能有一層，`dim` 必須等於 `EMBEDDING_DIMENSIONS`）；endpoint 可用 `NVIDIA_EMBEDDING_ENDPOINT` 調整（選填）。embedding 請求與對話共用 NVIDIA 的額度與模型健康狀態。
- Embedding 失敗會 graceful degrade：episode 仍會儲存（沒有向量欄位），只是無法被 `recall_memory` 找到。模型回 404/410 時只記一次 ERROR log。

---

## 直接 Cloud Build 提交

`cloudbuild.yaml` 現在只做 **ruff + pytest + compileall，然後建置並推送 image**，不部署、也不接收任何 env 檔。若不想用 wrapper 腳本，可以只建 image：

```bash
gcloud builds submit . \
  --project YOUR_PROJECT_ID \
  --config cloudbuild.yaml \
  --substitutions=_REGION=us-west1,_IMAGE_TAG=manual-$(date -u +%Y%m%d%H%M%S)
```

送出前請先確認不會上傳密鑰檔：`gcloud meta list-files-for-upload . | python3 scripts/envfile.py check-upload`。已佈建時請加上 `--service-account=projects/YOUR_PROJECT_ID/serviceAccounts/linebot-builder@YOUR_PROJECT_ID.iam.gserviceaccount.com --gcs-source-staging-dir=gs://YOUR_PROJECT_ID-cb-src-usw1/source`，否則原始碼會放到預設的 `gs://YOUR_PROJECT_ID_cloudbuild`。Artifact Registry repository 需已存在（wrapper 或 `provision_gcp.sh --step builder-sa` 會建立）。之後再自行 `gcloud run deploy --image ...`；Cloud Run service 已有的 secret 掛載與環境變數會沿用，否則建議直接用 `./scripts/deploy_cloud_run.sh`。

---

## 依賴鎖定與本機驗證

Image 與 Cloud Build 都跑 **Python 3.12**（`python:3.12-slim`，以 digest 固定），依賴只從 hash 鎖定檔安裝：

| 檔案 | 用途 |
| --- | --- |
| `requirements.in` | 執行期直接依賴，**唯一需要手動編輯**的清單 |
| `requirements.lock` | 由 `requirements.in` 產生，含全部遞移依賴與 sha256；Dockerfile 用 `pip install --no-cache-dir --require-hashes --no-deps -r requirements.lock` 安裝 |
| `requirements-dev.in` | `-r requirements.in` + pytest / pytest-asyncio / `ruff==0.16.4`；以 `-c requirements.lock` 讓共用套件與 image 版本一致 |
| `requirements-dev.lock` | 由 `requirements-dev.in` 產生；Cloud Build 測試步驟與本機測試安裝它 |
| `requirements.txt` / `requirements-dev.txt` | 相容用，只有一行 `-r requirements*.lock`，不要在這裡加套件（沒有 hash 的套件會讓 pip 直接拒絕安裝） |

鎖定檔用 `--universal` 產生：同一份檔案同時適用 Cloud Run 的 linux/amd64 image 和本機 macOS / Linux 開發環境（uv 會列出每個版本所有 wheel 的 hash，平台相關的依賴以 environment marker 標註，pip 會自動略過不適用的行）。

### 重新鎖定依賴

需要 [uv](https://docs.astral.sh/uv/)。**先鎖 runtime，再鎖 dev**（dev 以 `requirements.lock` 當 constraint）。兩行都要帶同一組 `--no-emit-package`（與兩個鎖定檔開頭記錄的指令相同）：

```bash
NO_EMIT=(--no-emit-package fastjsonschema --no-emit-package google-auth-oauthlib
  --no-emit-package jsonschema --no-emit-package jsonschema-specifications
  --no-emit-package nbformat --no-emit-package oauthlib --no-emit-package prompt-toolkit
  --no-emit-package pyarrow --no-emit-package referencing --no-emit-package requests-oauthlib
  --no-emit-package rpds-py --no-emit-package shellingham --no-emit-package typer
  --no-emit-package wcwidth)
uv pip compile requirements.in --universal --python-version 3.12 --generate-hashes "${NO_EMIT[@]}" -o requirements.lock
uv pip compile requirements-dev.in --universal --python-version 3.12 --generate-hashes "${NO_EMIT[@]}" -o requirements-dev.lock
```

- `--no-emit-package`：`google-colab-cli`（Colab control plane，ADR-0010）只以函式庫方式使用 `colab_cli.client`、`.runtime`、`.contents`，這三個模組不會 import 它的 CLI 依賴（typer、prompt-toolkit、nbformat、google-auth-oauthlib 及其依賴）與 pyarrow（經 jupyter-mimetypes 帶進來，安裝後約 157 MB）。uv 仍照完整依賴解析版本，只是不寫進鎖定檔；因為 image 以 `--no-deps` 安裝，這些套件不會進 image（Colab 相關套件安裝後共約 16 MB）。`tests/test_packaging.py` 會檢查兩個鎖定檔都沒有它們、三個模組在缺少它們時仍可 import，且 `src/` 從不 import `colab_cli.common`、`.auth`、`.commands`、`.cli`
- 只改 `.in` 時照跑上面的指令，既有版本會盡量保留
- 升級全部套件：兩行都加 `--upgrade`；只升級一個套件：加 `--upgrade-package <name>`
- 鎖完跑一次本機測試；`tests/test_packaging.py` 會檢查 `src/` 與 `main.py` import 的每個套件都在 `requirements.lock` 且有 hash
- aiohttp 或它的依賴版本變了時，也重新鎖定 Colab VM 上 auth proxy 用的 `colab/proxy-requirements.lock`（`tests/test_colab_vm_scripts.py` 會檢查它的版本與 hash 和 `requirements.lock` 相同）：

  ```bash
  uv pip compile colab/proxy-requirements.in --universal --python-version 3.11 --generate-hashes -o colab/proxy-requirements.lock
  ```

### 本機測試（Python 3.12）

```bash
uv venv -p python3.12 .venv312
uv pip sync --python .venv312/bin/python requirements-dev.lock
.venv312/bin/python -m ruff check .
.venv312/bin/python -m pytest -q
```

`.venv312/` 已列在 `.gitignore`、`.gcloudignore`、`.dockerignore`，不會被 commit、上傳或打包進 image。

### 更新 base image digest

`Dockerfile` 的 `FROM` 與 `cloudbuild.yaml` 的 test step 用同一個 `python:3.12-slim@sha256:...`（multi-arch index digest），要一起更新（`tests/test_packaging.py` 會檢查兩者一致）：

```bash
docker buildx imagetools inspect python:3.12-slim   # 取 "Digest:" 那一行
```

更新 digest 時順便把 Dockerfile 註解的日期與 Python 版本改掉。

### 記憶體預算（Cloud Run `--memory 512Mi`）

2026-09-30 以 `.venv312`（Python 3.12.11、macOS arm64、上述 lock，已含 trafilatura）量測的 RSS；憑證全部是 stub、所有對外連線被擋掉：

| 階段 | RSS |
| --- | --- |
| Python 啟動 | ~22 MB |
| `import main` | ~66 MB |
| 進入 app lifespan（stub 憑證、in-memory backend） | ~81 MB |
| 再打一次 `/health` | ~81 MB |
| 再載入延遲 import 的套件（Firestore、Cloud Storage、edge-tts、Pillow、OpenCC） | ~108 MB |
| 再用 trafilatura 擷取一頁（第一次會在擷取執行緒裡 import，約 0.4–1 秒） | ~124 MB |
| 峰值（`ru_maxrss`） | ~124 MB |
| Colab control plane 的套件（`colab_cli.client`／`.runtime`／`.contents`、jupyter-kernel-client、pyzmq 等；只在 `COLAB_ENABLED` 時由 Cloud Tasks 與 cron 第一次呼叫時載入，冷啟動不載入） | 再 +~10 MB |

遠低於 ~350 MB 的警戒線，**維持 `512Mi`**，不需要 1Gi。處理中的圖片與 TTS 音訊會再多用一些；這些數字不是在 linux/amd64 容器內量的，實際值可能略有差異。若 Cloud Run 出現 `Memory limit exceeded`，再把 `scripts/deploy_cloud_run.sh` 的 `--memory` 調到 `1Gi`。
