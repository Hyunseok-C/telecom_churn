#=============================================================================
# 고객 이탈 예측 : 모델별 성능 비교
# 수정: 25-11-25 18:50
#=============================================================================
#-----------------------------------------------------------------------
# 1. 성능 확인
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
# (방법1) 모델을 돌린 후 7가지 측도 데이터 프레임 생성
data_logit
data_lda
data_qda
data_knn
data_tree
data_bag
data_rf
data_gbm
data_svm_poly
# *(주의사항) 모든 모델을 돌린 상태여야함

result_metrics <- rbind(
  data_logit,
  data_lda,
  data_qda,
  data_knn,
  data_tree,
  data_bag,
  data_rf,
  data_gbm,
  data_svm_poly
)

# 모델 이름 열 생성
result_metrics$Model <- rownames(result_metrics)
result_metrics$Model <- factor(result_metrics$Model,
                   levels = c("logit", "lda", "qda", "knn", "tree", "bag", "rf", "gbm","svm_poly"),
                   labels = c("Logistic Regression",
                              "LDA",
                              "QDA",
                              "KNN",
                              "Decision Tree",
                              "Bagging",
                              "Random Forest",
                              "GBM",
                              "SVM (Poly)")
)
result_metrics <- result_metrics[, c("Model", setdiff(names(result_metrics), "Model"))]
rownames(result_metrics) <- 1:nrow(result_metrics)
result_metrics

#-----------------------------------------------------------------------
# (방법2) 모델을 돌리지 않고 수기로 7가지 측도 데이터 프레임 생성
result_metrics <- data.frame(
  Model = c(
    "Logistic Regression", "LDA", "QDA", "KNN",
    "Decision Tree", "Bagging", "Random Forest", "GBM", "SVM (Poly)"
  ),
  
  CV_Accuracy = c(
    0.8604577, 0.8503334, 0.8567075, 0.8938480,
    0.9261041, 0.9261067, 0.9435593, 0.9351009, 0.9186043
  ),
  CV_Acc_SD = c(
    0.013904717, 0.016868940, 0.023871880, 0.005451168,
    0.006476334, 0.008353371, 0.005349573, 0.008132240, 0.004117932
  ),
  CV_Kappa = c(
    0.2223295, 0.2464754, 0.4130754, 0.4483647,
    0.6669605, 0.6707937, 0.7013951, 0.7044947, 0.6447019
  ),
  CV_Kappa_SD = c(
    0.06680390, 0.06340694, 0.07859013, 0.02297608,
    0.02522189, 0.03549419, 0.02318871, 0.03754622, 0.02042344
  ),
  
  Test_Accuracy = c(
    0.8650675, 0.8530753, 0.8659625, 0.9070645,
    0.9323537, 0.9340330, 0.9340030, 0.9370315, 0.9265367
  ),
  Test_Kappa = c(
    0.1973097, 0.2149548, 0.4236923, 0.5299239,
    0.6262660, 0.7098911, 0.7015539, 0.7123642, 0.6871213
  ),
  
  Sensitivity = c(
    0.9894552, 0.9666081, 0.9476110, 0.9894552,
    0.9861620, 0.9806678, 0.9859402, 0.9894552, 0.9701230
  ),
  Specificity = c(
    0.1428571, 0.1987762, 0.4387755, 0.4285714,
    0.8086124, 0.6363253, 0.8263513, 0.6236531, 0.6734694
  ),
  Precision = c(
    0.8710700, 0.8744038, 0.9070946, 0.9059315,
    0.9395787, 0.9441624, 0.9396985, 0.9389988, 0.9452055
  ),
  Balanced_Acc = c(
    0.5661652, 0.5802482, 0.6921682, 0.7090132,
    0.8084749, 0.8219666, 0.8092967, 0.8110541, 0.8217962
  ),
  F1 = c(
    0.9259868, 0.9181970, 0.9205646, 0.9478114,
    0.9618734, 0.9620690, 0.9622642, 0.9640411, 0.9575022
  ),
  
  stringsAsFactors = FALSE
)
result_metrics

#-----------------------------------------------------------------------
# 2. 최종 성능 비교 시각화
#-----------------------------------------------------------------------
### 시각화1-------------------------------------------------------------
## <모델별 Test 정확도와 kappa 비교 시각화>
library(ggplot2)

df <- result_metrics

# y축 범위 설정
acc_min <- 0.83
acc_max <- 0.95

kappa_min <- 0.20
kappa_max <- 1.00


# Kappa를 Accuracy 축에 맞게 선형 변환
df$Kappa_scaled <- (df$Test_Kappa - kappa_min) / 
  (kappa_max - kappa_min) *
  (acc_max - acc_min) +
  acc_min

model_cols <- c(
  "Logistic Regression" = "#FFB3BA", # Red (파스텔 레드)
  "LDA"                 = "#FFCC99", # Orange (파스텔 오렌지)
  "QDA"                 = "#FFFF99", # Yellow (파스텔 옐로우)
  "KNN"                 = "#CCFFCC", # Green (파스텔 그린)
  "Decision Tree"       = "#99FFEB", # Teal/Cyan (파스텔 시안)
  "Bagging"             = "#99CCFF", # Light Blue (파스텔 블루)
  "Random Forest"       = "#C5B3FF", # Indigo (파스텔 인디고)
  "GBM"                 = "#E1B3FF", # Violet (파스텔 바이올렛)
  "SVM (Poly)"          = "#FFD6E7"  # Soft Pink (무지개 마지막 연핑크)
)


df <- df %>%
  arrange(Test_Accuracy) %>%                      # Accuracy 기준 정렬
  mutate(Model = factor(Model, levels = Model))  # 정렬된 순서로 factor 지정

ggplot(df, aes(x = Model)) +
  
  # Accuracy 막대
  geom_col(aes(y = Test_Accuracy, fill = Model),
           width = 0.6, alpha = 0.9,
           color = "black", linewidth = 0.3) +
  
  # Accuracy 라벨
  geom_text(aes(y = Test_Accuracy, label = round(Test_Accuracy, 4)),
            vjust = -0.5, size = 4) +
  
  # Kappa
  geom_point(aes(y = Kappa_scaled), color = "darkblue",
             size = 3) +
  geom_text(aes(y = Kappa_scaled, 
                label = round(Test_Kappa, 4)), color = "darkblue",
            vjust = -1, size = 3.5, show.legend = FALSE) +
  
  # 왼쪽/오른쪽 y축 설정
  scale_y_continuous(
    name = "Accuracy",
    sec.axis = sec_axis(
      transform = ~ (.-acc_min) / (acc_max - acc_min) *
        (kappa_max - kappa_min) + kappa_min,
      name = "Kappa"
    )
  ) +
  
  scale_fill_manual(values = model_cols) +
  
  # 실제 그리는 범위만 0.88~0.95로 잘라서 보기
  coord_cartesian(ylim = c(acc_min, acc_max)) +
  
  labs(
    title = "모델별 Test Accuracy와 Kappa 비교",
    x = "모델"
  ) +
  theme_light() +
  theme(
    legend.position = "none",                 # 범례 제거
    plot.title = element_text(size = 16, face = "bold"),
    axis.text.x = element_text(angle = 20, hjust = 1)
  )

### 시각화2-------------------------------------------------------------
## <모델별 CV와 Test 오분류율>
library(ggplot2)
library(dplyr)
library(tidyr)

result_metrics
plot_df <- result_metrics[,c("Model", "CV_Accuracy", "Test_Accuracy")]

# Error 계산
df <- plot_df %>%
  mutate(
    CV_Error   = 1 - CV_Accuracy,
    Test_Error = 1 - Test_Accuracy
  ) %>%
  arrange(desc(Test_Error)) %>%                             # Test_Error 작은 순으로 정렬
  mutate(Model = factor(Model, levels = Model))       # 이 순서를 factor 레벨로 고정

# long format 변환
df_long <- df %>%
  pivot_longer(cols = c(CV_Error, Test_Error),
               names_to = "Type", values_to = "Error")

# 시각화
ggplot(df_long, aes(x = Model, y = Error, group = Model)) +
  geom_line(color = "gray60", linewidth = 1) +
  geom_point(aes(color = Type), size = 4) +
  ylim(0.062, 0.15) +
  geom_text(aes(label = round(Error, 3), color = Type),
            vjust = -1,hjust = -0.3,  size = 3.2, show.legend = FALSE) +
  scale_color_manual(values = c("CV_Error" = "#e41a1c", "Test_Error" = "#377eb8"),
                     labels = c("CV Error", "Test Error")) +
  labs(
    title = "CV Error vs Test Error by Model",
    x = "Model",
    y = "Error Rate",
    color = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.border = element_rect(color = "black", fill = NA),
    axis.text.x = element_text(angle = 20, hjust = 1),
    plot.title = element_text(hjust = 0.5)
  )

### 시각화3-------------------------------------------------------------
## <모델별 성능 9가지 비교 레이다 차트>
library(fmsb)
library(dplyr)

df <- result_metrics[,c(1,6:12)]

cols <- c(
  "#FF4C4C",  # Red
  "#FF914D",  # Orange
  "#FFD93D",  # Yellow
  "#4CD964",  # Green
  "#1ABC9C",  # Teal / Cyan
  "#3498DB",  # Blue
  "#5E60CE",  # Indigo
  "#9B59B6",  # Violet
  "#FF6F91"   # Pink
)

# 지표별 min/max (모든 모델 기준) -> 정규화
metric_mat <- as.matrix(df[, -1])

col_min <- apply(metric_mat, 2, min)
col_max <- apply(metric_mat, 2, max)

# 혹시 max == min 인 경우(모두 같은 값) 나눗셈 방지용
range_zero <- (col_max == col_min)

# 축 라벨 (0~1)
axis_labs <- seq(0, 1, by = 0.25)

p <- ncol(df) - 1 
metric_names <- c("Test_Accuracy", "Test_Kappa", "Sensitivity", 
                  "Specificity", "Precision", "Balanced_Acc", "F1")


# 모델별 레이다 차트
par(mfrow = c(2, 5))
plot.new()
for (i in 1:nrow(df)) {
  
  model_name      <- df$Model[i]
  model_vals_raw  <- as.numeric(df[i, -1])   # 원래 지표 값
  
  # --- 각 지표별로 0~1로 표준화 ---
  model_vals_scaled <- (model_vals_raw - col_min) / (col_max - col_min)
  # max=min인 지표는 모두 같은 값이므로 0.5로 두기 (중간 위치)
  model_vals_scaled[range_zero] <- 0.5
  
  # 레이다 차트용 데이터 (1행: 1, 2행: 0, 3행: 표준화된 값)
  radar_mat <- rbind(
    rep(1, p),
    rep(0, p),
    model_vals_scaled
  )
  colnames(radar_mat) <- metric_names
  rownames(radar_mat) <- c("max", "min", model_name)
  
  radar_df <- as.data.frame(radar_mat)   
  
  radarchart(
    radar_df,
    axistype    = 1,
    seg         = 4,
    caxislabels = axis_labs,
    pcol        = cols[i],
    pfcol       = scales::alpha(cols[i], 0.3),
    plwd        = 2,
    plty        = 1,
    cglcol      = "grey80",
    cglty       = 1,
    cglwd       = 0.8,
    axislabcol  = "grey30",
    vlcex       = 0.8,
    title       = paste0("Model: ", model_name)
  )
}
par(mfrow = c(1, 1))



### 시각화4-------------------------------------------------------------
## <모델별 ROC 곡선 비교>
library(pROC)
library(dplyr)
library(purrr)
library(ggplot2)

# 실제 정답 (Test Set의 타깃)
y_true <- factor(test$Churn, levels = c("0","1"))

# 각 모델별 Test 예측확률 (양성: "1")
p_logit     <- predict(fit_logit,      newdata = test, type = "prob")[, "1"]
p_lda       <- predict(fit_lda,        newdata = test, type = "prob")[, "1"]
p_qda       <- predict(fit_qda,        newdata = test, type = "prob")[, "1"]
p_knn       <- predict(fit_knn_all,    newdata = test, type = "prob")[, "1"]

p_tree      <- predict(fit_tree_cv,    newdata = test, type = "prob")[, "1"]
p_bag       <- predict(fit_bag,        newdata = test, type = "prob")[, "1"]
p_rf        <- predict(fit_rf,         newdata = test, type = "prob")[, "1"]
p_gbm       <- predict(fit_gbm,        newdata = test, type = "prob")[, "1"]

kmod_poly <- fit_svm_poly$finalModel
x_test_poly_pp <- predict(fit_svm_poly$preProcess,  x_test)
p_svm_poly <- as.numeric(predict(kmod_poly, as.matrix(x_test_poly_pp), type = "decision"))

# ROC 오브젝트 → ggplot용 데이터프레임으로 변환하는 헬퍼 함수
get_roc_df <- function(name, prob){
  roc_obj <- pROC::roc(
    y_true,
    prob,
    direction = "<"   # 점수↑ → 1일 확률↑
  )
  
  auc_val <- as.numeric(auc(roc_obj))
  
  data.frame(
    Model = name,
    Label = paste0(name, " (AUC = ", round(auc_val, 3), ")"),
    TPR   = roc_obj$sensitivities,
    FPR   = 1 - roc_obj$specificities,
    AUC   = auc_val
  )
}

# 모든 모델 ROC 데이터 한 번에 만들기
roc_all <- bind_rows(
  get_roc_df("Logistic Regression", p_logit),
  get_roc_df("LDA",                 p_lda),
  get_roc_df("QDA",                 p_qda),
  get_roc_df("KNN",                 p_knn),
  get_roc_df("Decision Tree",       p_tree),
  get_roc_df("Bagging",             p_bag),
  get_roc_df("Random Forest",       p_rf),
  get_roc_df("GBM(boosting)",       p_gbm),
  get_roc_df("SVM (Poly)",          p_svm_poly)
)

# 그룹 나누기
grp_stat    <- c("Logistic Regression", "LDA", "QDA")
grp_single  <- c("KNN", "Decision Tree", "SVM (Poly)")
grp_ens     <- c("Bagging", "Random Forest", "GBM(boosting)")

roc_stat   <- roc_all %>% filter(Model %in% grp_stat)
roc_single <- roc_all %>% filter(Model %in% grp_single)
roc_ens    <- roc_all %>% filter(Model %in% grp_ens)

# 각 모델 별 팔레트
model_cols <- c(
  "Logistic Regression" = "#FF6B6B",  # 진한 레드
  "LDA"                 = "#FF8E44",  # 진한 오렌지
  "QDA"                 = "#F4D35E",  # 선명 옐로우
  "KNN"                 = "#7CCF8A",  # 진한 라이트 그린
  "Decision Tree"       = "#3EC1D3",  # 시안/티얼
  "Bagging"             = "#4A90E2",  # 비비드 블루
  "Random Forest"       = "#7B5EC6",  # 퍼플/인디고
  "GBM(boosting)"       = "#CE62D6",  # 바이올렛
  "SVM (Poly)"          = "#FF85C2"   # 선명 핑크
)

# 공통 ROC 플로팅 함수
plot_roc_group <- function(df, main_title){
  
  # Model → Label 매핑 테이블
  lab_map <- df %>%
    distinct(Model, Label) %>%
    arrange(Model)
  
  ggplot(df, aes(x = FPR, y = TPR, color = Model)) +   # 색은 Model 기준
    geom_line(linewidth = 1.1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    coord_equal() +
    labs(
      title = main_title,
      x     = "False Positive Rate (1 - Specificity)",
      y     = "True Positive Rate (Sensitivity)",
      color = NULL
    ) +
    scale_color_manual(
      values = model_cols,     # 색상: Model → 진한 색 배정
      labels = lab_map$Label   # 범례: Label(AUC 포함)로 표시
    ) +
    theme_minimal(base_size = 14) +
    theme(
      panel.border    = element_rect(color = "black", fill = NA),
      plot.title      = element_text(hjust = 0.5, face = "bold"),
      legend.position = "bottom"
    )
}          

# 최종 ROC 시각화 3개
p_roc_stat   <- plot_roc_group(roc_stat,   "ROC: 선형·통계 기반 모델 비교")
p_roc_single <- plot_roc_group(roc_single, "ROC: 단일 비선형 모델 비교")
p_roc_ens    <- plot_roc_group(roc_ens,    "ROC: 트리 기반 앙상블 모델 비교")

p_roc_stat
p_roc_single
p_roc_ens

par(mfrow = c(1, 1))


