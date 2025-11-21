#=============================================================================
# 고객 이탈 예측 : 모델별 성능 비교
# 수정: 25-11-21 02:55
#=============================================================================
#-----------------------------------------------------------------------
# 1. 성능 확인
#-----------------------------------------------------------------------
data_logit
data_lda
data_qda
data_knn
data_tree
data_bag
data_rf
data_gbm
data_svm_poly

# (주의사항) 모든 모델을 돌린 상태여야함

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
# 2. 최종 성능 비교 시각화
#-----------------------------------------------------------------------
### 시각화1-------------------------------------------------------------
## 모델별 Test 정확도와 kappa 비교 시각화
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
## 모델별 CV와 Test 오분류율
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
## 모델별 성능 9가지 비교 레이다 차트
library(fmsb)
library(dplyr)

df <- result_metrics[,c(1,6:11)]

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

# 2. 지표별 min/max (모든 모델 기준) -> 정규화
metric_mat <- as.matrix(df[, -1])

col_min <- apply(metric_mat, 2, min)
col_max <- apply(metric_mat, 2, max)

# 혹시 max == min 인 경우(모두 같은 값) 나눗셈 방지용
range_zero <- (col_max == col_min)

# 축 라벨 (0~1)
axis_labs <- seq(0, 1, by = 0.25)

p <- ncol(df) - 1 
metric_names <- c("Test_Accuracy", "Test_Kappa", "Sensitivity", 
                  "Specificity", "Precision", "Balanced_Acc")


# 3. 모델별 레이다 차트
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

