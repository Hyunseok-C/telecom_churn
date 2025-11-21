#=============================================================================
# 고객 이탈 예측 : 배깅 모델
# 수정: 25-11-21 01:30
#=============================================================================
library(caret)
library(dplyr)
library(ggplot2)

#-----------------------------------------------------------------------
# 0. 데이터 불러오기
#-----------------------------------------------------------------------
churn <- read.csv("C:\\Users\\chs02\\OneDrive\\바탕 화면\\telecom_churn.csv")

#-----------------------------------------------------------------------
# 1. 학습/검증 데이터 분할
#-----------------------------------------------------------------------
select_vars <- setdiff(names(churn), c("DataPlan", "MonthlyCharge"))
select_vars

set.seed(123)
train_idx <- sample(1:nrow(churn), 0.8 * nrow(churn))
train <- churn[train_idx, select_vars]
test  <- churn[-train_idx, select_vars]

# 종속변수 factor 변환
train$Churn <- factor(train$Churn, levels = c(0,1))
test$Churn  <- factor(test$Churn,  levels = c(0,1))

#-----------------------------------------------------------------------
# 2. 공통 교차검증 설정 (5-fold, 1회 반복)
#-----------------------------------------------------------------------
ctrl <- trainControl(method="cv", number=5)

#-----------------------------------------------------------------------
# 3. 배깅
#-----------------------------------------------------------------------
library(ipred)

#----------------------------------------------------------
# (3-1) Bagging에서 nbagg(트리 수)에 따른 OOB Error 탐색 & 시각화
#----------------------------------------------------------
# (3-1-1) nbagg 후보값 벡터와 결과 저장용 데이터프레임 생성
nbagg_list <- seq(10, 300, by = 5)

bag_results <- data.frame(
  nbagg    = nbagg_list,
  oob_error = NA
)

# (3-1-2) 각 nbagg에 대해 Bagging 학습 후 OOB Error(coob) 저장
# (주의) 계산량 많음
set.seed(123)
for (i in seq_along(nbagg_list)) {
  
  nb <- nbagg_list[i]
  
  bag_model <- bagging(
    Churn ~ .,
    data = train,
    nbagg = nb,
    coob  = TRUE   # ← OOB Error 계산 옵션
  )
  
  bag_results$oob_error[i] <- bag_model$err    # OOB Error 저장
}

# (3-1-3) 최소 OOB Error 지점(최적 nbagg) 및 기준선 계산
best_idx   <- which.min(bag_results$oob_error)        # OOB Error 최소 인덱스
best_tree  <- bag_results$nbagg[best_idx]             # 해당 nbagg 값
best_error <- bag_results$oob_error[best_idx]         # 최소 OOB Error

best_df <- data.frame(nbagg = best_tree, oob_error = best_error)  # 단일 행 데이터프레임

median_oob <- median(bag_results$oob_error)  # 전체 OOB Error 중앙값
plateau_bg <- 100                             # OOB Error가 평탄 nbagg 기준

# (3-1-4) nbagg vs OOB Error 시각화
ggplot(bag_results, aes(x = nbagg, y = oob_error)) +
  
  # OOB Error 라인 및 점
  geom_line(color = "#E74C3C", linewidth = 1.2) +
  geom_point(size = 2.5, color = "#E74C3C") +
  
  # 평탄 구간 시작(plateau) 표시
  annotate("point", x = plateau_bg, y = bag_results$oob_error[plateau_bg],
           color = "darkgreen", size = 3) +
  geom_vline(xintercept = plateau_bg,
             color = "darkgreen",
             linetype = "dashed",
             linewidth = 0.8) +
  annotate("text",
           x = plateau_bg + 5,
           y = 0.1,
           label = "Plateau start (~100 trees)",
           color = "darkgreen",
           hjust = 0, vjust = 1.2,
           fontface = "bold") +
  
  # OOB Error 중앙값 수평선
  geom_hline(yintercept = median_oob,
             color = "navy",
             linetype = "dotted",
             linewidth = 0.8) +
  annotate("text",
           x = min(bag_results$nbagg) + 20,
           y = median_oob,
           label = paste0("Median OOB = ", round(median_oob, 4)),
           color = "navy",
           vjust = 1.5,
           fontface = "bold") +
  
  # 최적 nbagg(최소 OOB Error) 지점 표시
  annotate("point",
           x = best_tree, y = best_error,
           color = "blue", size = 3.5) +
  geom_vline(xintercept = best_tree,
             color = "blue",
             linetype = "dashed",
             linewidth = 0.7) +
  annotate("text",
           x = best_tree + 5,
           y = best_error,
           label = paste0("최소 nbagg = ", best_tree,
                          "\nOOB Error = ", round(best_error, 4)),
           color = "blue",
           fontface = "bold",
           hjust = 0, vjust = -1.2) +
  
  # y축 범위(확대해서 보기)
  scale_y_continuous(limits = c(0.06, 0.15)) +
  
  labs(
    title = "Bagging: 트리 수(nbagg)에 따른 OOB Error",
    x = "트리 수 (nbagg)",
    y = "OOB Error"
  ) +
  theme_light() +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    axis.title  = element_text(face = "bold")
  )

#----------------------------------------------------------
# (3-2) 선택된 nbagg로 최종 Bagging 모형 적합 & test 성능 평가
#----------------------------------------------------------
# (3-2-1) 최종 Bagging 모형 학습 (plateau 기준으로 nbagg = 50 사용)
set.seed(123)
fit_bag <- train(
  Churn ~ .,
  data      = train,
  method    = "treebag",
  trControl = ctrl,
  metric    = "Accuracy",
  nbagg     = 100      # 트리 수
)
fit_bag

# (3-2-4) test 데이터에 대한 예측 및 혼동행렬/정확도 계산
pred_bag <- predict(fit_bag, newdata = test)
cm_bag   <- caret::confusionMatrix(pred_bag, test$Churn)
acc_bag  <- cm_bag$overall["Accuracy"]

cm_bag   # 혼동행렬
acc_bag  # 0.9355322

#-----------------------------------------------------------------------
# 4. 배깅 결과 요약
#-----------------------------------------------------------------------
data_bag <- data.frame(
  # CV 성능 (resample 이용)
  CV_Accuracy   = mean(fit_bag$resample$Accuracy),
  CV_Acc_SD     = sd(fit_bag$resample$Accuracy),
  CV_Kappa      = mean(fit_bag$resample$Kappa),
  CV_Kappa_SD   = sd(fit_bag$resample$Kappa),
  
  # Test 성능 (confusionMatrix 이용)
  Test_Accuracy = cm_bag$overall["Accuracy"],
  Test_Kappa    = cm_bag$overall["Kappa"],
  Sensitivity   = cm_bag$byClass["Sensitivity"],
  Specificity   = cm_bag$byClass["Specificity"],
  Precision     = cm_bag$byClass["Precision"],
  Balanced_Acc  = cm_bag$byClass["Balanced Accuracy"]
)
rownames(data_bag) <- "bag"
data_bag

