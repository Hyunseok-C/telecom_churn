#=============================================================================
# 고객 이탈 예측 : 랜덤포레스트 모델
# 수정: 25-11-25 00:20
#=============================================================================
library(caret)
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
# 3. 랜덤포레스트
#-----------------------------------------------------------------------
library(reshape2)
library(randomForest)

## (3-1) 랜덤포레스트 기본 모델 (ntree = 1000)
# 목적: 트리 수에 따른 OOB Error 시각화
set.seed(123)
rf_mo <- randomForest(
  Churn ~ ., 
  data = train,
  ntree = 1000,                                      # 트리 수
  mtry  = floor(sqrt(ncol(train)-1)),                # 기본 mtry = sqrt(p)
  importance = TRUE
)
rf_mo

## (3-2) OOB Error 추출
err_df <- as.data.frame(rf_mo$err.rate)          # OOB + 클래스별 에러율
err_df$Trees <- 1:nrow(err_df)                      # x축: 트리 수

# long format 변환
err_long <- melt(err_df, id.vars = "Trees",
                 variable.name = "Type",
                 value.name   = "Error")

# OOB Error만 추출
oob_only <- subset(err_long, Type == "OOB")

# OOB Error가 가장 낮은 지점
best_tree  <- which.min(oob_only$Error)
best_error <- oob_only$Error[best_tree]

# 중앙값 OOB Error 계산
median_oob <- median(oob_only$Error)
plateau_pt  <- 300

ggplot(oob_only, aes(x = Trees, y = Error)) +
  geom_line(color = "#E74C3C", linewidth = 1.1) +
  
  # 중앙값 기준선
  geom_hline(yintercept = median_oob,
             color = "darkblue",
             linetype = "dotted",
             linewidth = 1) +
  annotate("text",
           x = 50,
           y = median_oob,
           label = paste0("Median OOB = ", round(median_oob, 4)),
           color = "darkblue",
           vjust = 1.5,
           fontface = "bold") +
  
  # 평탄Tree 표시
  annotate("point", x = plateau_pt, y = oob_only$Error[plateau_pt],
           color = "darkgreen", size = 3) +
  geom_vline(xintercept = plateau_pt,
             color = "darkgreen",
             linetype = "dashed",
             linewidth = 0.8) +
  annotate("text",
           x = plateau_pt + 10,
           y = max(oob_only$Error),
           label = "Plateau start (~300 trees)",
           color = "darkgreen",
           hjust = 0, vjust = 1.2,
           fontface = "bold") +
  
  # 최소Tree 표시
  annotate("point", x = best_tree, y = best_error,
           color = "blue", size = 3) +
  geom_vline(xintercept = best_tree,
             color = "blue", linetype = "dashed") +
  annotate("text",
           x = best_tree + 20,
           y = best_error,
           label = paste0("최소 tree = ", best_tree,
                          "\nOOB Error = ", round(best_error, 4)),
           color = "blue", hjust = 0, vjust = -1.2,
           fontface = "bold") +
  
  labs(title = "Random Forest: 트리 수에 따른 OOB Error",
       x = "Number of Trees",
       y = "OOB Error Rate") +
  theme_light() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    axis.title = element_text(face = "bold")
  )

cat(" OOB Error 최소 트리 수:", best_tree, "\n")
cat(" 최소 OOB Error:", round(best_error, 4), "\n")

## (3-3) 교차검증 기반 Random Forest (caret)
# 목적: mtry 최적값 찾기
set.seed(123)
fit_rf <- train(
  Churn ~ ., 
  data       = train, 
  method     = "rf",
  trControl  = ctrl,                                # 5-fold CV
  tuneGrid   = expand.grid(mtry = 1:8),             # mtry 1~8 탐색
  importance = TRUE,
  ntree      = 300                                   # CV용 트리 수
)
print(fit_rf) # 튜닝 결과

## (3-4) CV 성능 확인
fit_rf$resample
rf_cv_acc <- mean(fit_rf$resample$Accuracy); rf_cv_acc # 0.934
rf_cv_kap <- mean(fit_rf$resample$Kappa); rf_cv_kap # 0.702

## (3-5) mtry 튜닝 결과 시각화
rf_df <- fit_rf$results

# 최고 Accuracy 위치
best_idx  <- which.max(rf_df$Accuracy)
best_mtry <- rf_df$mtry[best_idx]
best_acc  <- rf_df$Accuracy[best_idx]

ggplot(rf_df, aes(x = mtry, y = Accuracy)) +
  geom_line(color = "#2E86C1", linewidth = 1) +
  geom_point(color = "#2E86C1", size = 3) +
  
  # Best mtry 표시
  geom_vline(xintercept = best_mtry,
             color = "red", linetype = "dashed", linewidth = 0.8) +
  annotate("text",
           x = best_mtry + 0.2,
           y = best_acc,
           label = paste0("Best mtry = ", best_mtry,
                          "\nAcc = ", round(best_acc, 4)),
           color = "red", fontface = "bold",
           hjust = 0, vjust = -0.8) +
  
  scale_x_continuous(breaks = seq(min(rf_df$mtry), max(rf_df$mtry), by = 2)) +
  labs(
    title = "Random Forest mtry 튜닝 결과",
    x = "mtry (랜덤 변수 개수)",
    y = "Accuracy (교차검증)"
  ) +
  theme_light() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    axis.title = element_text(face = "bold")
  )

## (3-6) Test 데이터 예측 및 성능 평가
pred_rf_cv <- predict(fit_rf, newdata = test)
cm_rf <- confusionMatrix(pred_rf_cv, test$Churn)
acc_rf <- confusionMatrix(pred_rf_cv, test$Churn)$overall["Accuracy"]

cm_rf  # 혼동행렬
acc_rf # 0.934


#-----------------------------------------------------------------------
# 4. 랜덤포레스트 결과 요약
#-----------------------------------------------------------------------
data_rf <- data.frame(
  # CV 성능 (resample 이용)
  CV_Accuracy   = mean(fit_rf$resample$Accuracy),
  CV_Acc_SD     = sd(fit_rf$resample$Accuracy),
  CV_Kappa      = mean(fit_rf$resample$Kappa),
  CV_Kappa_SD   = sd(fit_rf$resample$Kappa),
  
  # Test 성능 (confusionMatrix 이용)
  Test_Accuracy = cm_rf$overall["Accuracy"],
  Test_Kappa    = cm_rf$overall["Kappa"],
  Sensitivity   = cm_rf$byClass["Sensitivity"],
  Specificity   = cm_rf$byClass["Specificity"],
  Precision     = cm_rf$byClass["Precision"],
  Balanced_Acc  = cm_rf$byClass["Balanced Accuracy"],
  F1 = cm_rf$byClass["F1"]
)
rownames(data_rf) <- "rf"
data_rf


#-----------------------------------------------------------------------
# 5. 변수 중요도 시각화
#-----------------------------------------------------------------------
library(randomForest)
imp <- importance(fit_rf$finalModel)
imp_df <- data.frame(
  Variable = rownames(imp),
  MeanDecreaseGini = imp[, "MeanDecreaseGini"]
)

# 변수 중요도를 높은 순으로 정렬
imp_df <- imp_df[order(imp_df$MeanDecreaseGini, decreasing = TRUE), ]

# ggplot 시각화
ggplot(imp_df, aes(x = reorder(Variable, MeanDecreaseGini),
                   y = MeanDecreaseGini,
                   fill = MeanDecreaseGini)) +    # 중요도 기반 색상
  geom_col(color = "black", linewidth = 0.3) +
  coord_flip() +
  scale_fill_gradient(
    low = "#D6EAF8",      # 낮은 값 → 연한 파랑
    high = "#2E86C1"      # 높은 값 → 진한 파랑
  ) +
  labs(
    title = "Random Forest: 변수 중요도",
    x = "Variables",
    y = "Mean Decrease Gini",
    fill = "Importance"   # 범례 제목
  ) +
  theme_light() +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    axis.title = element_text(face = "bold"),
    legend.position = "none"
  )



