#=============================================================================
# 고객 이탈 예측 : 로지스틱 모델
# 수정: 25-11-20 22:55
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
# 3. 로지스틱
#-----------------------------------------------------------------------
## (3-1) 로지스틱 회귀 + 5-fold CV 학습
fit_logit <- train(
  Churn ~ .,
  data      = train,
  method    = "glm",          # 로지스틱 회귀
  family    = "binomial",
  trControl = ctrl,
  metric    = "Accuracy"      # 기본: Accuracy + Kappa
)
fit_logit

## (3-2) CV 성능확인
fit_logit$resample
logit_cv_acc <- mean(fit_logit$resample$Accuracy); logit_cv_acc # 0.8608386
logit_cv_kap <- mean(fit_logit$resample$Kappa); logit_cv_kap    # 0.2182808

## (3-3) Test 예측 및 성능평가
# Test 데이터 예측
pred_class <- predict(fit_logit, newdata = test)  # 분류(0/1)

# 혼동행렬
cm_logit <- confusionMatrix(pred_class, test$Churn)
acc_logit <- cm_logit$overall["Accuracy"]

cm_logit  # 혼동행렬
acc_logit # 0.8650675


#-------------------------------------------------------------
# 3. ROC 시각화
#-------------------------------------------------------------
library(pROC)
library(dplyr)

# 1) 예측 확률 얻기
pred_prob <- predict(fit_logit, newdata = test, type = "prob")[, 2]

# 2) ROC 객체 생성
roc_logit <- roc(test$Churn, pred_prob, levels = c("0", "1"), direction = "<")

# 3) ROC 데이터프레임 변환 (ggplot용)
roc_df <- data.frame(
  fpr = 1 - roc_logit$specificities,
  tpr = roc_logit$sensitivities
)

auc_val <- auc(roc_logit)

ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(color = "#FF6666", linewidth = 1.4) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
  
  labs(
    title = paste0("ROC Curve (Logistic Regression)\nAUC = ", round(auc_val, 4)),
    x = "1 - Specificity (False Positive Rate)",
    y = "Sensitivity (True Positive Rate)"
  ) +
  
  theme_light() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

#-------------------------------------------------------------
# 3. 로지스틱 결과 요약
#-------------------------------------------------------------
data_logit <- data.frame(
  # CV 성능 (resample 이용)
  CV_Accuracy   = mean(fit_logit$resample$Accuracy),
  CV_Acc_SD     = sd(fit_logit$resample$Accuracy),
  CV_Kappa      = mean(fit_logit$resample$Kappa),
  CV_Kappa_SD   = sd(fit_logit$resample$Kappa),
  
  # Test 성능 (confusionMatrix 이용)
  Test_Accuracy = cm_logit$overall["Accuracy"],
  Test_Kappa    = cm_logit$overall["Kappa"],
  Sensitivity   = cm_logit$byClass["Sensitivity"],
  Specificity   = cm_logit$byClass["Specificity"],
  Precision     = cm_logit$byClass["Precision"],
  Balanced_Acc  = cm_logit$byClass["Balanced Accuracy"]
)
rownames(data_logit) <- "logit"
data_logit
