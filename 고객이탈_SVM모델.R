#=============================================================================
# 고객 이탈 예측 : 서포트 벡터 머신(SVM) 모델
# 수정: 25-11-19 23:30
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
set.seed(123)
train_idx <- sample(1:nrow(churn), 0.8 * nrow(churn))
train <- churn[train_idx, ]
test  <- churn[-train_idx, ]

# 종속변수 factor 변환
train$Churn <- factor(train$Churn, levels = c(0,1))
test$Churn  <- factor(test$Churn,  levels = c(0,1))

#-----------------------------------------------------------------------
# 2. 공통 교차검증 설정 (5-fold, 1회 반복)
#-----------------------------------------------------------------------
ctrl <- trainControl(method="cv", number=5)

#-----------------------------------------------------------------------
# 3. 서보트 벡터 머신
#-----------------------------------------------------------------------
set.seed(123)
fit_svm <- train(
  Churn ~ .,
  data      = train,
  method    = "svmRadial",   # RBF 커널 SVM
  trControl = ctrl,
  metric    = "Accuracy",
  preProcess = c("center", "scale"),  # SVM은 스케일링 중요
  tuneLength = 5   # C, sigma를 5개 정도 값으로 자동 튜닝
)

fit_svm          # 튜닝 결과 + CV Accuracy, Kappa 요약
fit_svm$bestTune # 최적 C, sigma 조합

# 4. CV 성능 요약 (필요하면)
svm_cv_result <- fit_svm$results[
  fit_svm$results$C     == fit_svm$bestTune$C &
    fit_svm$results$sigma == fit_svm$bestTune$sigma,
]

svm_cv_result   # 최적 파라미터에서의 CV Accuracy, Kappa, SD 등

# 5. Test 데이터 예측 및 성능 평가
svm_pred <- predict(fit_svm, newdata = test)

cm_svm <- confusionMatrix(svm_pred, test$Churn)
cm_svm

#-----------------------------------------------------------------------
# 결과 요약
#-----------------------------------------------------------------------
data_gbm <- data.frame(
  # CV 성능 (resample 이용)
  CV_Accuracy   = mean(fit_svm$resample$Accuracy),
  CV_Acc_SD     = sd(fit_svm$resample$Accuracy),
  CV_Kappa      = mean(fit_svm$resample$Kappa),
  CV_Kappa_SD   = sd(fit_svm$resample$Kappa),
  
  # Test 성능 (confusionMatrix 이용)
  Test_Accuracy = cm_svm$overall["Accuracy"],
  Test_Kappa    = cm_svm$overall["Kappa"],
  Sensitivity   = cm_svm$byClass["Sensitivity"],
  Specificity   = cm_svm$byClass["Specificity"],
  Precision     = cm_svm$byClass["Precision"],
  Balanced_Acc  = cm_svm$byClass["Balanced Accuracy"]
)
rownames(data_gbm) <- "gbm"
data_gbm