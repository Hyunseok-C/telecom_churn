#=============================================================================
# 고객 이탈 예측 : LDA/QDA 모델
# 수정: 25-11-20 23:55
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
# 3. LDA (선형판별분석)
# - k번째 클래스의 관측치들이 다변량 정규분포를 따름 (평균 벡터, 공통의 공분산행렬)
#-----------------------------------------------------------------------
library(MASS)

## (3-1) LDA + 5-fold CV 학습
set.seed(123)
fit_lda <- train(
  Churn ~ .,
  data      = train,
  method    = "lda",
  trControl = ctrl,
  metric    = "Accuracy"
)

## (3-2) CV 성능 확인
fit_lda$resample
lda_cv_acc <- mean(fit_lda$resample$Accuracy); lda_cv_acc # 0.8503334
lda_cv_kap <- mean(fit_lda$resample$Kappa); lda_cv_kap    # 0.2464754

## (3-3) Test 예측 및 성능평가
# Test 예측
lda_pred <- predict(fit_lda, newdata = test)

# Test 성능 평가
cm_lda <- confusionMatrix(lda_pred, test$Churn)
acc_lda <- cm_lda$overall["Accuracy"]

cm_lda  # 혼동행렬
acc_lda # 0.8530735


#-----------------------------------------------------------------------
# 4. LDA 결과 요약
#-----------------------------------------------------------------------
data_lda <- data.frame(
  # CV 성능 (resample 이용)
  CV_Accuracy   = mean(fit_lda$resample$Accuracy),
  CV_Acc_SD     = sd(fit_lda$resample$Accuracy),
  CV_Kappa      = mean(fit_lda$resample$Kappa),
  CV_Kappa_SD   = sd(fit_lda$resample$Kappa),
  
  # Test 성능 (confusionMatrix 이용)
  Test_Accuracy = cm_lda$overall["Accuracy"],
  Test_Kappa    = cm_lda$overall["Kappa"],
  Sensitivity   = cm_lda$byClass["Sensitivity"],
  Specificity   = cm_lda$byClass["Specificity"],
  Precision     = cm_lda$byClass["Precision"],
  Balanced_Acc  = cm_lda$byClass["Balanced Accuracy"]
)
rownames(data_lda) <- "lda"
data_lda


#-----------------------------------------------------------------------
# 5. QDA (이차판별분석)
# - 각 클래스가 서로 다른 공분산 행렬을 가짐
#-----------------------------------------------------------------------
## (5-1) LDA + 5-fold CV 학습
set.seed(123)
fit_qda <- train(
  Churn ~ .,
  data      = train,
  method    = "qda",
  trControl = ctrl,
  metric    = "Accuracy"
)

## (5-2) CV 성능 확인
fit_qda$resample
qda_cv_acc <- mean(fit_qda$resample$Accuracy); qda_cv_acc # 0.8567075
qda_cv_kap <- mean(fit_qda$resample$Kappa); qda_cv_kap    # 0.4103574

## (5-3) Test 예측 및 성능평가
# Test 예측
qda_pred <- predict(fit_qda, newdata = test)

# Test 성능 평가
cm_qda <- confusionMatrix(qda_pred, test$Churn)
acc_qda  <- cm_qda$overall["Accuracy"]

cm_qda  # 혼동행렬
acc_qda # 0.8695652

#-----------------------------------------------------------------------
# 6. QDA 결과 요약
#-----------------------------------------------------------------------
data_qda <- data.frame(
  # CV 성능 (resample 이용)
  CV_Accuracy   = mean(fit_qda$resample$Accuracy),
  CV_Acc_SD     = sd(fit_qda$resample$Accuracy),
  CV_Kappa      = mean(fit_qda$resample$Kappa),
  CV_Kappa_SD   = sd(fit_qda$resample$Kappa),
  
  # Test 성능 (confusionMatrix 이용)
  Test_Accuracy = cm_qda$overall["Accuracy"],
  Test_Kappa    = cm_qda$overall["Kappa"],
  Sensitivity   = cm_qda$byClass["Sensitivity"],
  Specificity   = cm_qda$byClass["Specificity"],
  Precision     = cm_qda$byClass["Precision"],
  Balanced_Acc  = cm_qda$byClass["Balanced Accuracy"]
)
rownames(data_qda) <- "qda"
data_qda

