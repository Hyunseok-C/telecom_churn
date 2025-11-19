#=============================================================================
# 고객 이탈 예측 : 부스팅 모델
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
# 3. 부스팅
#-----------------------------------------------------------------------
# (3-1) GBM(GBoosting) 튜닝 그리드 설정
grid_gbm <- expand.grid(
  interaction.depth = c(3, 5, 7),     # 트리 깊이(각 트리의 분기 수)
  n.trees           = c(200, 300, 500),   # 트리(부스팅 반복) 수
  shrinkage         = c(0.001, 0.05, 0.01, 0.1),   # learning rate (λ)
  n.minobsinnode    = c(10)            # 리프 노드 최소 관측 수
)

# (3-2) GBM 모델 학습 (5-fold CV 기반 하이퍼파라미터 튜닝)
set.seed(123)
fit_gbm <- train(
  Churn ~ .,
  data      = train,
  method    = "gbm",
  trControl = ctrl,        # 교차검증 설정
  tuneGrid  = grid_gbm,    # 후보 파라미터 그리드
  metric    = "Accuracy",  # 정확도 기준으로 비교
  verbose   = FALSE        # gbm 학습 출력 숨김
)

# GBM 학습결과 및 최적 파라미터 확인
fit_gbm
fit_gbm$bestTune

# (3-2) CV 성능확인
fit_gbm$resample
gbm_cv_acc <- mean(fit_gbm$resample$Accuracy); gbm_cv_acc # 
gbm_cv_kap <- mean(fit_gbm$resample$Kappa); gbm_cv_kap # 

# (3-3) Test 데이터 성능 측정
pred_gbm <- predict(fit_gbm, newdata = test)
cm_gbm   <- confusionMatrix(pred_gbm, test$Churn)
cm_gbm

# Test Accuracy 추출
acc_gbm  <- cm_gbm$overall["Accuracy"]
acc_gbm   # 0.9355322

#-----------------------------------------------------------------------
# 결과 요약
#-----------------------------------------------------------------------
data_gbm <- data.frame(
  # CV 성능 (resample 이용)
  CV_Accuracy   = mean(fit_gbm$resample$Accuracy),
  CV_Acc_SD     = sd(fit_gbm$resample$Accuracy),
  CV_Kappa      = mean(fit_gbm$resample$Kappa),
  CV_Kappa_SD   = sd(fit_gbm$resample$Kappa),
  
  # Test 성능 (confusionMatrix 이용)
  Test_Accuracy = cm_gbm$overall["Accuracy"],
  Test_Kappa    = cm_gbm$overall["Kappa"],
  Sensitivity   = cm_gbm$byClass["Sensitivity"],
  Specificity   = cm_gbm$byClass["Specificity"],
  Precision     = cm_gbm$byClass["Precision"],
  Balanced_Acc  = cm_gbm$byClass["Balanced Accuracy"]
)
rownames(data_gbm) <- "gbm"
data_gbm

#-----------------------------------------------------------------------
# 4.(추가) XGBoost (부스팅)
#-----------------------------------------------------------------------
library(xgboost)
#-----------------------------------------------------------------------
grid_xgb <- expand.grid(
  nrounds = c(200, 300, 500),
  max_depth = c(3, 5, 7),
  eta = c(0.05, 0.01, 0.1),
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

set.seed(123)
fit_xgb <- train(
  Churn ~ .,
  data = train,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = grid_xgb,
  metric = "Accuracy"
)
fit_xgb
fit_xgb$bestTune

## CV 성능 평가
fit_xgb$resample
xgb_cv_acc <- mean(fit_xgb$resample$Accuracy); xgb_cv_acc # 0.9388614
xgb_cv_kap <- mean(fit_xgb$resample$Kappa); xgb_cv_kap # 0.723382

# 테스트 데이터 예측 및 정확도
pred_xgb <- predict(fit_xgb, newdata = test)
cm_xgb   <- caret::confusionMatrix(pred_xgb, test$Churn)
acc_xgb  <- cm_xgb$overall["Accuracy"]

cm_xgb  # 혼동행렬
acc_xgb # 0.9415292

#-----------------------------------------------------------------------
# 결과 요약
#-----------------------------------------------------------------------
data_xgb <- data.frame(
  # CV 성능 (resample 이용)
  CV_Accuracy   = mean(fit_xgb$resample$Accuracy),
  CV_Acc_SD     = sd(fit_xgb$resample$Accuracy),
  CV_Kappa      = mean(fit_xgb$resample$Kappa),
  CV_Kappa_SD   = sd(fit_xgb$resample$Kappa),
  
  # Test 성능 (confusionMatrix 이용)
  Test_Accuracy = cm_xgb$overall["Accuracy"],
  Test_Kappa    = cm_xgb$overall["Kappa"],
  Sensitivity   = cm_xgb$byClass["Sensitivity"],
  Specificity   = cm_xgb$byClass["Specificity"],
  Precision     = cm_xgb$byClass["Precision"],
  Balanced_Acc  = cm_xgb$byClass["Balanced Accuracy"]
)
rownames(data_xgb) <- "xgb"
data_xgb

