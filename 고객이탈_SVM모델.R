#=============================================================================
# 고객 이탈 예측 : 서포트 벡터 머신(SVM) 모델
# -> 두 클래스를 가장 넓은 마진(margin)으로 구분하는 최적의 결정경계를 찾는 모델
# 수정: 25-11-21 14:40
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
# 3. 서보트 벡터 머신
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
## (3-1) 선형 커널 SVM
#-----------------------------------------------------------------------
## (3-1-1) 선형 SVM 튜닝 그리드 설정
grid_linear <- expand.grid(
  C = 10 ^ seq(-1, 1, by = 1)
)

## (3-1-2) 선형 SVM 모델 학습 (5-fold CV 기반)
set.seed(123)
fit_svm_linear <- train(
  Churn ~ .,
  data       = train,
  method     = "svmLinear",
  trControl  = ctrl,
  preProcess = c("center", "scale"),
  tuneGrid   = grid_linear,
  metric     = "Accuracy"
)
fit_svm_linear
fit_svm_linear$results
# 정확도 0.8555888 동일

## (3-1-3) Test 데이터 예측 및 성능 평가
svm_linear_pred <- predict(fit_svm_linear, newdata = test)

cm_svm_linear <- confusionMatrix(svm_linear_pred, test$Churn)
acc_svm_linear <- cm_svm_linear$overall["Accuracy"]

cm_svm_linear  # 혼동행렬
acc_svm_linear # 0.853


#-----------------------------------------------------------------------
## (3-2) RBF 커널 SVM
#-----------------------------------------------------------------------
## (3-2-1) RBF SVM 튜닝 그리드 설정
grid_rbf <- expand.grid(
  sigma = 10 ^ seq(-2, 1, by = 1),   # gamma 역할
  C     = 10 ^ seq(-1, 1, by = 1)
)

## (3-2-2) RBF SVM 모델 학습 (5-fold CV 기반)
set.seed(123)
fit_svm_rbf <- train(
  Churn ~ .,
  data       = train,
  method     = "svmRadial",
  trControl  = ctrl,
  preProcess = c("center", "scale"),
  tuneGrid   = grid_rbf,
  metric     = "Accuracy"
)
fit_svm_rbf          # 튜닝 결과 + CV Accuracy, Kappa 요약
fit_svm_rbf$bestTune # 최적 C, sigma 조합

## (3-2-3) CV 성능 요약
svm_cv_rbf_result <- fit_svm_rbf$results[
  fit_svm_rbf$results$C     == fit_svm_rbf$bestTune$C &
    fit_svm_rbf$results$sigma == fit_svm_rbf$bestTune$sigma,
]

svm_cv_rbf_result   # 최적 파라미터에서의 CV Accuracy, Kappa, SD 등
# CV 정확도 0.9145, kappa 0.59

## (3-2-4) Test 데이터 예측 및 성능 평가
svm_rbf_pred <- predict(fit_svm_rbf, newdata = test)

cm_svm_rbf <- confusionMatrix(svm_rbf_pred, test$Churn)
acc_svm_rbf <- cm_svm_rbf$overall["Accuracy"]

cm_svm_rbf  # 혼동행렬
acc_svm_rbf # 0.9175412

#-----------------------------------------------------------------------
## (3-3) 다항 커널 SVM
#-----------------------------------------------------------------------
## (3-3-1) 다항 SVM 튜닝 그리드 설정
grid_poly <- expand.grid(
  degree = c(2, 3, 4),
  scale  = 1,                          # gamma 비슷한 스케일, 여기서는 1로 고정
  C      = 10 ^ seq(-1, 1, by = 1)
)

## (3-3-2) 다항 SVM 모델 학습 (5-fold CV 기반)
# (주의) 연산량 더 많음
set.seed(123)
fit_svm_poly <- train(
  Churn ~ .,
  data       = train,
  method     = "svmPoly",
  trControl  = ctrl,
  preProcess = c("center", "scale"),
  tuneGrid   = grid_poly,
  metric     = "Accuracy"
)
fit_svm_poly
fit_svm_poly$bestTune


## (3-3-3) CV 성능 요약
svm_cv_poly_result <- fit_svm_poly$results[
  fit_svm_poly$results$C     == fit_svm_poly$bestTune$C &
    fit_svm_poly$results$degree == fit_svm_poly$bestTune$degree,
]

svm_cv_poly_result   # 최적 파라미터에서의 CV Accuracy, Kappa, SD 등
# CV 정확도 0.9186, kappa 0.64

## (3-3-4) Test 데이터 예측 및 성능 평가
svm_poly_pred <- predict(fit_svm_poly, newdata = test)

cm_svm_poly <- confusionMatrix(svm_poly_pred, test$Churn)
acc_svm_poly <- cm_svm_poly$overall["Accuracy"]

cm_svm_poly  # 혼동행렬
acc_svm_poly # 0.9265367

#-----------------------------------------------------------------------
# 4. 튜닝 그리드 시각화
#-----------------------------------------------------------------------
## (1) Linear SVM 정확도 히트맵
fit_svm_linear$results # 모두 동일

## (2) RBF SVM 정확도 히트맵
# caret 결과 불러오기
df_rbf <- fit_svm_rbf$results

# 시각화를 위해 factor로 변환
df_rbf$C     <- factor(df_rbf$C)
df_rbf$sigma <- factor(df_rbf$sigma)

ggplot(df_rbf, aes(x = C, y = sigma, fill = Accuracy)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = round(Accuracy, 3)), color = "black", size = 4) +
  scale_fill_gradient(low = "#d0e1ff", high = "#0747a6") +
  labs(
    title = "RBF SVM: 정확도 히트맵 (C × Sigma)",
    x = "Cost (C)",
    y = "Gamma (Sigma)",
    fill = "Accuracy"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",  
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.text.x = element_text(hjust = 1)
  )


## (3) Poly SVM 정확도 히트맵
# caret 결과 불러오기
df_poly <- fit_svm_poly$results

# 시각화를 위해 factor로 변환
df_poly$C     <- factor(df_poly$C)
df_poly$degree <- factor(df_poly$degree)

ggplot(df_poly, aes(x = C, y = degree, fill = Accuracy)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = round(Accuracy, 3)), color = "black", size = 4) +
  scale_fill_gradient(low = "#d0e1ff", high = "#0747a6") +
  labs(
    title = "Poly SVM: 정확도 히트맵 (C × Degree)",
    x = "Cost (C)",
    y = "Degree",
    fill = "Accuracy"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",  
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.text.x = element_text(hjust = 1)
  )

#-----------------------------------------------------------------------
# 5. SVM 결과 요약
#-----------------------------------------------------------------------
#-------------------------------------------------------------
# (5-1) 선형 커널 SVM 요약
#-------------------------------------------------------------
data_svm_linear <- data.frame(
  CV_Accuracy   = mean(fit_svm_linear$resample$Accuracy),
  CV_Acc_SD     = sd(fit_svm_linear$resample$Accuracy),
  CV_Kappa      = mean(fit_svm_linear$resample$Kappa),
  CV_Kappa_SD   = sd(fit_svm_linear$resample$Kappa),
  
  Test_Accuracy = cm_svm_linear$overall["Accuracy"],
  Test_Kappa    = cm_svm_linear$overall["Kappa"],
  Sensitivity   = cm_svm_linear$byClass["Sensitivity"],
  Specificity   = cm_svm_linear$byClass["Specificity"],
  Precision     = cm_svm_linear$byClass["Precision"],
  Balanced_Acc  = cm_svm_linear$byClass["Balanced Accuracy"]
)
rownames(data_svm_linear) <- "svm_linear"
data_svm_linear


#-------------------------------------------------------------
# (5-2) RBF 커널 SVM 요약
#-------------------------------------------------------------
data_svm_rbf <- data.frame(
  CV_Accuracy   = mean(fit_svm_rbf$resample$Accuracy),
  CV_Acc_SD     = sd(fit_svm_rbf$resample$Accuracy),
  CV_Kappa      = mean(fit_svm_rbf$resample$Kappa),
  CV_Kappa_SD   = sd(fit_svm_rbf$resample$Kappa),
  
  Test_Accuracy = cm_svm_rbf$overall["Accuracy"],
  Test_Kappa    = cm_svm_rbf$overall["Kappa"],
  Sensitivity   = cm_svm_rbf$byClass["Sensitivity"],
  Specificity   = cm_svm_rbf$byClass["Specificity"],
  Precision     = cm_svm_rbf$byClass["Precision"],
  Balanced_Acc  = cm_svm_rbf$byClass["Balanced Accuracy"]
)
rownames(data_svm_rbf) <- "svm_rbf"
data_svm_rbf

#-------------------------------------------------------------
# (5-3) Poly 커널 SVM 요약
#-------------------------------------------------------------
data_svm_poly <- data.frame(
  # CV 성능 (resample 이용)
  CV_Accuracy   = mean(fit_svm_poly$resample$Accuracy),
  CV_Acc_SD     = sd(fit_svm_poly$resample$Accuracy),
  CV_Kappa      = mean(fit_svm_poly$resample$Kappa),
  CV_Kappa_SD   = sd(fit_svm_poly$resample$Kappa),
  
  # Test 성능 (confusionMatrix 이용)
  Test_Accuracy = cm_svm_poly$overall["Accuracy"],
  Test_Kappa    = cm_svm_poly$overall["Kappa"],
  Sensitivity   = cm_svm_poly$byClass["Sensitivity"],
  Specificity   = cm_svm_poly$byClass["Specificity"],
  Precision     = cm_svm_poly$byClass["Precision"],
  Balanced_Acc  = cm_svm_poly$byClass["Balanced Accuracy"]
)
rownames(data_svm_poly) <- "svm_poly"
data_svm_poly

#-------------------------------------------------------------
# (5-4) 세 가지 SVM 성능 비교 테이블 (CV / Test Accuracy 중심)
#-------------------------------------------------------------
library(reshape2)

svm_compare <- rbind(
  data_svm_linear,
  data_svm_rbf,
  data_svm_poly
)

# CV 정확도와 Test 정확도만 뽑아서 보기 좋게
svm_compare_acc <- svm_compare[, c("CV_Accuracy", "Test_Accuracy")]
svm_compare_acc

svm_compare_acc$Model <- rownames(svm_compare_acc)

df_long <- melt(svm_compare_acc, id.vars = "Model",
                variable.name = "Type",
                value.name = "Accuracy")

# 그래프
ggplot(df_long, aes(x = Model, y = Accuracy, group = Type, color = Type)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 4) +
  geom_text(aes(label = round(Accuracy, 3)), 
            vjust = -0.8, size = 4.2, fontface = "bold") +
  scale_color_manual(values = c("CV_Accuracy" = "#e41a1c",
                                "Test_Accuracy" = "#377eb8"),
                     labels = c("CV Accuracy", "Test Accuracy")) +
  ylim(0.85, 0.93) +
  labs(
    title = "SVM 커널 비교: CV과 Test 정확도",
    x = "SVM Kernel",
    y = "Accuracy",
    color = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.border = element_rect(color = "black", fill = NA),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "top"
  )

#-----------------------------------------------------------------------
# 6. ROC 시각화
#-----------------------------------------------------------------------
library(kernlab)
library(pROC)
library(dplyr)
library(ggplot2)

# 1) 공통 준비
y_true <- test$Churn
x_test <- test[, setdiff(names(test), "Churn")]

kmod_lin  <- fit_svm_linear$finalModel
kmod_rbf  <- fit_svm_rbf$finalModel
kmod_poly <- fit_svm_poly$finalModel

# 2) train 때 사용된 preProcess를 그대로 test에 적용
x_test_lin_pp  <- predict(fit_svm_linear$preProcess, x_test)
x_test_rbf_pp  <- predict(fit_svm_rbf$preProcess,   x_test)
x_test_poly_pp <- predict(fit_svm_poly$preProcess,  x_test)

# 3) decision value (연속형 점수) 추출
score_lin_vec  <- as.numeric(predict(kmod_lin,  as.matrix(x_test_lin_pp),  type = "decision"))
score_rbf_vec  <- as.numeric(predict(kmod_rbf,  as.matrix(x_test_rbf_pp),  type = "decision"))
score_poly_vec <- as.numeric(predict(kmod_poly, as.matrix(x_test_poly_pp), type = "decision"))

# 4) ROC 계산
roc_lin  <- roc(y_true, score_lin_vec)
roc_rbf  <- roc(y_true, score_rbf_vec)
roc_poly <- roc(y_true, score_poly_vec)

# 5) ggplot용 데이터
df_roc <- bind_rows(
  data.frame(
    Kernel = paste0("Linear (AUC = ", round(auc(roc_lin), 3), ")"),
    TPR    = roc_lin$sensitivities,
    FPR    = 1 - roc_lin$specificities
  ),
  data.frame(
    Kernel = paste0("RBF (AUC = ", round(auc(roc_rbf), 3), ")"),
    TPR    = roc_rbf$sensitivities,
    FPR    = 1 - roc_rbf$specificities
  ),
  data.frame(
    Kernel = paste0("Poly (AUC = ", round(auc(roc_poly), 3), ")"),
    TPR    = roc_poly$sensitivities,
    FPR    = 1 - roc_poly$specificities
  )
)

# 6) ROC 시각화
ggplot(df_roc, aes(x = FPR, y = TPR, color = Kernel)) +
  geom_line(linewidth = 1.1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  coord_equal() +
  labs(
    title = "SVM Kernel Comparison: ROC Curve",
    x     = "False Positive Rate (1 - Specificity)",
    y     = "True Positive Rate (Sensitivity)",
    color = "Kernel"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.border = element_rect(color = "black", fill = NA),
    plot.title   = element_text(hjust = 0.5)
  )
